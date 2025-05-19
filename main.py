import whisper
from transformers import MarianMTModel, MarianTokenizer
from moviepy import VideoFileClip, AudioFileClip
from gtts import gTTS
from tempfile import NamedTemporaryFile
import os
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment

input_video = "small_clip.mp4"
model = whisper.load_model("medium")  # or "medium" for faster results

result = model.transcribe(input_video, language="ja", fp16=False)
# print(result["text"])        # Full Japanese text
# print(result["segments"])      # Includes timestamps

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ja-en")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ja-en")

output_segments = []

for segment in result["segments"]:
    inputs = tokenizer(segment["text"], return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs)
    translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    # segment["translation"] = translation
    target_duration = segment["end"] - segment["start"]
    # Generate speech
    tts = gTTS(text=translation, lang='en')
    with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        tmp_path = tmp_file.name
    
        # Load & convert to numpy
    y, sr = librosa.load(tmp_path, sr=None)
    original_duration = librosa.get_duration(y=y, sr=sr)
    rate = original_duration / target_duration

    # Time stretch to match target duration
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    stretched_file = f"segment_{segment['id']}.wav"
    sf.write(stretched_file, y_stretched, sr)

    output_segments.append(AudioSegment.from_wav(stretched_file))

    os.remove(tmp_path)  # Clean up

# Combine all segments into one final audio file
final_audio = AudioSegment.silent(duration=0)
for segment in output_segments:
    final_audio += segment

output_audio = "final_output_audio.wav"
final_audio.export(output_audio, format="wav")
print(f"✅ Final audio saved as {output_audio}")


video = VideoFileClip(input_video)
audio = AudioFileClip(output_audio)

final_video = video.with_audio(audio)
final_video.write_videofile("final_dubbed_anime.mp4", codec="libx264", audio_codec="aac")

print("Final video saved as final_dubbed_anime.mp4")

