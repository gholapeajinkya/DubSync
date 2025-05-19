import urllib.parse
import requests
import os
import whisper
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
from gtts import gTTS
from pydub import AudioSegment
import subprocess

video_path = "small_clip.mp4"
audio_path = "original_audio.wav"
output_path = "translated_audio.wav"
temp_folder = "tts_segments"
output_dir = "outputs"

os.makedirs(temp_folder, exist_ok=True)

# 1. Extract audio
print("🎬 Extracting audio...")
video = VideoFileClip(video_path)
video.audio.write_audiofile(audio_path)


# 2. Separate background audio Using Demucs
print("🎶 Separating background audio...")
subprocess.run(["demucs", "-o", output_dir, audio_path])
separated_dir = os.path.join(output_dir, "htdemucs", os.path.splitext(os.path.basename(audio_path))[0])
audio_path = os.path.join(separated_dir, "vocals.wav")
bg_music_path = os.path.join(separated_dir, "other.wav")
drums_music_path = os.path.join(separated_dir, "drums.wav")
bass_music_path = os.path.join(separated_dir, "bass.wav")

bg_music_path = AudioFileClip(bg_music_path)
drums_music_path = AudioFileClip(drums_music_path)
bass_music_path = AudioFileClip(bass_music_path)

# 3. Transcribe with Whisper
print("🧠 Transcribing audio...")
model = whisper.load_model("medium")
result = model.transcribe(audio_path, language="ja", word_timestamps=False, fp16=False)
segments = result["segments"]

# 4. Translate and generate TTS for each segment
print("🌍 Translating and generating speech...")
final_audio = AudioSegment.silent(duration=0)


def google_translate_text(text, source_lang="ja", target_lang="en"):
    base_url = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "gtx",
        "sl": source_lang,
        "tl": target_lang,
        "dt": "t",
        "q": text,
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    response = requests.get(url)

    if response.status_code == 200:
        try:
            result = response.json()
            return "".join([part[0] for part in result[0]])
        except Exception as e:
            print(f"Error parsing translation response: {e}")
    else:
        print("Translation failed with status", response.status_code)
    return ""

for idx, segment in enumerate(segments):
    start = segment["start"]
    end = segment["end"]
    duration_ms = int((end - start) * 1000)
    original_text = segment["text"]

    # Translate
    try:
        translation = google_translate_text(original_text)
    except Exception as e:
        print(f"Translation failed for segment {idx}: {e}")
        translation = "..."

    # TTS
    tts = gTTS(text=translation, lang='en')
    segment_path = os.path.join(temp_folder, f"segment_{idx}.mp3")
    tts.save(segment_path)

    spoken = AudioSegment.from_file(segment_path)
    pad_ms = max(0, duration_ms - len(spoken))
    aligned = spoken + AudioSegment.silent(duration=pad_ms)

    final_audio += aligned

    print(f"✅ Segment {idx+1}/{len(segments)} | {translation}")

# 5. Save final audio
final_audio.export(output_path, format="wav")
print(f"\n✅ Translated audio saved to: {output_path}")

video = VideoFileClip(video_path)

translated_audio = AudioFileClip(output_path)

# Combine audio tracks
final_audio = CompositeAudioClip([translated_audio, bg_music_path, drums_music_path, bass_music_path]) 

video = video.with_audio(final_audio)

video.write_videofile("dubbed_video.mp4", codec="libx264", audio_codec="aac")
print("🎥 Dubbed video saved as dubbed_video.mp4")
