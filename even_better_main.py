import os
import subprocess
from pydub import AudioSegment
# from TTS.api import TTS

# --- CONFIG ---
video_file = "small_clip.mp4"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# --- 1. Extract Audio ---
audio_path = os.path.join(output_dir, "original_audio.wav")
subprocess.run(["ffmpeg", "-y", "-i", video_file, "-ac", "2", "-ar", "44100", "-vn", audio_path])

# --- 2. Separate background audio Using Demucs ---
subprocess.run(["demucs", "-o", output_dir, audio_path])
separated_dir = os.path.join(output_dir, "htdemucs", os.path.splitext(os.path.basename(audio_path))[0])
bg_music_path = os.path.join(separated_dir, "other.wav")

# # --- 3. Load Segments ---
# with open("segments.json", "r", encoding="utf-8") as f:
#     segments = json.load(f)

# # --- 4. Generate TTS Clips and Stitch Them Together ---
# tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
# final_audio = AudioSegment.silent(duration=int(segments[-1]["end"] * 1000))

# for i, seg in enumerate(segments):
#     text = seg["text"]
#     start_ms = int(seg["start"] * 1000)

#     tts_output_path = os.path.join(output_dir, f"tts_{i}.wav")
#     tts.tts_to_file(text=text, file_path=tts_output_path)

#     tts_clip = AudioSegment.from_wav(tts_output_path)
#     final_audio = final_audio.overlay(tts_clip, position=start_ms)

# final_voice_path = os.path.join(output_dir, "new_voice.wav")
# final_audio.export(final_voice_path, format="wav")

# # --- 5. Mix with Background ---
# bg = AudioSegment.from_wav(bg_music_path)
# voice = AudioSegment.from_wav(final_voice_path)
# mixed = bg.overlay(voice)
# final_audio_path = os.path.join(output_dir, "final_audio.wav")
# mixed.export(final_audio_path, format="wav")

# # --- 6. Replace Audio in Video ---
# final_video_path = os.path.join(output_dir, "dubbed_output.mp4")
# subprocess.run([
#     "ffmpeg", "-y",
#     "-i", video_file,
#     "-i", final_audio_path,
#     "-c:v", "copy",
#     "-map", "0:v:0",
#     "-map", "1:a:0",
#     "-shortest", final_video_path
# ])

# print(f"\n✅ Done! Dubbed video saved to: {final_video_path}")
