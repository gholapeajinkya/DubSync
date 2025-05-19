import streamlit as st
import os
import subprocess
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
import whisper
from pydub import AudioSegment
import urllib.parse
import requests
from gtts import gTTS
import shutil 

temp_folder = "resources"
os.makedirs(temp_folder, exist_ok=True)
# File uploader for MP4 video
video_file = st.file_uploader("Upload an MP4 video file", type=["mp4"])


def extract_audio_from_video(video_path):
    with st.spinner("🎬 Extracting audio from the video..."):
        video = VideoFileClip(video_path)
        audio_path = os.path.join(temp_folder, "extracted_audio.wav")
        video.audio.write_audiofile(audio_path)
        return audio_path


def separate_audio_layers(audio_path):
    with st.spinner("🎶 Separating audio layers..."):
        output_dir = os.path.join(temp_folder, "demucs_output")
        subprocess.run(["demucs", "-o", output_dir, audio_path])
        return output_dir


def transcribe_audio(audio_path):
    with st.spinner("🧠 Transcribing audio..."):
        model = whisper.load_model("medium")
        result = model.transcribe(
            audio_path, language="ja", word_timestamps=False, fp16=False)
        print(result["segments"])
        return result["segments"]


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


def google_translate_segments(segments, source_lang="ja", target_lang="en"):
    with st.spinner("🌍 Translating ..."):
        translated_audio = os.path.join(temp_folder, "translated_audio.wav")
        final_audio = AudioSegment.silent(duration=0)
        last_end_time = 0
        for idx, segment in enumerate(segments):
            start_ms = segment["start"]*1000
            end_ms = segment["end"]*1000
            duration_ms = (end_ms - start_ms)
            original_text = segment["text"]

            # Translate
            try:
                translation = google_translate_text(original_text, source_lang, target_lang)
            except Exception as e:
                print(f"Translation failed for segment {idx}: {e}")
                translation = "..."

            # TTS
            tts = gTTS(text=translation, lang='en')
            segment_path = os.path.join(temp_folder, f"segment_{idx}.mp3")
            tts.save(segment_path)
            spoken = AudioSegment.from_file(segment_path)

            # Add silence before this segment if needed
            gap_duration = start_ms - int(last_end_time * 1000)
            if gap_duration > 0:
                final_audio += AudioSegment.silent(duration=gap_duration)

            # Clip or pad the TTS output to exactly fit segment duration
            if len(spoken) > duration_ms:
                spoken = spoken[:duration_ms]
            else:
                spoken += AudioSegment.silent(duration=duration_ms - len(spoken))

            # Add processed speech
            final_audio += spoken
            last_end_time = segment["end"]
            # if len(spoken) > duration_ms:
            #     spoken = spoken[:duration_ms]
            # else:
            #     spoken += AudioSegment.silent(duration=(duration_ms - len(spoken)))
            # final_audio += spoken

            print(f"✅ Segment {idx+1}/{len(segments)} | ${segment["start"]} {translation}")

            # 5. Save final audio
        final_audio.export(translated_audio, format="wav")
        print(f"\n✅ Translated audio saved to: {translated_audio}")
        return translated_audio


# Display the uploaded video
if video_file is not None:
    st.video(video_file, muted=False)
    uploaded_video_path = os.path.join(temp_folder, "uploaded_video.mp4")
    with open(uploaded_video_path, "wb") as f:
        f.write(video_file.read())

    # Extract audio from the video
    audio_path = extract_audio_from_video(uploaded_video_path)

    # Run Demucs to separate background audio
    output_dir = separate_audio_layers(audio_path)

    # Display the separated audio files
    separated_dir = os.path.join(
        output_dir, "htdemucs", os.path.splitext(os.path.basename(audio_path))[0])
    vocals_path = os.path.join(separated_dir, "vocals.wav")
    bg_music_path = os.path.join(separated_dir, "other.wav")
    bass_path = os.path.join(separated_dir, "bass.wav")
    drums_path = os.path.join(separated_dir, "drums.wav")

    if os.path.exists(vocals_path) and os.path.exists(bg_music_path) and os.path.exists(bass_path) and os.path.exists(drums_path):
        st.success("Audio separation completed!")
        st.write("🎤 Vocals: ")
        st.audio(vocals_path)
        st.write("🎵 Background Music: ")
        st.audio(bg_music_path)
        st.write("🥁 Drums: ")
        st.audio(drums_path)
        st.write("🎸 Bass: ")
        st.audio(bass_path)

    else:
        st.error("Audio separation failed. Please check the logs.")

    # TODO: Create small segments of the audio files to recognize unique voices and translate them as per gender

    segments = transcribe_audio(vocals_path)
    translated_audio = google_translate_segments(segments)
    st.write("🎤 Translated audio: "); st.audio(translated_audio)

    # Combine audio tracks
    video = VideoFileClip(uploaded_video_path)

    translated_audio = AudioFileClip(translated_audio)
    bg_audio = AudioFileClip(bg_music_path)
    drums_audio = AudioFileClip(drums_path)
    bass_audio = AudioFileClip(bass_path)
    # Combine audio tracks
    combined_audio = CompositeAudioClip([translated_audio, bg_audio, drums_audio, bass_audio])

    video = video.with_audio(combined_audio)
    dubbed_video_path = os.path.join(temp_folder, "dubbed_video.mp4")
    video.write_videofile(dubbed_video_path, codec="libx264", audio_codec="aac")
    st.success("🎥 Dubbed video saved")
    st.video(dubbed_video_path)
    # Clean up temporary files
    shutil.rmtree(temp_folder)
