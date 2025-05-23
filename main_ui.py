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
import yt_dlp
from openai import AzureOpenAI
from dotenv import load_dotenv
import ast
import time 

temp_folder = "resources"
os.makedirs(temp_folder, exist_ok=True)

# Load environment variables from .env file
load_dotenv()

# Read the API key from the .env file
api_type = "azure"
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")
api_version = "2025-01-01-preview"  # or latest supported version
DEPLOYMENT_NAME = "gpt-4-dubbing"  # Your Azure OpenAI deployment name

# Create a client using Azure credentials
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=api_base,
)

with st.sidebar:
    # File uploader for MP4 video
    video_file = st.file_uploader("Upload an MP4 video file", type=["mp4"])
    st.write("OR")
    video_url = st.text_input(
        "Enter the URL of an MP4 video (Youtube or other)")

if video_url:
    if "youtube.com" in video_url or "youtu.be" in video_url:
        with st.spinner("Downloading video from URL..."):
            try:
                # Use yt-dlp for YouTube links
                ydl_opts = {
                    'format': 'bestvideo+bestaudio/best',
                    # Output to stdout
                    'outtmpl': os.path.join(temp_folder, "uploaded_video.mp4"),
                    'merge_output_format': 'mp4',  # Merge video and audio into mp4
                    'quiet': False,  # Suppress output
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    video_file = ydl.prepare_filename(info).replace(".webm", ".mp4").replace(
                        # os.path.join(temp_folder, "uploaded_video.mp4")
                        ".mkv", ".mp4")
                st.success("YouTube video downloaded successfully! " +
                           ydl.extract_info(video_url, download=False)['title'])
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        try:
            with st.spinner("Downloading video from URL..."):
                response = requests.get(video_url, stream=True)
                if response.status_code == 200:
                    video_file = response.raw
                    uploaded_video_path = os.path.join(
                        temp_folder, "uploaded_video.mp4")
                    with open(uploaded_video_path, "wb") as f:
                        shutil.copyfileobj(response.raw, f)
                    st.success("Video downloaded successfully!")
                else:
                    st.error("Failed to download video. Please check the URL.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


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
        model = whisper.load_model("large")
        result = model.transcribe(
            audio_path, language="ja", word_timestamps=False, fp16=False)
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
        with st.spinner(f"Translating segments using AI..."):
            # Rewrite the translation using GPT
            input_segments = []
            for segment in segments:
                input_segments.append({
                    "id": segment["id"],
                    "text": segment["text"]
                })
            ai_segments = rewrite_segment_with_gpt(input_segments)
            ai_segments = ast.literal_eval(str(ai_segments))

            for idx, (segment, ai_segment) in enumerate(zip(segments, ai_segments)):
                start_ms = segment["start"]*1000
                end_ms = segment["end"]*1000
                duration_ms = (end_ms - start_ms)
                translation = ai_segment["text"]
                # original_text = segment["text"]
                # # Translate
                # try:
                #     translation = google_translate_text(
                #         original_text, source_lang, target_lang)
                #     if translation:
                #         segment["translation"] = translation
                # except Exception as e:
                #     print(f"Translation failed for segment {idx}: {e}")
                #     translation = "..."

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
                    spoken += AudioSegment.silent(
                        duration=duration_ms - len(spoken))

                # Add processed speech
                final_audio += spoken
                last_end_time = segment["end"]
                print(
                    f"✅ Segment {idx+1}/{len(segments)} | ${segment["start"]} {translation}")
            # 5. Save final audio
            final_audio.export(translated_audio, format="wav")
            return translated_audio


def clean_response_text(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        # Remove the wrapping triple backticks
        st.warning("Removing wrapping triple backticks")
        text = "\n".join(text.strip("`").split("\n")[1:])
    unwanted_prefix = "Here's the rewritten script for the English dubbing:"
    if text.startswith(unwanted_prefix):
        text = text[len(unwanted_prefix):].strip()
    return text


def rewrite_segment_with_gpt(segments):
    prompt = f"""
        You are a dubbing script writer. Rewrite the following translation into a natural, expressive line suitable for English 
        anime dubbing.
        It should capture the tone of the original Japanese line, and based on previous conversation make sense of the next
        line and be spoken within words.
        This is segment object:
        {segments} contains raw text Japanese and English translation iterate over it.
        Return the rewritten line in same format, and make sure to start the respond with 'Here's the rewritten script for the English dubbing:'
        """

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system",
                "content": "You are a professional anime dubbing scriptwriter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    response = response.choices[0].message.content.strip()
    print(response)
    return clean_response_text(response)


# Display the uploaded video
if video_file is not None:
    start_time = time.time()
    st.video(video_file, muted=False)
    uploaded_video_path = os.path.join(temp_folder, "uploaded_video.mp4")
    if not os.path.exists(uploaded_video_path):
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
        col1, col2 = st.columns(2)
        with col1:
            st.write("🎤 Vocals: ")
            st.audio(vocals_path)
            st.write("🎵 Background Music: ")
            st.audio(bg_music_path)
        with col2:
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
    combined_audio = CompositeAudioClip(
        [translated_audio, bg_audio, drums_audio, bass_audio])

    video = video.with_audio(combined_audio)
    dubbed_video_path = os.path.join(temp_folder, "dubbed_video.mp4")
    video.write_videofile(
        dubbed_video_path, codec="libx264", audio_codec="aac")
    st.success("🎥 Dubbed video saved")
    st.video(dubbed_video_path)
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # Display the elapsed time
    st.success(f"Processing completed in {elapsed_time:.2f} seconds.")
    # Clean up temporary files
    shutil.rmtree(temp_folder)
