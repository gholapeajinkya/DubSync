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
import sys
import torch

st.set_page_config(page_title="DubSync",layout="wide")

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

device = "cuda" if torch.cuda.is_available() else "cpu"

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

with st.sidebar:
    # File uploader for MP4 video
    video_file = st.file_uploader("Upload an MP4 video file", type=["mp4"], disabled=st.session_state.is_processing)
    st.write("OR")
    video_url = st.text_input(
        "Enter the URL of an MP4 video (Youtube or other)", disabled=st.session_state.is_processing)
    st.divider()

    st.write("Input Language")
    languages = [("Japanese", "ja"), ("English", "en"),
                 ("Chinese", "zh"), ("Korean", "ko"),
                 ("Hindi", "hi"), ("Marathi", "mr"), ("Spanish", "es"),
                 ("French", "fr"), ("German", "de")]

    input_language = st.selectbox(
        "Select the language of the original video",
        options=[label for label, value in languages],
        index=0,
        disabled=st.session_state.is_processing,
    )
    input_language_value = dict(languages)[input_language]
    st.write("Output Language")
    output_language = st.selectbox(
        "Select the language of the dubbed video",
        options=[label for label, value in languages],
        index=1,
        disabled=st.session_state.is_processing,
    )
    output_language_value = dict(languages)[output_language]
    st.divider()
    st.write("Transcription Options")
    whisper_models = ["tiny", "base", "small", "medium", "large"]
    selected_model = st.selectbox(
        "Select Whisper model for transcription",
        options=whisper_models,
        index=4,  # Default to 'large'
        disabled=st.session_state.is_processing,
    )
    use_ai_transcription = st.toggle("Use AI for transcription", value=False, disabled=st.session_state.is_processing)

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

def clean_up():
    shutil.rmtree(temp_folder)
    st.session_state.is_processing = False

def extract_audio_from_video(video_path):
    with st.spinner("🎬 Extracting audio from the video..."):
        try:
            video = VideoFileClip(video_path)
            audio_path = os.path.join(temp_folder, "extracted_audio.wav")
            video.audio.write_audiofile(audio_path)
            return audio_path
        except Exception as e:
            st.error(f"Failed to extract audio: {e}")
            clean_up()
            sys.exit(1)
            return None

def separate_audio_layers(audio_path):
    with st.spinner("🎶 Separating audio layers..."):
        output_dir = os.path.join(temp_folder, "demucs_output")
        subprocess.run(["demucs", "-o", output_dir, f"--device={device}", audio_path])
        return output_dir

def transcribe_audio(audio_path):
    with st.spinner("🧠 Transcribing vocals..."):
        model = whisper.load_model(selected_model, device=device)
        result = model.transcribe(
            audio_path, language=input_language_value, word_timestamps=True, fp16=False, condition_on_previous_text=True)
        return result["segments"]

def google_text_to_speech(text, source_lang="ja", target_lang="en"):
    base_url = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "gtx",
        "sl": source_lang,
        "tl": target_lang,
        "dt": "t",
        "q": text,
    }
    print(f"Google Translate URL: {base_url}?{urllib.parse.urlencode(params)}")
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    response = requests.get(url)

    if response.status_code == 200:
        try:
            result = response.json()
            return "".join([part[0] for part in result[0]])
        except Exception as e:
            print(f"Error parsing translation response: {e}")
            clean_up()
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
                if original_text:
                    translation = google_text_to_speech(
                        original_text, source_lang, target_lang)
                    if translation:
                        segment["translation"] = translation
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
    unwanted_prefix = f"Here's the rewritten script for the dubbing:"
    if text.startswith(unwanted_prefix):
        text = text[len(unwanted_prefix):].strip()
    return text

def translate_with_gpt(segments, source_lang="ja", target_lang="en"):
    with st.spinner(f"Translating segments using AI..."):
        # Rewrite the translation using GPT
        input_segments = []
        for segment in segments:
            input_segments.append({
                "id": segment["id"],
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"]
            })
        prompt = f"""
            You are an expert anime dubbing scriptwriter.
            Your job is to take a list of {input_segments} from a {input_language.lower()} anime and rewrite the {output_language} translations to sound natural, expressive, and emotionally aligned with how the lines would be spoken in an English dub.

            Each segment contains:
            - `start` and `end` timestamps
            - `id`: The segment ID
            - `text`: The original {input_language.lower()} line

            Your goal:
            - Translate the {input_language.lower()} line into {output_language.lower()}.
            - Make small natural rewrites to the {output_language.lower()} translation.
            - Add filler sounds like "uh", "hmm", "ahh", or stuttering where it fits the character's tone.
            - Preserve emotional nuance (anger, sarcasm, nervousness, etc).
            - Keep the rewritten line short enough to be spoken within the original segment’s timing.

            Always consider **previous lines** and **what comes next**, and ensure the dialogue flows naturally across segments.

            Return the rewritten translation line and id in json format, and make sure to start the respond with 'Here's the rewritten script for the dubbing:'
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
        response_segment = clean_response_text(
            response.choices[0].message.content.strip())
        print("==============================================================================================")
        response_segment = clean_response_text(response_segment)
        print(f"translate_with_gpt response => \n{response_segment}")
        ai_segments = ast.literal_eval(str(response_segment))
        translated_audio = os.path.join(temp_folder, "translated_audio.wav")
        final_audio = AudioSegment.silent(duration=0)
        last_end_time = 0
        for idx, (segment, ai_segment) in enumerate(zip(segments, ai_segments)):
            start_ms = segment["start"]*1000
            end_ms = segment["end"]*1000
            duration_ms = (end_ms - start_ms)
            original_text = ai_segment["text"]
            # Translate
            try:
                translation = google_text_to_speech(
                    original_text, source_lang, target_lang)
                if translation:
                    segment["translation"] = translation
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
                spoken += AudioSegment.silent(
                    duration=duration_ms - len(spoken))

            # Add processed speech
            final_audio += spoken
            last_end_time = segment["end"]
            print(
                f"✅ Segment {idx+1}/{len(segments)} | {segment["start"]} {translation}")
        # 5. Save final audio
        final_audio.export(translated_audio, format="wav")
        return translated_audio

# Display the uploaded video
if video_file is not None:
    st.session_state.is_processing = True
    start_time = time.time()
    input_col1, input_col2 = st.columns(2)
    st.divider()
    output_col1, output_col2 = st.columns(2)
    with input_col1:
        st.video(video_file, muted=False)
    uploaded_video_path = os.path.join(temp_folder, "uploaded_video.mp4")
    if not os.path.exists(uploaded_video_path):
        with open(uploaded_video_path, "wb") as f:
            f.write(video_file.read())

    # Extract audio from the video
    audio_path = extract_audio_from_video(uploaded_video_path)

    # Run Demucs to separate background audio
    with input_col2:
        output_dir = separate_audio_layers(audio_path)

    # Display the separated audio files
    separated_dir = os.path.join(
        output_dir, "htdemucs", os.path.splitext(os.path.basename(audio_path))[0])
    vocals_path = os.path.join(separated_dir, "vocals.wav")
    bg_music_path = os.path.join(separated_dir, "other.wav")
    bass_path = os.path.join(separated_dir, "bass.wav")
    drums_path = os.path.join(separated_dir, "drums.wav")
    with input_col2:
        if os.path.exists(vocals_path) and os.path.exists(bg_music_path) and os.path.exists(bass_path) and os.path.exists(drums_path):
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

    # # TODO: Create small segments of the audio files to recognize unique voices and translate them as per gender

    segments = transcribe_audio(vocals_path)

    if (use_ai_transcription):
        # Translate the segments using AI
        translated_audio = translate_with_gpt(
            segments, source_lang=input_language_value, target_lang=output_language_value)
    else:
        # Translate the segments using Google Translate
        translated_audio = google_translate_segments(
            segments, source_lang=input_language_value, target_lang=output_language_value)
    with output_col2:
        st.write("🎤 Translated audio: ")
        st.audio(translated_audio)

    video = VideoFileClip(uploaded_video_path)

    translated_audio = AudioFileClip(translated_audio)
    bg_audio = AudioFileClip(bg_music_path)
    drums_audio = AudioFileClip(drums_path)
    bass_audio = AudioFileClip(bass_path)
    # Combine audio tracks
    combined_audio = CompositeAudioClip(
        [translated_audio, bg_audio, drums_audio, bass_audio])

    video = video.with_audio(combined_audio)
    dubbed_video_path = os.path.join(
        temp_folder, f"dubbed_video_{output_language}_{use_ai_transcription}.mp4")
    video.write_videofile(dubbed_video_path, codec="libx264", audio_codec="aac")
    with output_col1:
        with st.spinner("🎬 Generating dubbed video..."):
            st.video(dubbed_video_path)
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # Display the elapsed time
    st.success(f"Processing completed in {elapsed_time/60:.2f} mins.")
    # Clean up temporary files
    clean_up()
    

