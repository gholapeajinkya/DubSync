import streamlit as st
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
import whisper
from pydub import AudioSegment
import requests
import shutil
import yt_dlp
from openai import AzureOpenAI
from dotenv import load_dotenv
import ast
import time
import sys
import torch
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title=f"DubSync ({device})", layout="wide")

sample_output_dir = "sample_outputs"
temp_folder = "resources"
cropped_audio_dir = os.path.join(temp_folder, "cropped_audio")
cloned_audio_dir = os.path.join(temp_folder, "cloned_audio")
os.makedirs(temp_folder, exist_ok=True)
os.makedirs(cropped_audio_dir, exist_ok=True)
os.makedirs(cloned_audio_dir, exist_ok=True)
os.makedirs(sample_output_dir, exist_ok=True)

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
    try:
        with st.spinner("🎶 Separating audio layers..."):
            output_dir = os.path.join(temp_folder, "demucs_output")
            subprocess.run(["demucs", "-o", output_dir,
                            f"--device={device}", audio_path])
            return output_dir
    except Exception as e:
        st.error(f"Audio separation failed: {e}")
        clean_up()
        sys.exit(1)


def is_model_cached(model_name):
    cache_dir = os.path.expanduser("~/.cache/whisper")
    model_files = [f"{model_name}.pt"]
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in model_files)


def transcribe_audio(audio_path):
    # TODO: Add support for multiple languages
    # TODO: Try faster_whisper for faster transcription
    if not is_model_cached(selected_model):
        message = f"Model {selected_model} not found in cache. It will be downloaded."
    else:
        message = f"Model {selected_model} is already cached. Transcribing vocals..."
    with st.spinner(message):
        model = whisper.load_model(selected_model, device=device)
        result = model.transcribe(
            audio_path, language=input_language_value, word_timestamps=False, fp16=False, condition_on_previous_text=True)
        return result["segments"]


def generate_audio_from_segments(segments, original_audio_path):
    with st.spinner("🌍 Generating audio from segments..."):
        translated_audio = os.path.join(temp_folder, "translated_audio.wav")
        final_audio = AudioSegment.silent(duration=0)
        last_end_time = 0
        for idx, segment in enumerate(segments):
            start_ms = segment["start"]*1000
            end_ms = segment["end"]*1000
            duration_ms = (end_ms - start_ms)

            audio_segment_path = os.path.join(
                cloned_audio_dir, f"output_{idx}.wav")
            spoken = AudioSegment.from_file(audio_segment_path)

            # Add silence before this segment if needed
            gap_duration = start_ms - int(last_end_time * 1000)
            if gap_duration > 0:
                original_audio = AudioSegment.from_file(original_audio_path)
                gap_start = int(last_end_time * 1000)
                gap_end = int(start_ms)
                gap_audio = original_audio[gap_start:gap_end]
                final_audio += gap_audio
                # final_audio += AudioSegment.silent(duration=gap_duration)

            # Clip or pad the TTS output to exactly fit segment duration
            if len(spoken) > duration_ms:
                spoken = spoken[:duration_ms]
            else:
                spoken += AudioSegment.silent(
                    duration=duration_ms - len(spoken))

            # Add processed speech
            final_audio += spoken
            last_end_time = segment["end"]
            print(f"Segment {idx+1}/{len(segments)} | {segment['start']}")
        # 5. Save final audio
        final_audio.export(translated_audio, format="wav")
        return translated_audio


def clean_response_text(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
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
        response_segment = clean_response_text(response_segment)
        print(f"translate_with_gpt response => \n{response_segment}")
        ai_segments = ast.literal_eval(str(response_segment))
        # Map ai_segments by id for quick lookup
        ai_segments_dict = {s["id"]: s for s in ai_segments}
        # Add translated text into segments
        for segment in segments:
            ai_seg = ai_segments_dict.get(segment["id"])
            if ai_seg and "translation" in ai_seg:
                segment["translation"] = ai_seg["translation"]
            elif ai_seg and "text" in ai_seg:  # fallback if key is 'text'
                segment["translation"] = ai_seg["text"]
        return segments


def run_f5_tts_infer(model, ref_audio, ref_text, gen_text, output_dir=None, output_file=None):
    # TODO: Add multilingual support
    command = [
        "f5-tts_infer-cli",
        "--model", model,
        "--ref_audio", ref_audio,
        "--ref_text", ref_text,
        "--gen_text", gen_text,
        "--speed", "0.7"
    ]
    if output_file:
        command += ["--output_file", output_file]
    if output_dir:
        command += ["--output_dir", output_dir]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True)
        print("Command output:", result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        return None


def voice_cloning(segments, audio_path, max_threads=4):
    audio = AudioSegment.from_file(audio_path)
    crop_audio_futures = []
    with st.spinner("🔊 Cropping audio segments..."):
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            for segment in segments:
                start_ms = int(segment["start"] * 1000)
                end_ms = int(segment["end"] * 1000)
                cropped = audio[start_ms:end_ms]
                output_file = os.path.join(
                    cropped_audio_dir, f"cropped_{segment['id']}.wav")
                crop_audio_futures.append(executor.submit(
                    cropped.export, output_file, format="wav"))
        # Wait for all cropping tasks to complete
        for future in as_completed(crop_audio_futures):
            try:
                future.result()  # This will raise an exception if the export failed
            except Exception as e:
                st.error(f"Error cropping audio: {e}")
                clean_up()
                sys.exit(1)
                return None
    with st.spinner("🤖 Cloning voice..."):
        for segment in segments:
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            cropped = audio[start_ms:end_ms]
            cropped.export(os.path.join(cropped_audio_dir,
                           f"cropped_{segment['id']}.wav"), format="wav")
        try:
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = []
                for segment in segments:
                    ref_audio = os.path.join(
                        cropped_audio_dir, f"cropped_{segment['id']}.wav")
                    ref_text = segment["text"]
                    gen_text = segment["translation"]
                    # Check if ref_text and gen_text are not empty
                    if not ref_text or not gen_text:
                        print(
                            f"Skipping segment {segment} due to empty ref_text or gen_text")
                        continue
                    if gen_text:
                        future = executor.submit(run_f5_tts_infer, "F5TTS_v1_Base",
                                                 ref_audio, ref_text, gen_text,
                                                 output_dir=cloned_audio_dir,
                                                 output_file=f"output_{segment['id']}.wav")
                        futures.append(future)
                for future in as_completed(futures):
                    output = future.result()
                    print(f"F5 TTS Infer output: {output}")

        except Exception as e:
            st.error(f"Voice cloning failed: {e}")
            clean_up()
            sys.exit(1)
            return None


def set_processing(value=True):
    st.session_state.is_processing = value


if __name__ == "__main__":
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    with st.sidebar:
        # File uploader for MP4 video
        st.write("Video Input Options")
        video_file = st.file_uploader("Upload a MP4 video file", type=[
            "mp4"], disabled=st.session_state.is_processing, on_change=set_processing, args=(True,))
        st.write("OR")
        video_url = st.text_input(
            "Enter the URL of a MP4 video (Youtube or other)",
            disabled=st.session_state.is_processing,
            on_change=set_processing,  # Will pass value below
            args=(True,)
        )
        st.divider()

        st.write("Language Options")
        languages = [("Japanese", "ja"), ("English", "en"), ("Chinese", "zh")]

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
        whisper_models = ["tiny", "base", "small",
                          "medium", "large", "large-v2", "large-v3", "turbo"]
        selected_model = st.selectbox(
            "Select Whisper model for transcription",
            options=whisper_models,
            index=4,  # Default to 'large'
            disabled=st.session_state.is_processing,
        )

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
                        uploaded_video_path = os.path.join(
                            temp_folder, "uploaded_video.mp4")
                        with open(uploaded_video_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        video_file = open(uploaded_video_path, "rb")
                        st.success("Video downloaded successfully!")
                    else:
                        st.error(
                            "Failed to download video. Please check the URL.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

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
        # Translate the segments using AI
        segments = translate_with_gpt(
            segments, source_lang=input_language_value, target_lang=output_language_value)
        with output_col2:
            filtered_segments = [
                {
                    "id": s.get("id"),
                    "start (sec)": s.get("start"),
                    "end (sec)": s.get("end"),
                    "duration (sec)": (s.get("end") - s.get("start")),
                    "speech_prob": f"{round(round(s.get('no_speech_prob', 0), 2) * 100, 2)}%",
                    "translation": s.get("translation"),
                    "text": s.get("text"),
                }
                for s in segments
            ]
            df = pd.DataFrame(filtered_segments)
            st.data_editor(df, use_container_width=True,
                           hide_index=True, num_rows="dynamic", disabled=True)
            segments_csv_path = os.path.join(
                sample_output_dir, f"output_segments_{selected_model}_{output_language.lower()}.csv")
            df.to_csv(segments_csv_path, index=False, encoding="utf-8")
        with output_col1:
            voice_cloning(segments, vocals_path)

        translated_audio = generate_audio_from_segments(segments, audio_path)
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
            sample_output_dir, f"dubbed_video_{selected_model}_{output_language.lower()}.mp4")
        video.write_videofile(
            dubbed_video_path, codec="libx264", audio_codec="aac")
        with output_col1:
            with st.spinner("🎬 Generating dubbed video..."):
                st.video(dubbed_video_path)
        end_time = time.time()
        # # Calculate the elapsed time
        elapsed_time = end_time - start_time
        elapsed_minutes = round(elapsed_time / 60, 2)
        # Display the elapsed time
        st.success(f"Processing completed in {elapsed_minutes} mins.")
        # Clean up temporary files
        clean_up()
