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
import json
import numpy as np
import librosa
import noisereduce as nr
from scipy.signal import butter, filtfilt

device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title=f"DubSync ({device})", layout="wide")

demo_dir = "demos"
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
api_version = os.getenv("OPENAI_API_VERSION")  # or latest supported version
# Your Azure OpenAI deployment name
DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")

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
                            f"--device={device}", audio_path], capture_output=True, text=True)
            return output_dir
    except Exception as e:
        st.error(f"Audio separation failed: {e}")
        clean_up()
        sys.exit(1)


def transcribe_audio(audio_path):
    # TODO: Add support for multiple languages
    # TODO: Try faster_whisper for faster transcription
    fp16 = device == "cuda"
    with st.spinner("🎤 Transcribing vocals..."):
        model = whisper.load_model(selected_model, device=device)
        result = model.transcribe(
            audio_path,
            language=input_language_value,
            word_timestamps=False,
            fp16=fp16,
            condition_on_previous_text=True,
            verbose=True,
            task="transcribe"
        )

        # Store raw segments in CSV
        segments = result["segments"]
        print(f"Transcription segments: {segments}")
        return segments


def generate_audio_from_segments(segments, original_audio_path):
    with st.spinner("🌍 Generating audio from segments..."):
        translated_audio = os.path.join(temp_folder, "translated_audio.wav")
        final_audio = AudioSegment.silent(duration=0)
        last_end_time = 0
        for segment in segments:
            idx = segment["id"]
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


def translate_with_gpt(segments):
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
            You are an expert anime, movie, series dubbing scriptwriter.
            Your job is to take a list of {input_segments} from a {input_language.lower()} video and rewrite the {output_language} translations to sound natural, expressive, and emotionally aligned with how the lines would be spoken in an English dub.

            Each segment contains:
            - `start` and `end` timestamps
            - `id`: The segment ID
            - `text`: The original {input_language.lower()} line

            Your goal:
            - Translate the {input_language.lower()} line into {output_language.lower()}.
            - Make small natural rewrites to the {output_language.lower()} translation.
            - Add filler sounds like "uh", "hmm", "ahh", or stuttering where it fits the character's tone.
            - Preserve emotional nuance (anger, sarcasm, nervousness, etc).
            - Keep the rewritten line short enough to be spoken within the original segment’s timing (`end`-`start`).

            Always consider **previous lines** and **what comes next**, and ensure the dialogue flows naturally across segments.

            Return the rewritten translation line and id in json format, and make sure to start the respond with 'Here's the rewritten script for the dubbing:'
            """

        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system",
                    "content": "You are a professional anime, movie, series dubbing scriptwriter."},
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
        "--speed", "0.8"
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
    # crop_audio_futures = []
    with st.spinner("🔊 Cropping audio segments..."):
        for segment in segments:
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            cropped = audio[start_ms:end_ms]
            cropped.export(os.path.join(cropped_audio_dir,
                           f"cropped_{segment['id']}.wav"), format="wav")
            print(f"cropped_{segment['id']}.wav", flush=True)
        # with ThreadPoolExecutor(max_workers=max_threads) as executor:
        #     for segment in segments:
        #         start_ms = int(segment["start"] * 1000)
        #         end_ms = int(segment["end"] * 1000)
        #         cropped = audio[start_ms:end_ms]
        #         output_file = os.path.join(
        #             cropped_audio_dir, f"cropped_{segment['id']}.wav")
        #         crop_audio_futures.append(executor.submit(
        #             cropped.export, output_file, format="wav"))
        # # Wait for all cropping tasks to complete
        # for future in as_completed(crop_audio_futures):
        #     try:
        #         future.result()  # This will raise an exception if the export failed
        #     except Exception as e:
        #         st.error(f"Error cropping audio: {e}")
        #         clean_up()
        #         sys.exit(1)
        #         return None
    with st.spinner("🤖 Cloning voice..."):
        try:
            # with ThreadPoolExecutor(max_workers=max_threads) as executor:
            #     futures = []
            #     for segment in segments:
            #         ref_audio = os.path.join(
            #             cropped_audio_dir, f"cropped_{segment['id']}.wav")
            #         ref_text = segment["text"]
            #         gen_text = segment["translation"]
            #         # Check if ref_text and gen_text are not empty
            #         if not ref_text or not gen_text:
            #             print(
            #                 f"Skipping segment {segment} due to empty ref_text or gen_text")
            #             continue
            #         if gen_text:
            #             future = executor.submit(run_f5_tts_infer, "F5TTS_v1_Base",
            #                                      ref_audio, ref_text, gen_text,
            #                                      output_dir=cloned_audio_dir,
            #                                      output_file=f"output_{segment['id']}.wav")
            #             futures.append(future)
            #     for future in as_completed(futures):
            #         output = future.result()
            #         print(f"F5 TTS Infer output: {output}")
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
                run_f5_tts_infer("F5TTS_v1_Base", ref_audio, ref_text, gen_text,
                                 output_dir=cloned_audio_dir, output_file=f"output_{segment['id']}.wav")
        except Exception as e:
            st.error(f"Voice cloning failed: {e}")
            clean_up()
            sys.exit(1)
            return None


def set_processing(value=True):
    st.session_state.is_processing = value


def generate_srt_subtitles(segments, output_path, subtitle_type="translation"):
    """
    Generate SRT subtitle file from segments
    subtitle_type: "translation" for dubbed text, "original" for original text
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as srt_file:
            for i, segment in enumerate(segments, 1):
                start_time = segment["start"]
                end_time = segment["end"]

                # Convert seconds to SRT time format (HH:MM:SS,mmm)
                start_srt = seconds_to_srt_time(start_time)
                end_srt = seconds_to_srt_time(end_time)

                # Choose text based on subtitle type
                if subtitle_type == "translation" and "translation" in segment:
                    text = segment["translation"]
                elif "text" in segment:
                    text = segment["text"]
                else:
                    text = "..."  # Fallback for empty segments

                # Clean up text for SRT format
                text = text.replace('\n', ' ').strip()
                if not text:
                    continue  # Skip empty segments

                # Write SRT entry
                srt_file.write(f"{i}\n")
                srt_file.write(f"{start_srt} --> {end_srt}\n")
                srt_file.write(f"{text}\n\n")

        return output_path
    except Exception as e:
        st.error(f"Failed to generate subtitles: {e}")
        return None


def seconds_to_srt_time(seconds):
    """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def load_demo_data():
    """Load demo data from demo.json file"""
    demo_json_path = os.path.join(demo_dir, "demo.json")
    try:
        with open(demo_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Demo file not found: {demo_json_path}")
        return []
    except json.JSONDecodeError as e:
        st.error(f"Error reading demo file: {e}")
        return []


def clean_audio_for_transcription(audio_path, output_path=None, noise_reduction_strength=0.8, enable_cleaning=True):
    """
    Clean and preprocess audio for better Whisper transcription quality.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path for cleaned audio output (optional)
        noise_reduction_strength: Strength of noise reduction (0.1 to 1.0)
        enable_cleaning: Whether to apply cleaning or return original
    
    Returns:
        Path to cleaned audio file
    """
    if not enable_cleaning:
        return audio_path
        
    with st.spinner("🧹 Cleaning audio for better transcription..."):
        try:
            # Load audio with librosa for better preprocessing
            audio_data, sample_rate = librosa.load(audio_path, sr=16000)  # Whisper prefers 16kHz
            
            # 1. Noise reduction using spectral gating
            # First, estimate noise from the first 1 second (assuming it contains background noise)
            noise_sample_duration = min(1.0, len(audio_data) / sample_rate * 0.1)  # 10% or 1 sec max
            noise_sample_size = int(noise_sample_duration * sample_rate)
            
            if len(audio_data) > noise_sample_size:
                # Use noisereduce if available, otherwise use simple high-pass filter
                try:
                    import noisereduce as nr
                    audio_data = nr.reduce_noise(
                        y=audio_data, 
                        sr=sample_rate,
                        stationary=False,  # Non-stationary noise reduction
                        prop_decrease=noise_reduction_strength  # Use configurable strength
                    )
                except ImportError:
                    # Fallback: Simple high-pass filter to remove low-frequency noise
                    nyquist = sample_rate * 0.5
                    low_cutoff = 80  # Remove frequencies below 80Hz
                    low_cutoff_normalized = low_cutoff / nyquist
                    b, a = butter(5, low_cutoff_normalized, btype='high')
                    audio_data = filtfilt(b, a, audio_data)
                    print("noisereduce not available, using high-pass filter instead")
            
            # 2. Normalize audio levels
            # RMS normalization to -20dB to prevent clipping while maintaining dynamics
            rms = np.sqrt(np.mean(audio_data**2))
            if rms > 0:
                target_rms = 10**(-20/20)  # -20dB
                audio_data = audio_data * (target_rms / rms)
            
            # 3. Remove silence and very quiet parts (optional, be careful not to remove speech pauses)
            # Use librosa's voice activity detection
            intervals = librosa.effects.split(
                audio_data, 
                top_db=30,  # Threshold for silence detection
                frame_length=2048,
                hop_length=512
            )
            
            # 4. Apply gentle compression to even out volume levels
            # Simple soft limiting
            threshold = 0.95
            audio_data = np.where(
                np.abs(audio_data) > threshold,
                np.sign(audio_data) * (threshold + (np.abs(audio_data) - threshold) * 0.1),
                audio_data
            )
            
            # 5. Final normalization to prevent clipping
            max_amplitude = np.max(np.abs(audio_data))
            if max_amplitude > 0.95:
                audio_data = audio_data * (0.95 / max_amplitude)
            
            # Save cleaned audio
            if output_path is None:
                output_path = audio_path.replace('.wav', '_cleaned.wav')
            
            # Convert back to pydub AudioSegment for consistency with your existing code
            audio_int16 = (audio_data * 32767).astype(np.int16)
            cleaned_audio = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # 16-bit
                channels=1
            )
            
            cleaned_audio.export(output_path, format="wav")
            print(f"Audio cleaned and saved to: {output_path}")
            return output_path
            
        except Exception as e:
            st.warning(f"Audio cleaning failed, using original audio: {e}")
            return audio_path  # Return original if cleaning fails


def enhance_vocals_separation(demucs_output_dir):
    """
    Further enhance the separated vocals for better transcription.
    """
    try:
        separated_dir = os.path.join(
            demucs_output_dir, "htdemucs", 
            os.path.splitext(os.path.basename(audio_path))[0]
        )
        vocals_path = os.path.join(separated_dir, "vocals.wav")
        
        if os.path.exists(vocals_path):
            enhanced_vocals_path = vocals_path.replace('.wav', '_enhanced.wav')
            return clean_audio_for_transcription(vocals_path, enhanced_vocals_path)
        else:
            return vocals_path
    except Exception as e:
        st.warning(f"Vocal enhancement failed: {e}")
        return vocals_path


def analyze_audio_quality(audio_path, label="Audio"):
    """
    Analyze audio quality metrics for debugging and optimization.
    """
    try:
        import librosa
        audio_data, sample_rate = librosa.load(audio_path, sr=None)
        
        # Calculate basic metrics
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.max(np.abs(audio_data))
        snr_estimate = 20 * np.log10(rms / (np.std(audio_data) + 1e-10))
        
        # Dynamic range
        dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
        
        metrics = {
            "Sample Rate": f"{sample_rate} Hz",
            "Duration": f"{len(audio_data) / sample_rate:.2f} seconds",
            "RMS Level": f"{20 * np.log10(rms + 1e-10):.2f} dB",
            "Peak Level": f"{20 * np.log10(peak + 1e-10):.2f} dB",
            "Estimated SNR": f"{snr_estimate:.2f} dB",
            "Dynamic Range": f"{dynamic_range:.2f} dB"
        };
        
        st.write(f"**{label} Quality Metrics:**");
        for metric, value in metrics.items():
            st.write(f"- {metric}: {value}");
            
        return metrics
    except Exception as e:
        st.warning(f"Could not analyze audio quality: {e}")
        return {}


# Main Streamlit app
if __name__ == "__main__":
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    with st.sidebar:
        st.write("Language Options")
        input_languages = [
            ("Japanese", "ja"), ("English", "en"), ("Chinese", "zh"),
            ("Korean", "ko"), ("Spanish", "es"), ("French", "fr"),
            ("German", "de"), ("Italian", "it"), ("Portuguese", "pt"),
            ("Russian", "ru"), ("Arabic", "ar"), ("Hindi", "hi")
        ]
        output_languages = [("English", "en"), ("Chinese", "zh")]

        input_language = st.selectbox(
            "Select the language of the original video",
            options=[label for label, value in input_languages],
            index=0,
            disabled=st.session_state.is_processing,
        )
        input_language_value = dict(input_languages)[input_language]
        st.write("Output Language")
        output_language = st.selectbox(
            "Select the language of the dubbed video",
            options=[label for label, value in output_languages],
            index=0,
            disabled=st.session_state.is_processing,
        )
        output_language_value = dict(output_languages)[output_language]
        st.divider()
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

        st.write("Transcription Options")
        whisper_models = ["tiny", "base", "small",
                          "medium", "large", "large-v2", "large-v3", "turbo"]
        selected_model = st.selectbox(
            "Select Whisper model for transcription",
            options=whisper_models,
            index=7,  # Default to 'large'
            disabled=st.session_state.is_processing,
            help="Larger models provide better accuracy but are slower."
        )
        st.divider()
        st.write("Audio Enhancement Options")
        enable_audio_cleaning = st.checkbox(
            "Enable advanced audio cleaning",
            value=True,
            disabled=st.session_state.is_processing,
            help="Applies noise reduction, normalization, and other enhancements for better transcription"
        )
        
        if enable_audio_cleaning:
            noise_reduction_strength = st.slider(
                "Noise reduction strength",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.1,
                disabled=st.session_state.is_processing,
                help="Higher values remove more noise but may affect speech quality"
            )
            
            show_audio_metrics = st.checkbox(
                "Show audio quality metrics",
                value=False,
                disabled=st.session_state.is_processing,
                help="Display technical metrics about audio quality before and after cleaning"
            )
        
        # ...existing code...
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
        input_video_col, output_video_col = st.columns(2)
        st.divider()
        with input_video_col:
            st.subheader("Input Video")
            st.video(video_file, muted=False)
        uploaded_video_path = os.path.join(temp_folder, "uploaded_video.mp4")
        if not os.path.exists(uploaded_video_path):
            with open(uploaded_video_path, "wb") as f:
                f.write(video_file.read())

        # Extract audio from the video
        audio_path = extract_audio_from_video(uploaded_video_path)

        # Run Demucs to separate background audio
        with output_video_col:
            output_dir = separate_audio_layers(audio_path)

        # Display the separated audio files
        separated_dir = os.path.join(
            output_dir, "htdemucs", os.path.splitext(os.path.basename(audio_path))[0])
        vocals_path = os.path.join(separated_dir, "vocals.wav")
        bg_music_path = os.path.join(separated_dir, "other.wav")
        bass_path = os.path.join(separated_dir, "bass.wav")
        drums_path = os.path.join(separated_dir, "drums.wav")
        
        # Clean vocals for better transcription
        with output_video_col:
            if enable_audio_cleaning and 'show_audio_metrics' in locals() and show_audio_metrics:
                st.subheader("Audio Quality Analysis")
                with st.expander("Original Vocals Quality"):
                    analyze_audio_quality(vocals_path, "Original Vocals")
            
            cleaned_vocals_path = clean_audio_for_transcription(
                vocals_path, 
                enable_cleaning=enable_audio_cleaning,
                noise_reduction_strength=noise_reduction_strength if enable_audio_cleaning else 0.8
            )
            
            if enable_audio_cleaning and 'show_audio_metrics' in locals() and show_audio_metrics and cleaned_vocals_path != vocals_path:
                with st.expander("Cleaned Vocals Quality"):
                    analyze_audio_quality(cleaned_vocals_path, "Cleaned Vocals")
            
            segments = transcribe_audio(cleaned_vocals_path)
        with output_video_col:
            # Translate the segments using AI
            segments = translate_with_gpt(segments)
        filtered_segments = [
            {
                "id": s.get("id"),
                "start (sec)": s.get("start"),
                "end (sec)": s.get("end"),
                "translation": s.get("translation"),
                "text": s.get("text"),
                "no_speech_prob": s.get('no_speech_prob', 0),
                "avg_logprob": s.get("avg_logprob", 0),
            }
            for s in segments
        ]
        df = pd.DataFrame(filtered_segments)
        st.subheader("Transcription")
        st.data_editor(df, use_container_width=True,
                       hide_index=True, num_rows="dynamic", disabled=True)
        segments_csv_path = os.path.join(
            sample_output_dir, f"output_segments_{selected_model}_{output_language.lower()}.csv")
        df.to_csv(segments_csv_path, index=False, encoding="utf-8")
        with output_video_col:
            voice_cloning(segments, cleaned_vocals_path)

        translated_audio = generate_audio_from_segments(segments, audio_path)
        video = VideoFileClip(uploaded_video_path)

        translated_audio = AudioFileClip(translated_audio)
        bg_audio = AudioFileClip(bg_music_path)
        drums_audio = AudioFileClip(drums_path)
        bass_audio = AudioFileClip(bass_path)
        with output_video_col:
            with st.spinner("🎬 Generating dubbed video..."):
                # Combine audio tracks
                combined_audio = CompositeAudioClip(
                    [translated_audio, bg_audio, drums_audio, bass_audio])

                video = video.with_audio(combined_audio)
                dubbed_video_file_name = f"dubbed_video_{selected_model}_{output_language.lower()}.mp4"
                subtitle_path = os.path.join(
                    sample_output_dir, f"dubbed_video_{selected_model}_{output_language.lower()}.srt")
                # Embed subtitles in the video
                dubbed_video_path = os.path.join(
                    sample_output_dir, f"dubbed_video_{selected_model}_{output_language.lower()}.mp4")
                generate_srt_subtitles(segments, subtitle_path)
                video.write_videofile(
                    dubbed_video_path,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile=os.path.join(
                        sample_output_dir, "temp-audio.m4a"),  # explicitly inside /tmp
                    remove_temp=True
                )
                st.subheader("Dubbed video")
                st.video(dubbed_video_path, subtitles=subtitle_path)
        end_time = time.time()
        # # Calculate the elapsed time
        elapsed_time = end_time - start_time
        elapsed_minutes = round(elapsed_time / 60, 2)
        # Display the elapsed time
        st.success(f"Processing completed in {elapsed_minutes} mins.")
        video_file = None
        # Clean up temporary files
        clean_up()
    else:
        st.title("🎬 DubSync - AI-Powered Video Dubbing")
        st.markdown("""
        Welcome to DubSync! Transform your videos with AI-powered dubbing that preserves voice characteristics and emotions.
        
        📖 **[Read the Complete Documentation & Setup Guide](https://github.com/gholapeajinkya/DubSync/blob/main/README.md#dubsync)** - Learn about features, installation, requirements, and how it works.
        """)

        st.subheader("📺 Demo Videos")

        # Load demo data from JSON
        demo_data = load_demo_data()

        if demo_data:
            for i, demo in enumerate(demo_data):
                # Create columns for videos
                if demo['videos']:
                    with st.expander(f"Demo {i + 1}", expanded=True):
                        demo_videos = st.columns(len(demo['videos']))

                        for j, video_info in enumerate(demo['videos']):
                            video_id = video_info['id']
                            video_name = video_info['name']

                            # Construct video path based on demo structure
                            video_path = os.path.join(
                                demo_dir, demo['id'], 'videos', video_id, 'video.mp4')

                            with demo_videos[j]:
                                st.markdown(f"**{video_name}**")

                                if os.path.exists(video_path):
                                    st.video(video_path)
                                else:
                                    st.info("Video will appear here")
                           
                        # Create columns for transcriptions if they exist
                        videos_with_transcriptions = [v for v in demo['videos'] if 'transcription_url' in v]
                        
                        if videos_with_transcriptions:
                            st.markdown("### 📊 Transcription Results")
                            transcription_cols = st.columns(len(videos_with_transcriptions))
                            
                            for k, video_info in enumerate(videos_with_transcriptions):
                                video_id = video_info['id']
                                video_name = video_info['name']
                                
                                # Construct transcription path
                                transcription_path = os.path.join(
                                    demo_dir, demo['id'], 'videos', video_id, 'transcription.csv')
                                
                                with transcription_cols[k]:
                                    st.markdown(f"**{video_name} - Transcription**")
                                    
                                    if os.path.exists(transcription_path):
                                        try:
                                            transcription_df = pd.read_csv(transcription_path)
                                            st.dataframe(transcription_df, use_container_width=True, hide_index=True, height=400)
                                        except Exception as e:
                                            st.error(f"Error loading transcription: {e}")
                                    else:
                                        st.info("Transcription will appear here")
