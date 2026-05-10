import os
import subprocess
import shutil

from moviepy import AudioFileClip, CompositeAudioClip, VideoFileClip
from state import AgentState, VoiceCloningWorkerState
from pydub import AudioSegment
from langgraph.types import Send


def crop_dialogue_segments(segments, audio, temp_folder):
    """Crop audio for each segment and save to disk."""
    cropped_audio_dir = os.path.join(temp_folder, "cropped_dialogue")
    os.makedirs(cropped_audio_dir, exist_ok=True)
    cloned_audio_dir = os.path.join(temp_folder, "cloned_audio")
    os.makedirs(cloned_audio_dir, exist_ok=True)
    for segment in segments:
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        cropped = audio[start_ms:end_ms]
        cropped.export(os.path.join(cropped_audio_dir,
                       f"cropped_{segment['id']}.wav"), format="wav")
        print(f"cropped_{segment['id']}.wav", flush=True)


def run_f5_tts_infer(model, ref_audio, ref_text, gen_text, output_dir=None, output_file=None):
    """Run F5-TTS inference for voice cloning."""
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


def prepare_voice_cloning_node(state: AgentState) -> AgentState:
    """Prepare audio segments for parallel voice cloning. Crops dialogue segments."""
    audio_path = state.get("vocals_path")
    try:
        segments = state.get("transcription_segments", [])
        audio = AudioSegment.from_file(audio_path)
        temp_folder = state.get("temp_folder")
        crop_dialogue_segments(segments, audio, temp_folder)
        # Initialize cloned_segments as empty list for fan-in collection
        return {**state, "cloned_segments": []}
    except Exception as e:
        return {**state, "error": f"Error preparing voice cloning: {e}"}


def fanout_voice_cloning(state: AgentState) -> list[Send]:
    """Fan-out function that sends each segment to a parallel worker."""
    segments = state.get("transcription_segments", [])
    vocals_path = state.get("vocals_path")
    temp_folder = state.get("temp_folder")

    sends = []
    for segment in segments:
        # Skip segments with empty text or translation
        if not segment.get("text") or not segment.get("translation"):
            print(
                f"Skipping segment {segment.get('id')} due to empty text or translation")
            continue
        sends.append(
            Send("voice_cloning_worker", {
                "segment": segment,
                "vocals_path": vocals_path,
                "temp_folder": temp_folder,
                "cloned_segments": [],
                "error": None
            })
        )
    return sends


def voice_cloning_worker(state: VoiceCloningWorkerState) -> dict:
    """Worker node that clones voice for a single segment."""
    segment = state["segment"]
    temp_folder = state.get("temp_folder")
    cropped_audio_dir = os.path.join(temp_folder, "cropped_dialogue")
    cloned_audio_dir = os.path.join(temp_folder, "cloned_audio")
    os.makedirs(cloned_audio_dir, exist_ok=True)
    try:
        ref_audio = os.path.join(
            cropped_audio_dir, f"cropped_{segment['id']}.wav")
        ref_text = segment["text"]
        gen_text = segment["translation"]
        output_file = f"output_{segment['id']}.wav"

        print(f"Cloning voice for segment {segment['id']}...", flush=True)
        run_f5_tts_infer(
            "F5TTS_v1_Base",
            ref_audio,
            ref_text,
            gen_text,
            output_dir=cloned_audio_dir,
            output_file=output_file
        )

        cloned_path = os.path.join(cloned_audio_dir, output_file)
        return {
            "cloned_segments": [{
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "cloned_path": cloned_path
            }]
        }
    except Exception as e:
        print(f"Error cloning segment {segment['id']}: {e}")
        return {
            "cloned_segments": [{
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "cloned_path": None,
                "error": str(e)
            }]
        }


def combine_cloned_audio_and_generate_video_node(state: AgentState) -> AgentState:
    """Combine all cloned audio segments into a single audio file."""
    try:
        cloned_segments = state.get("cloned_segments", [])
        if not cloned_segments:
            return {**state, "error": "No cloned segments to combine"}

        # Sort segments by start time to ensure correct order
        sorted_segments = sorted(cloned_segments, key=lambda x: x["start"])

        # Get the duration of the original dialogue audio for proper timing
        vocals_path = state.get("vocals_path")
        original_audio = AudioSegment.from_file(vocals_path)
        total_duration_ms = len(original_audio)

        # Create silent audio track of the same duration
        combined = AudioSegment.silent(duration=total_duration_ms)

        # Overlay each cloned segment at its proper position
        for segment in sorted_segments:
            if segment.get("cloned_path") and os.path.exists(segment["cloned_path"]):
                cloned_audio = AudioSegment.from_file(segment["cloned_path"])
                start_ms = int(segment["start"] * 1000)
                combined = combined.overlay(cloned_audio, position=start_ms)
                print(
                    f"Added segment {segment['id']} at {start_ms}ms", flush=True)
            else:
                print(
                    f"Skipping segment {segment['id']} - no cloned audio found")
        temp_folder = state.get("temp_folder")
        cloned_audio_dir = os.path.join(temp_folder, "cloned_audio")
        # Export combined audio
        combined_cloned_audio_path = os.path.join(
            cloned_audio_dir, "combined_cloned_audio.wav")
        combined.export(combined_cloned_audio_path, format="wav")
        print(f"Combined audio saved to {combined_cloned_audio_path}", flush=True)
        video = VideoFileClip(state.get("input_video_path"))
        
        # Build list of audio clips, only including those with valid paths
        audio_clips = []
        translated_audio = AudioFileClip(combined_cloned_audio_path)
        audio_clips.append(translated_audio)
        
        # Add optional audio tracks if they exist
        effects_path = state.get("effects_path")
        if effects_path and os.path.exists(effects_path):
            audio_clips.append(AudioFileClip(effects_path))
            print(f"Added effects audio: {effects_path}", flush=True)
        
        drums_path = state.get("drums_path")
        if drums_path and os.path.exists(drums_path):
            audio_clips.append(AudioFileClip(drums_path))
            print(f"Added drums audio: {drums_path}", flush=True)
        
        bass_path = state.get("bass_path")
        if bass_path and os.path.exists(bass_path):
            audio_clips.append(AudioFileClip(bass_path))
            print(f"Added bass audio: {bass_path}", flush=True)
        
        print(f"Total audio clips to combine: {len(audio_clips)}", flush=True)
        
        combined_audio = CompositeAudioClip(audio_clips)
        final_video = video.with_audio(combined_audio)
        dubbed_video_path = os.path.join(temp_folder, "dubbed_video.mp4")
        
        final_video.write_videofile(
            dubbed_video_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(temp_folder, "temp-audio.m4a"),
            remove_temp=True,
            logger="bar"
        )
        
        # Close all clips to release file handles
        final_video.close()
        video.close()
        for clip in audio_clips:
            clip.close()
        
        # # Cleanup temporary folders after video generation
        # folders_to_delete = ["cloned_audio", "cropped_dialogue", "demucs_output"]
        # for folder_name in folders_to_delete:
        #     folder_path = os.path.join(temp_folder, folder_name)
        #     if os.path.exists(folder_path):
        #         shutil.rmtree(folder_path)
        #         print(f"Deleted temporary folder: {folder_path}", flush=True)
        
        return {**state, "combined_audio_path": combined_cloned_audio_path, "dubbed_video_path": dubbed_video_path}
    except Exception as e:
        return {**state, "error": f"Error combining cloned audio: {e}"}


# Keep the original function for backward compatibility (deprecated)
def voice_cloning_node(state: AgentState) -> AgentState:
    """Legacy: Clones the voice using F5-TTS sequentially. Use parallel nodes instead."""
    audio_path = state.get("vocals_path")
    try:
        segments = state.get("transcription_segments", [])
        audio = AudioSegment.from_file(audio_path)
        temp_folder = state.get("temp_folder")
        cropped_audio_dir = os.path.join(temp_folder, "cropped_dialogue")
        crop_dialogue_segments(segments, audio, temp_folder)
        cloned_audio_dir = os.path.join(temp_folder, "cloned_audio")
        for segment in segments:
            ref_audio = os.path.join(
                cropped_audio_dir, f"cropped_{segment['id']}.wav")
            ref_text = segment["text"]
            gen_text = segment["translation"]
            if not ref_text or not gen_text:
                print(
                    f"Skipping segment {segment} due to empty ref_text or gen_text")
                continue
            run_f5_tts_infer("F5TTS_v1_Base", ref_audio, ref_text, gen_text,
                             output_dir=cloned_audio_dir, output_file=f"output_{segment['id']}.wav")
    except Exception as e:
        state["error"] = f"Error cloning voice: {e}"
    return state
