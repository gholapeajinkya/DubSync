import os
import subprocess
from state import AgentState
from pydub import AudioSegment

cropped_audio_dir = os.path.join("resources", "cropped_dialogue")
os.makedirs(cropped_audio_dir, exist_ok=True)
cloned_audio_dir = os.path.join("resources", "cloned_audio")

os.makedirs(cloned_audio_dir, exist_ok=True)
def crop_dialogue_segments(segments, audio):
    for segment in segments:
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            cropped = audio[start_ms:end_ms]
            cropped.export(os.path.join(cropped_audio_dir,
                           f"cropped_{segment['id']}.wav"), format="wav")
            print(f"cropped_{segment['id']}.wav", flush=True)

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

def voice_cloning_node (state: AgentState) -> AgentState:
    """Clones the voice using F5-TTS from the dialogue layer and saves it to a temporary location."""
    audio_path = state.get("dialogue_path")
    try:
        segments = state.get("transcription_segments", [])
        audio = AudioSegment.from_file(audio_path)
        crop_dialogue_segments(segments, audio)
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
        state["error"] = f"Error cloning voice: {e}"
    return state