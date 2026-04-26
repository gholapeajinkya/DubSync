from state import AgentState
import os
import subprocess
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def separate_audio_layers_node(state: AgentState) -> AgentState:
    """Separates the audio into different layers (e.g., dialogue, music, effects) and saves them to temporary locations."""
    try:
      output_dir = os.path.join(state["temp_folder"], "demucs_output")
      subprocess.run(["demucs", "-o", output_dir,
                      f"--device={device}", state["audio_path"]], capture_output=True, text=True)
      state["dialogue_path"] = os.path.join(output_dir, "separated_dialogue.wav")
      state["music_path"] = os.path.join(output_dir, "separated_music.wav")
      state["effects_path"] = os.path.join(output_dir, "separated_effects.wav")
      state["error"] = None
    except Exception as e:
        state["error"] = f"Error separating audio layers: {e}"
    return state
