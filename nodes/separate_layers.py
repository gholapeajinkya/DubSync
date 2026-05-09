from state import AgentState
import os
import subprocess
import torch
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"

def separate_audio_layers_node(state: AgentState) -> AgentState:
    """Separates the audio into different layers (e.g., dialogue, music, effects) and saves them to temporary locations."""
    try:
      output_dir = os.path.join(state["temp_folder"], "demucs_output")
      os.makedirs(output_dir, exist_ok=True)
      
      # Run demucs - outputs to output_dir/htdemucs/trackname/
      subprocess.run(["demucs", "-o", output_dir,
                      f"--device={device}", state["audio_path"]], capture_output=True, text=True)
      
      # Find the generated files in nested folder and move to output_dir
      audio_basename = os.path.splitext(os.path.basename(state["audio_path"]))[0]
      nested_dir = os.path.join(output_dir, "htdemucs", audio_basename)
      
      # Move all wav files from nested folder to output_dir
      if os.path.exists(nested_dir):
          for filename in os.listdir(nested_dir):
              if filename.endswith(".wav"):
                  src = os.path.join(nested_dir, filename)
                  dst = os.path.join(output_dir, filename)
                  shutil.move(src, dst)
          # Clean up nested folders
          shutil.rmtree(os.path.join(output_dir, "htdemucs"), ignore_errors=True)
      
      state["vocals_path"] = os.path.join(output_dir, "vocals.wav")
      state["drums_path"] = os.path.join(output_dir, "drums.wav")
      state["bass_path"] = os.path.join(output_dir, "bass.wav")
      state["effects_path"] = os.path.join(output_dir, "other.wav")
      state["error"] = None
    except Exception as e:
        state["error"] = f"Error separating audio layers: {e}"
    return state
