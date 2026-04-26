from state import AgentState
from moviepy import VideoFileClip
import os

temp_folder = "resources"
os.makedirs(temp_folder, exist_ok=True)


def extract_audio_node(state: AgentState) -> AgentState:
    """Extracts the audio from the provided video file and saves it to a temporary location."""
    try:
        video = VideoFileClip(state["video_path"])
        audio_path = os.path.join(temp_folder, "extracted_audio.wav")
        video.audio.write_audiofile(audio_path)
        state["audio_path"] = audio_path
        state["error"] = None
    except Exception as e:
        state["error"] = f"Error extracting audio: {e}"
    return state
