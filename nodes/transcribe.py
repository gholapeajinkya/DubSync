import whisper
import torch
from state import AgentState

device = "cuda" if torch.cuda.is_available() else "cpu"
fp16 = device == "cuda"


def transcribe_audio_node(state: AgentState) -> AgentState:
    """Transcribes the cleaned dialogue audio using OpenAI's Whisper model and saves the transcription to the state for further processing.
    """
    audio_path = state.get("cleaned_audio_path")
    if not audio_path:
        state["error"] = "No cleaned audio path found for transcription."
        return state
    try:
        model = whisper.load_model("large", device=device)
        result = model.transcribe(
            audio_path,
            language="ja",  # Assuming Japanese audio, adjust as needed
            word_timestamps=False,
            fp16=fp16,
            condition_on_previous_text=True,
            verbose=True,
            task="transcribe"
        )
        print(f"Transcription segments: {result['segments']}")
        state["transcription_segments"] = result["segments"]
    except Exception as e:
        state["error"] = f"Error during transcription: {e}"
        return state
    return state