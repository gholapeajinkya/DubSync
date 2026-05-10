from typing import TypedDict, Optional, Annotated, Required


def merge_cloned_segments(existing: list, new: list) -> list:
    """Reducer to merge cloned segment results from parallel workers."""
    if existing is None:
        existing = []
    if new is None:
        return existing
    return existing + new


class AgentState(TypedDict, total=False):
    # Required params (must be provided)
    input_video_path: Required[str]  # Path to the input video file
    temp_folder: Required[str]  # Path to temporary folder for intermediate files
    # Optional params (populated during processing)
    audio_path: Optional[str]  # Extracted audio file from input video
    vocals_path: Optional[str]  # Path to isolated vocals audio
    drums_path: Optional[str]  # Path to isolated drums audio
    bass_path: Optional[str]  # Path to isolated bass audio
    effects_path: Optional[str]  # Path to isolated effects audio
    cleaned_audio_path: Optional[str]  # Path to cleaned dialogue audio for transcription
    transcription_segments: Optional[list]  # Segments of transcribed dialogue
    translations: Optional[list]
    # Parallel voice cloning support
    cloned_segments: Annotated[list, merge_cloned_segments]  # Collected results from parallel workers
    combined_audio_path: Optional[str]  # Final combined audio path
    dubbed_video_path: Optional[str]  # Final dubbed video path
    error: Optional[str]


class VoiceCloningWorkerState(TypedDict):
    """State for individual voice cloning worker."""
    segment: dict  # Single segment to process
    dialogue_path: str
    temp_folder: str  # Path to temporary folder
    cloned_segments: Annotated[list, merge_cloned_segments]
    error: Optional[str]