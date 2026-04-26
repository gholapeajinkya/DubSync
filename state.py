from typing import TypedDict, Optional

class AgentState(TypedDict):
  video_path: str
  audio_path: str
  temp_folder: str = "resources"
  dialogue_path: Optional[str]
  music_path: Optional[str]
  effects_path: Optional[str]
  cleaned_audio_path: Optional[str]
  transcription_segments: Optional[list]
  translations: Optional[list]
  error: Optional[str]