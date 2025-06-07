https://github.com/suno-ai/bark?tab=readme-ov-file

https://huggingface.co/spaces/suno/bark


1.30 => 40

source venv/bin/activate
deactivate



https://docs.streamlit.io/develop/api-reference/layout


streamlit run main_ui.py

✅ Best Practice Pipeline
🔊 Split original audio into timestamped segments (e.g., Whisper)

🌐 Translate each segment

✍️ Optionally rephrase to fit time better

🗣️ Use high-quality TTS (SSML or segment-based)

🔇 Add silence to fill timing gaps

🔁 Merge all into final track aligned with original video


Diarization, in the context of speech recognition and audio processing, refers to the process of identifying and separating different speakers in an audio recording. It's essentially about determining "who said what" within an audio file, typically without knowing the speakers' identities in advance. This process segments the audio into distinct sections, each attributed to a different speaker. 