## Activate and deactivate python env
source venv/bin/activate
deactivate


## How to Run
`
streamlit run app.py
`

(Streamlit)[https://docs.streamlit.io/develop/api-reference/layout]

## The Plan

✅ Best Practice Pipeline
🔊 Split original audio into timestamped segments (e.g., Whisper)

🌐 Translate each segment

✍️ Optionally rephrase to fit time better

🗣️ Use high-quality TTS (SSML or segment-based)

🔇 Add silence to fill timing gaps

🔁 Merge all into final track aligned with original video


Diarization, in the context of speech recognition and audio processing, refers to the process of identifying and separating different speakers in an audio recording. It's essentially about determining "who said what" within an audio file, typically without knowing the speakers' identities in advance. This process segments the audio into distinct sections, each attributed to a different speaker. 

## New virtual env: 

How to create - `python3.10 -m venv venv310`
Activate - `source venv310/bin/activate`
Freeze requiremets - `pip freeze > requirements310.txt`


## Voice cloning
F5-TTS(https://SWivid.github.io/F5-TTS)

'''
f5-tts_infer-cli \
--model "E2-TTS" \
--ref_audio "cropped_audio.wav" \
--ref_text "僭越ながら今回のレイドでリーダーを務めさせていただきますソン・チオルですよろしくお願いします" \
--gen_text "I am proud to be the leader of this raid. It's Song Chi-Ol thank you"
'''


## Misc
https://github.com/suno-ai/bark?tab=readme-ov-file

https://huggingface.co/spaces/suno/bark



Previously used to take around 10-12 mins for 2 min video

- With 4 threads for voice cloning = 5 mins
- With 4 threads for segment audio cropping = 4.32 mins