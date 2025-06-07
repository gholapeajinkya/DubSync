import torch
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from TTS.utils.synthesizer import Synthesizer

# Load your sample voice file (wav)
voice_sample_path = "speaker_sample.wav"
text_to_speak = "Hello, this is a test of voice cloning."

# Step 1: Extract speaker embedding using resemblyzer
encoder = VoiceEncoder()

wav = preprocess_wav(voice_sample_path)
embedding = encoder.embed_utterance(wav)

# Step 2: Load YourTTS synthesizer
synthesizer = Synthesizer(
    tts_checkpoint="yourtts.pt",
    tts_config_path="config.json",    # Use YourTTS config from repo
    speaker_embeddings=True,
)

# Step 3: Generate speech with cloned voice
wav_gen = synthesizer.tts(text_to_speak, speaker_embedding=embedding)

# Step 4: Save output
sf.write("cloned_voice_output.wav", wav_gen, samplerate=22050)

print("Generated speech saved as cloned_voice_output.wav")
