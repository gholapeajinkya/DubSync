from pyannote.audio import Pipeline, Audio
from pyannote.core import Segment
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# --- Setup
AUDIO_PATH = "converted_vocals.wav"  # must be 16kHz mono WAV
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Load diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=HUGGINGFACE_TOKEN
)

# Run diarization to get speaker segments
diarization = pipeline(AUDIO_PATH)

# Load audio handler
audio_loader = Audio(sample_rate=16000, mono=True)

# Load embedding model
embedding_model = pipeline._embedding

# Collect embeddings
embeddings = []
timestamps = []

for turn, track, speaker in diarization.itertracks(yield_label=True):
    # try:
    segment = Segment(turn.start, turn.end)

    # ✅ Proper usage of Audio.crop: pass path and Segment
    waveform, sample_rate = audio_loader.crop(AUDIO_PATH, segment)

    # Skip short segments
    if waveform.shape[1] < sample_rate * 0.5:
        print(f"Skipping short segment {turn.start:.2f}-{turn.end:.2f}")
        continue

    # Compute embedding
    waveform = waveform.unsqueeze(0)
    embedding = embedding_model(waveform)
    embeddings.append(embedding.squeeze())
    timestamps.append((turn.start, turn.end))

    # except Exception as e:
    #     print(f"Skipping segment {turn.start:.2f}-{turn.end:.2f}: {e}")

# Cluster if we got valid embeddings
if len(embeddings) == 0:
    print("❌ No valid segments found for embedding.")
else:
    X = np.vstack(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0.8, linkage="average")
    labels = clustering.fit_predict(X)

    # Output speaker segments
    for idx, (start, end) in enumerate(timestamps):
        print(
            f"🗣️ Speaker {labels[idx]} | Start: {start:.2f}s | End: {end:.2f}s")
