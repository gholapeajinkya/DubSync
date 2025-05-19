import os
import json
import numpy as np

# Patch for librosa compatibility with numpy>=1.24
np.complex = complex

import librosa
from speechbrain.pretrained import SpeakerRecognition

# Load the speaker recognition model
recognizer = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_spkrec"
)

def analyze_voice(audio_path):
    result = {
        "file": os.path.basename(audio_path)
    }

    # Step 1: Voice Embedding (Unique Voice ID)
    try:
        embedding = recognizer.encode_batch(audio_path).squeeze().cpu().numpy()
        result["embedding"] = embedding.tolist()  # can be hashed or compared later
    except Exception as e:
        result["embedding"] = None
        result["error_embedding"] = str(e)

    # Step 2: Gender Detection using pitch
    try:
        y, sr = librosa.load(audio_path, sr=None)
        f0, voiced_flag, _ = librosa.pyin(y, fmin=75, fmax=300)

        if f0 is not None and np.any(voiced_flag):
            mean_pitch = np.nanmean(f0[voiced_flag])
            result["mean_pitch"] = round(mean_pitch, 2)
            result["gender"] = "Male" if mean_pitch < 165 else "Female"
        else:
            result["mean_pitch"] = None
            result["gender"] = "Unknown"
    except Exception as e:
        result["mean_pitch"] = None
        result["gender"] = "Error"
        result["error_pitch"] = str(e)

    return result

# === 🔄 Analyze a folder of audio files ===
def batch_analyze(folder_path, out_json="voice_results.json"):
    all_results = []

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".wav"):
            fpath = os.path.join(folder_path, fname)
            print(f"🔍 Analyzing: {fname}")
            result = analyze_voice(fpath)
            all_results.append(result)

    # Save results
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Saved results to {out_json}")

# 🔧 Change this to your folder of .wav files
# audio_folder = "voice_samples"
# batch_analyze(audio_folder)
