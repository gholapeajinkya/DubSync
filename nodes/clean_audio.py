from state import AgentState
import numpy as np
import librosa
import noisereduce as nr
from pydub import AudioSegment
from scipy.signal import butter, filtfilt


def clean_audio_node(state: AgentState) -> AgentState:
    """
      Cleans the separated audio layers (e.g., noise reduction) and saves them to temporary locations for better Whisper transcription quality.
    """
    audio_path = state.get(
        "vocals_path")  # Clean dialogue layer for better transcription
    if not audio_path:
        state["error"] = "No dialogue audio path found for cleaning."
        return state
    try:
        # Define output path
        output_path = audio_path.replace('.wav', '_cleaned.wav')
        
        # Load audio with librosa for better preprocessing
        audio_data, sample_rate = librosa.load(
            audio_path, sr=16000)  # Whisper prefers 16kHz

        # 1. Noise reduction using spectral gating
        # First, estimate noise from the first 1 second (assuming it contains background noise)
        noise_sample_duration = min(
            1.0, len(audio_data) / sample_rate * 0.1)  # 10% or 1 sec max
        noise_sample_size = int(noise_sample_duration * sample_rate)

        if len(audio_data) > noise_sample_size:
            # Use noisereduce if available, otherwise use simple high-pass filter
            try:
                audio_data = nr.reduce_noise(
                    y=audio_data,
                    sr=sample_rate,
                    stationary=False,  # Non-stationary noise reduction
                    prop_decrease=0.8  # Use configurable strength
                )
            except ImportError:
                # Fallback: Simple high-pass filter to remove low-frequency noise
                nyquist = sample_rate * 0.5
                low_cutoff = 80  # Remove frequencies below 80Hz
                low_cutoff_normalized = low_cutoff / nyquist
                b, a = butter(5, low_cutoff_normalized, btype='high')
                audio_data = filtfilt(b, a, audio_data)
                print("noisereduce not available, using high-pass filter instead")

        # 2. Normalize audio levels
        # RMS normalization to -20dB to prevent clipping while maintaining dynamics
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0:
            target_rms = 10**(-20/20)  # -20dB
            audio_data = audio_data * (target_rms / rms)

        # 3. Remove silence and very quiet parts (optional, be careful not to remove speech pauses)
        # Use librosa's voice activity detection
        intervals = librosa.effects.split(
            audio_data,
            top_db=30,  # Threshold for silence detection
            frame_length=2048,
            hop_length=512
        )

        # 4. Apply gentle compression to even out volume levels
        # Simple soft limiting
        threshold = 0.95
        audio_data = np.where(
            np.abs(audio_data) > threshold,
            np.sign(audio_data) * (threshold +
                                   (np.abs(audio_data) - threshold) * 0.1),
            audio_data
        )

        # 5. Final normalization to prevent clipping
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude > 0.95:
            audio_data = audio_data * (0.95 / max_amplitude)

        # Convert back to pydub AudioSegment for consistency with your existing code
        audio_int16 = (audio_data * 32767).astype(np.int16)
        cleaned_audio = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=1
        )

        cleaned_audio.export(output_path, format="wav")
        print(f"Audio cleaned and saved to: {output_path}")
        state["cleaned_audio_path"] = output_path

    except Exception as e:
        state["error"] = f"Error cleaning audio: {e}"
    return state
