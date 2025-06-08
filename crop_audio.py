from pydub import AudioSegment

# Load audio file
audio = AudioSegment.from_file("converted_vocals.wav")

# Define start and end time in milliseconds
start_time = 0  # 10 seconds
end_time = 8 * 1000    # 30 seconds

# Crop (slice) the audio
cropped_audio = audio[start_time:end_time]

# Save cropped audio
cropped_audio.export("cropped_audio.wav", format="wav")
