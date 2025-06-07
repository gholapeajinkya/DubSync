import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import os
import soundfile as sf
from pydub import AudioSegment
import requests
import time
import json

# Load environment variables from .env file
load_dotenv()
# Ensure the required environment variables are set
speech_key = os.getenv("AZURE_SPEECH_KEY")

service_region = os.getenv("AZURE_SERVICE_REGION")
# Path to the local audio file
# Must be WAV format, mono, 16-bit, 16kHz
audio_filename = "converted_vocals.wav"
audio_url = "https://ajinkyamlpoc4770520512.blob.core.windows.net/ag-storage-container/converted_vocals.wav?se=2025-06-07T07%3A40%3A10Z&sp=r&sv=2025-05-05&sr=b&sig=sr7XeVi3057BXlMf9VK0q2uNGgbd5bjiVMiG1L2ekSg%3D"


def verify_audio_properties(file_path):
    try:
        data, sample_rate = sf.read(file_path)
        info = sf.info(file_path)

        print("File Format:", info.format)               # Should be "WAV"
        print("Subtype:", info.subtype)                  # Should be "PCM_16"
        print("Sample Rate:", sample_rate)               # Should be 16000
        print("Channels:", info.channels)                # Should be 1 (mono)

        return (
            info.format == 'WAV' and
            info.subtype == 'PCM_16' and
            sample_rate == 16000 and
            info.channels == 1
        )
    except Exception as e:
        print("Error reading file:", e)
        return False


def convert_to_required_format(input_file, output_file="converted_vocals.wav"):
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1)         # mono
    audio = audio.set_frame_rate(16000)   # 16kHz
    audio = audio.set_sample_width(2)     # 16-bit = 2 bytes

    audio.export(output_file, format="wav")
    print("Converted and saved to:", output_file)


is_valid = verify_audio_properties(audio_filename)
if not is_valid:
    print("Audio file does not meet the required properties. Converting...")
    convert_to_required_format(audio_filename)
    audio_filename = "converted_vocals.wav"  # Update to the converted file

# Endpoint for transcription with diarization
endpoint = f"https://{service_region}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions"

# Step 1: Create transcription job
headers = {
    "Ocp-Apim-Subscription-Key": speech_key,
    "Content-Type": "application/json"
}

body = {
    "displayName": "DiarizationDemo",
    "description": "Speaker diarization demo(Japanese)",
    "locale": "ja-JP",
    "contentUrls": [audio_url],
    "properties": {
        "wordLevelTimestampsEnabled": True,
        "punctuationMode": "DictatedAndAutomatic",
        "profanityFilterMode": "Masked",
        "diarizationEnabled": True,
    },
    "model": None
}

try:
    response = requests.post(endpoint, headers=headers, json=body)
    response.raise_for_status()
except requests.exceptions.HTTPError as err:
    print("HTTP error:", err)
    print("Response content:", response.text)
    exit(1)

# Get transcription ID
transcription_location = response.headers["location"]
print("Transcription job created at:", transcription_location)

# Step 2: Poll for completion
while True:
    status_response = requests.get(transcription_location, headers=headers)
    status_response.raise_for_status()
    status = status_response.json()
    print("Status:", status["status"])

    if status["status"] in ["Succeeded", "Failed"]:
        break
    time.sleep(10)

# Step 3: Fetch diarized transcription results
# Step 3: Fetch diarized transcription results
if status["status"] == "Succeeded":
    print("Transcription succeeded!")
    with open("transcription_result_metadata.json", "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=4)

    files_url = status["links"]["files"]
    files_response = requests.get(files_url, headers=headers)
    files_response.raise_for_status()
    files_data = files_response.json()

    # Find the file with "transcription" kind
    transcription_file_url = None
    for file in files_data['values']:
        if file['kind'] == 'Transcription':
            transcription_file_url = file['links']['contentUrl']
            break

    if not transcription_file_url:
        print("No transcription file found in results.")
        exit(1)

    print("Downloading transcription results from:", transcription_file_url)
    result_response = requests.get(transcription_file_url)
    result_response.raise_for_status()
    transcription_result = result_response.json()

    # Print each utterance
    for utterance in transcription_result['recognizedPhrases']:
        speaker = utterance.get("speaker", "Unknown")
        print(f"[Speaker {speaker}] {utterance['nBest'][0]['display']}")
else:
    print("Transcription failed:", status)
