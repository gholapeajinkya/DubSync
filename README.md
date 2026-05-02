# DubSync

DubSync is a Streamlit-based application for automated video dubbing. It enables users to upload or link to a video, extract and transcribe its audio, translate the dialogue, and generate a dubbed version in a target language using advanced TTS and voice cloning.

![DubSync](assets/1753605713641.jpeg)

## 🚀 Live Demo

Try DubSync live on Hugging Face Spaces: [https://gholapeajinkya-dubsync.hf.space/](https://gholapeajinkya-dubsync.hf.space/)

## Features

- ✅ Upload or provide a YouTube video URL
- 🧠 Transcribe dialogue with OpenAI Whisper
- 🌐 Rewrite translations using Azure OpenAI (GPT-4)
- 🎤 Clone voices using F5-TTS with reference audio
- 🎼 Separate vocals & background using Demucs
- 🎚️ Sync re-synthesized voices with original timing
- 📝 Generate SRT subtitles for dubbed content
- 🎛️ Multi-threaded processing for speed
- 🧹 Cleanup temporary resources easily
- 🧪 Streamlit UI with real-time previews

## 📂 Folder Structure
```text
DubSync/
├── app.py                # Streamlit entry point
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
├── resources/            # Temp directory for processing
│   ├── cropped_audio/    # Cropped voice clips
│   ├── cloned_audio/     # F5-TTS output
│   └── demucs_output/    # Demucs separated layers
├── sample_outputs/       # sample results
```

## ⚙️ Tech Stack

| Task                     | Tool / Library         |
|--------------------------|------------------------|
| Transcription            | OpenAI Whisper         |
| AI Translation Rewrite   | Azure OpenAI (GPT-4)   |
| Voice Cloning            | F5-TTS CLI             |
| Audio Separation         | Demucs                 |
| Audio Processing         | pydub, moviepy         |
| Subtitle Generation      | SRT format             |
| UI                       | Streamlit              |

---

## 🚀 How It Works

1. **Upload or Input Video**
   - Either upload `.mp4` or provide a YouTube URL

2. **Audio Extraction & Separation**
   - Extracts audio and uses Demucs to split vocals/background

3. **Transcription**
   - Transcribes vocals using Whisper (selectable model)

4. **Translation & Script Rewriting**
   - GPT-4 rewrites translations with emotion, fillers, and timing awareness

5. **Voice Cloning**
   - F5-TTS clones voice using segment reference audio and generates matching speech

6. **Audio Assembly**
   - Audio segments are aligned and merged with original music

7. **Final Video Creation**
   - Reassembled into a final dubbed video with lips approximately in sync

---
## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) installed and available in PATH
- [Demucs](https://github.com/facebookresearch/demucs) for audio separation
- F5 TTS inference CLI for voice cloning
- Azure OpenAI and Google Translate API keys (set in `.env`)
- See [requirements.txt](requirements.txt) for all Python dependencies

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/gholapeajinkya/DubSync.git
    cd DubSync
    ```

2. Create and activate a virtual environment:
    ```sh
    python3.10 -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up your `.env` file with the following variables:
    ```
    AZURE_OPENAI_API_KEY=your_azure_openai_key
    AZURE_OPENAI_API_BASE=your_azure_openai_endpoint
    AZURE_OPENAI_API_VERSION=your_azure_aip_version
    AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name

    HF_TOKEN=your_hugging_face_token
    ```

5. Ensure `ffmpeg`, `demucs`, and `f5-tts_infer-cli` are installed and available in your PATH.

## Usage

Run the Streamlit app:

```sh
streamlit run app.py
```

## Whisper Models:

| Model      | Speed    | Accuracy   | Size      |
|------------|----------|------------|-----------|
| `tiny`     | Fastest  | Lowest     | ~39 MB    |
| `base`     | Fast     | Low-Medium | ~74 MB    |
| `small`    | Medium   | Medium     | ~244 MB   |
| `medium`   | Slower   | High       | ~769 MB   |
| `large`    | Slowest  | Highest    | ~1.55 GB  |
| `large-v2` | Slowest  | Highest    | ~1.55 GB  |
| `large-v3` | Slowest  | Highest    | ~1.55 GB  |
| `turbo`    | Fast     | High       | ~809 MB   |

