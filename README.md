# DubSync

DubSync is a Streamlit-based application for automated video dubbing. It enables users to upload or link to a video, extract and transcribe its audio, translate the dialogue, and generate a dubbed version in a target language using advanced TTS and voice cloning.

## Features

- Upload or download MP4 videos (including YouTube support)
- Extracts and separates audio layers (vocals, background, drums, bass) using Demucs
- Transcribes speech using OpenAI Whisper (multiple models supported)
- Translates dialogue using Google Translate or Azure OpenAI GPT-4
- Optionally rewrites translations for natural, expressive dubbing
- Clones voices for each segment using F5 TTS
- Reconstructs and merges dubbed audio with original music and video
- Outputs a fully dubbed video in the selected language

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
    git clone https://github.com/yourusername/DubSync.git
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
    OPENAI_API_KEY=your_azure_openai_key
    OPENAI_API_BASE=your_azure_openai_endpoint
    ```

5. Ensure `ffmpeg`, `demucs`, and `f5-tts_infer-cli` are installed and available in your PATH.

## Usage

Run the Streamlit app:

```sh
streamlit run [app.py](http://_vscodecontentref_/2)
```