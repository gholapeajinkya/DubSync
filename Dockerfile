FROM python:3.10

# Set working directory
WORKDIR /app

# # === ENV VARS ===
# ENV XDG_CONFIG_HOME=/tmp/.config \
#     XDG_CACHE_HOME=/tmp/.cache \
#     HF_HOME=/tmp/huggingface \
#     HF_HUB_CACHE=/tmp/huggingface/hub \
#     STREAMLIT_HOME=/tmp/.streamlit \
#     MPLCONFIGDIR=/tmp/matplotlib \
#     PYTHONUNBUFFERED=1

# === Install system packages ===
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    libprotobuf-dev \
    protobuf-compiler \
    libsentencepiece-dev \
    ffmpeg \
    software-properties-common \
    git \
    jq \
    && rm -rf /var/lib/apt/lists/*

# === Install Python CLI packages ===
RUN pip install gdown

# === Create writable directories ===
RUN mkdir -p $XDG_CONFIG_HOME/streamlit $XDG_CACHE_HOME $HF_HUB_CACHE $STREAMLIT_HOME

# === Add basic Streamlit config ===
RUN echo "[server]\nheadless = true\nenableCORS = false\ndefaultPort = 8501" > $XDG_CONFIG_HOME/streamlit/config.toml

# === Copy app code ===
COPY --chown=user requirements.txt ./
COPY --chown=user app.py ./
COPY demos ./demos
COPY download_assets.sh ./


# === Install Python dependencies ===
RUN pip3 install --no-cache-dir -r requirements.txt

# Run the shell script to process JSON
RUN chmod +x download_assets.sh && ./download_assets.sh

# === Expose port & healthcheck ===
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# === Run Streamlit app ===
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
