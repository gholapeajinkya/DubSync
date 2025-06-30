FROM python:3.10-slim

WORKDIR /app

ENV XDG_CONFIG_HOME=/tmp/.config
ENV XDG_CACHE_HOME=/tmp/.cache

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
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/

RUN pip3 install -r requirements.txt


EXPOSE 8501


ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]