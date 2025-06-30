#!/bin/bash
export XDG_CACHE_HOME="/tmp/.cache"
export XDG_CONFIG_HOME="/tmp/.config"

pip install -r requirements.txt

# If your main file is app.py at the root
streamlit run app.py --server.port=8000 --server.enableCORS=false