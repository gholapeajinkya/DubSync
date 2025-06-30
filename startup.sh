#!/bin/bash

# Optional: move to your app folder
cd /home/site/wwwroot

# Run your Streamlit app
streamlit run app.py --server.port 8000 --server.enableCORS false
