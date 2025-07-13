#!/bin/bash

echo "Files in /app/demos:"
ls -la ./demos

JSON_FILE="./demos/demo.json"  # Change this path if needed

# Helper function to download with gdown or curl
download_file() {
  URL="$1"
  OUTPUT="$2"

  if [[ "$URL" == *"drive.google.com"* ]]; then
    echo "Detected Google Drive URL. Using gdown..."
    FILE_ID=$(echo "$URL" | sed -n 's|.*\/d\/\([^\/]*\).*|\1|p')
    if [ -z "$FILE_ID" ]; then
      FILE_ID=$(echo "$URL" | sed -n 's|.*id=\([^&]*\).*|\1|p')
    fi
    if [ -n "$FILE_ID" ]; then
      gdown --id "$FILE_ID" -O "$OUTPUT"
    else
      echo "⚠️ Could not extract Google Drive file ID from: $URL"
    fi
  else
    curl -L "$URL" -o "$OUTPUT"
  fi
}

# Iterate over all demo objects
jq -c '.[]' "$JSON_FILE" | while read -r demo_entry; do
  # Parse demo-level ID
  DEMO_ID=$(echo "$demo_entry" | jq -r '.id')
  DEMO_TITLE=$(echo "$demo_entry" | jq -r '.title')
  DEMO_DIR="./demos/$DEMO_ID/videos"
  mkdir -p "$DEMO_DIR"
  
  echo "Processing demo: $DEMO_TITLE (ID: $DEMO_ID)"

  # Iterate over videos within this demo
  echo "$demo_entry" | jq -c '.videos[]' | while read -r video_entry; do
    VIDEO_ID=$(echo "$video_entry" | jq -r '.id')
    VIDEO_DIR="$DEMO_DIR/$VIDEO_ID"
    mkdir -p "$VIDEO_DIR"

    # Download video
    VIDEO_URL=$(echo "$video_entry" | jq -r '.video_url')
    if [[ "$VIDEO_URL" != "null" ]]; then
      echo "Downloading video for Demo $DEMO_ID, Video ID $VIDEO_ID"
      download_file "$VIDEO_URL" "$VIDEO_DIR/video.mp4"
    fi

    # Download subtitle
    SUBTITLE_URL=$(echo "$video_entry" | jq -r '.subtitle_url')
    if [[ "$SUBTITLE_URL" != "null" ]]; then
      echo "Downloading subtitle for Demo $DEMO_ID, Video ID $VIDEO_ID"
      download_file "$SUBTITLE_URL" "$VIDEO_DIR/subtitle.srt"
    fi

    # Download transcription (if exists)
    TRANSCRIPTION_URL=$(echo "$video_entry" | jq -r '.transcription_url // empty')
    if [[ -n "$TRANSCRIPTION_URL" ]]; then
      echo "Downloading transcription for Demo $DEMO_ID, Video ID $VIDEO_ID"
      download_file "$TRANSCRIPTION_URL" "$VIDEO_DIR/transcription.csv"
    fi
  done
done

echo "Files in /app/demos:"
ls -la ./demos