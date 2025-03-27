#!/bin/bash
# Set executable permissions
chmod +x ./bin/*

# Verify FFmpeg
./bin/ffmpeg -version || exit 1
pip install -r requirements.txt
