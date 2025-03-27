#!/bin/bash
# Make FFmpeg binaries executable
chmod +x ./bin/ffmpeg
chmod +x ./bin/ffplay
chmod +x ./bin/ffprobe

# Add to PATH
export PATH=$PATH:$(pwd)/bin
