#!/usr/bin/env python3

"""
Transcribe a video file to plain text using a whisper.cpp server.

The script extracts the audio from the video, sends it to the server,
and writes the transcription to a .txt file with the same base name as the video.
"""

import os
import sys
import subprocess
import tempfile
import requests
import argparse

def transcribe_video_to_text(video_path: str, server_url: str = "http://127.0.0.1:9191/inference"):
    """
    Transcribe the given video file to a .txt file.

    Args:
        video_path (str): Path to the video file.
        server_url (str): Whisper.cpp server inference endpoint.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    print(f"Processing video: {video_path}")

    # 1. Extract audio to a temporary WAV file (16kHz mono)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav_file:
        tmp_wav_path = tmp_wav_file.name

    try:
        print("Extracting audio with ffmpeg...")
        command = [
            "ffmpeg",
            "-i", video_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-y",
            "-loglevel", "error",
            tmp_wav_path,
        ]
        subprocess.run(command, check=True, capture_output=True, text=True, errors="ignore")
    except FileNotFoundError:
        print("Error: 'ffmpeg' not found. Please install it and ensure it is in your PATH.")
        os.remove(tmp_wav_path)
        return
    except subprocess.CalledProcessError as e:
        print("Error during ffmpeg audio extraction:")
        print(e.stderr)
        os.remove(tmp_wav_path)
        return

    # 2. Send audio to whisper server, request plain text output
    try:
        print(f"Sending audio to whisper server at {server_url}...")
        with open(tmp_wav_path, "rb") as audio_file:
            files = {"file": (os.path.basename(tmp_wav_path), audio_file, "audio/wav")}
            data = {"response_format": "text"}  # request plain text
            response = requests.post(server_url, files=files, data=data, timeout=3600)
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error sending request to whisper server: {e}")
        return
    finally:
        # Clean up temporary WAV file
        os.remove(tmp_wav_path)

    # 3. Write the transcription to a .txt file
    txt_path = os.path.splitext(video_path)[0] + ".txt"
    try:
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            # Write transcription as a single line (replace newlines with spaces)
            single_line = " ".join(response.text.splitlines())
            txt_file.write(single_line)
        print(f"Transcription saved to: {txt_path}")
    except IOError as e:
        print(f"Error writing transcription file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe a video file to plain text using a whisper.cpp server.")
    parser.add_argument("video_path", help="Path to the video file to transcribe.")
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:9191/inference",
        help="Whisper.cpp server inference endpoint (default: http://127.0.0.1:9191/inference).",
    )
    args = parser.parse_args()
    transcribe_video_to_text(args.video_path, args.server_url)

if __name__ == "__main__":
    main()
