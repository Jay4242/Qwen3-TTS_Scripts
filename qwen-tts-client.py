# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import base64
import io
import json
import os
from typing import Any, Dict

import requests
import sounddevice as sd
import soundfile as sf


def load_text(text_arg: str) -> str:
    if os.path.isfile(text_arg):
        with open(text_arg, "r", encoding="utf-8") as f:
            return f.read().strip()
    return text_arg


def encode_audio_base64(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode("utf-8")


def post_clone_request(
    server_url: str,
    ref_audio_base64: str,
    ref_text: str,
    syn_text: str,
    syn_lang: str,
    timeout: float,
) -> Dict[str, Any]:
    payload = {
        "ref_audio_base64": ref_audio_base64,
        "ref_text": ref_text,
        "syn_text": syn_text,
        "syn_lang": syn_lang,
    }
    response = requests.post(
        f"{server_url.rstrip('/')}/clone",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def play_audio_base64(audio_base64: str) -> None:
    audio_bytes = base64.b64decode(audio_base64)
    audio_buffer = io.BytesIO(audio_bytes)
    audio_data, sample_rate = sf.read(audio_buffer, dtype="float32")
    sd.play(audio_data, sample_rate)
    sd.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3 TTS base server client.")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL for the TTS server.",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        required=True,
        help="Path to reference WAV audio file.",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        required=True,
        help="Reference text or path to a text file.",
    )
    parser.add_argument(
        "--syn-text",
        type=str,
        required=True,
        help="Synthesis text or path to a text file.",
    )
    parser.add_argument(
        "--syn-lang",
        type=str,
        default="Auto",
        help="Synthesis language.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds.",
    )
    args = parser.parse_args()

    ref_text = load_text(args.ref_text)
    syn_text = load_text(args.syn_text)
    ref_audio_base64 = encode_audio_base64(args.ref_audio)

    response_json = post_clone_request(
        server_url=args.server_url,
        ref_audio_base64=ref_audio_base64,
        ref_text=ref_text,
        syn_text=syn_text,
        syn_lang=args.syn_lang,
        timeout=args.timeout,
    )

    audio_base64 = response_json.get("audio_base64")
    if not audio_base64:
        raise RuntimeError("Server response missing 'audio_base64'.")

    play_audio_base64(audio_base64)


if __name__ == "__main__":
    main()
