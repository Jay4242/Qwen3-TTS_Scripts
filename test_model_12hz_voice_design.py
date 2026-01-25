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
import time
import torch
import soundfile as sf
import argparse

# Dynamically set CPU thread limits based on available cores
import os
max_threads = max(1, (os.cpu_count() or 1) - 2)
os.environ["OMP_NUM_THREADS"] = str(max_threads)
os.environ["MKL_NUM_THREADS"] = str(max_threads)
torch.set_num_threads(max_threads)

from qwen_tts import Qwen3TTSModel


def main(text: str, instruct: str):
    device = "cpu"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.float32,
        attn_implementation="eager",
    )

    t0 = time.time()

    wavs, sr = tts.generate_voice_design(
        text=text,
        language="English",
        instruct=instruct,
    )

    t1 = time.time()
    print(f"[VoiceDesign English] time: {t1 - t0:.3f}s")

    sf.write("qwen3_tts_test_voice_design_english.wav", wavs[0], sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Spoken text to synthesize.",
    )
    parser.add_argument(
        "--instruct",
        type=str,
        required=True,
        help="Instruction for the voice design.",
    )
    args = parser.parse_args()
    # If the provided text argument is a path to a file, read its contents.
    if os.path.isfile(args.text):
        with open(args.text, "r", encoding="utf-8") as f:
            args.text = f.read().strip()
    try:
        main(args.text, args.instruct)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
