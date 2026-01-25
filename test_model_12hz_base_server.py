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
import binascii
import io
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from qwen_tts import Qwen3TTSModel

# Dynamically set CPU thread limits based on available cores
max_threads = max(1, (os.cpu_count() or 1) - 2)
os.environ["OMP_NUM_THREADS"] = str(max_threads)
os.environ["MKL_NUM_THREADS"] = str(max_threads)
torch.set_num_threads(max_threads)

@asynccontextmanager
async def lifespan(_: FastAPI):
    global tts
    preferred_device = DEVICE
    try:
        tts = load_tts_model(preferred_device)
    except Exception as exc:
        if preferred_device != "cpu":
            print(f"[startup] CUDA load failed, falling back to CPU: {exc}")
            tts = load_tts_model("cpu")
        else:
            raise
    try:
        yield
    finally:
        tts = None


app = FastAPI(title="Qwen3 TTS 12Hz Base ICL Server", lifespan=lifespan)

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COMMON_GEN_KWARGS = dict(
    max_new_tokens=2048,
    do_sample=True,
    top_k=50,
    top_p=1.0,
    temperature=0.9,
    repetition_penalty=1.05,
    subtalker_dosample=True,
    subtalker_top_k=50,
    subtalker_top_p=1.0,
    subtalker_temperature=0.9,
)


def load_tts_model(device: str) -> Qwen3TTSModel:
    return Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.float32,
        attn_implementation="eager",
    )


tts: Optional[Qwen3TTSModel] = None
VERBOSE = False


class CloneRequest(BaseModel):
    ref_audio_base64: str = Field(..., description="Base64-encoded WAV audio.")
    ref_text: str = Field(..., description="Reference transcript for the audio.")
    syn_text: str = Field(..., description="Text to synthesize.")
    syn_lang: str = Field("Auto", description="Synthesis language.")


class CloneResponse(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded WAV audio.")



@app.post("/clone", response_model=CloneResponse)
def clone_voice(request: CloneRequest, http_request: Request) -> CloneResponse:
    if tts is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    start_time = time.perf_counter()
    if VERBOSE:
        client_host = http_request.client.host if http_request.client else "unknown"
        print(
            f"[clone] client_ip={client_host} syn_text={request.syn_text!r} "
            f"syn_lang={request.syn_lang}"
        )

    try:
        audio_bytes = base64.b64decode(request.ref_audio_base64, validate=True)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 audio data.") from exc

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        wavs, sr = tts.generate_voice_clone(
            text=request.syn_text,
            language=request.syn_lang,
            ref_audio=temp_path,
            ref_text=request.ref_text,
            x_vector_only_mode=False,
            **COMMON_GEN_KWARGS,
        )
        if VERBOSE:
            elapsed = time.perf_counter() - start_time
            print(f"[clone] generation_time_sec={elapsed:.3f}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    output_buffer = io.BytesIO()
    sf.write(output_buffer, wavs[0], sr, format="WAV")
    audio_base64 = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
    return CloneResponse(audio_base64=audio_base64)


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Qwen3 TTS 12Hz base server.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose request logging.",
    )
    args = parser.parse_args()
    VERBOSE = args.verbose
    uvicorn.run(app, host="0.0.0.0", port=8000)
