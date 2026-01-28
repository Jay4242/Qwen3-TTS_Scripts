#!/usr/bin/env python3
import argparse
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from qwen_tts import Qwen3TTSModel

# Dynamically set CPU thread limits based on available cores
max_threads = max(1, (os.cpu_count() or 1) - 2)
os.environ["OMP_NUM_THREADS"] = str(max_threads)
os.environ["MKL_NUM_THREADS"] = str(max_threads)
torch.set_num_threads(max_threads)

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

BASE_DIR = Path(__file__).resolve().parent.parent


def resolve_reference_paths(model_name: str) -> tuple[Path, Path]:
    safe_name = Path(model_name).name
    audio_path = BASE_DIR / f"{safe_name}.wav"
    text_path = BASE_DIR / f"{safe_name}.txt"
    if not audio_path.is_file() or not text_path.is_file():
        raise FileNotFoundError(
            f"Reference files not found for model '{model_name}': "
            f"{audio_path.name}, {text_path.name}"
        )
    return audio_path, text_path

tts: Optional[Qwen3TTSModel] = None
VERBOSE = False


def load_tts_model(device: str) -> Qwen3TTSModel:
    return Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.float32,
        attn_implementation="eager",
    )


def load_ref_text(text_path: Path) -> str:
    try:
        return text_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Reference text not found: {text_path}") from exc


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


app = FastAPI(title="Qwen3 TTS 12Hz Base OpenAI-Compatible Server", lifespan=lifespan)


class OpenAITtsRequest(BaseModel):
    model: str = Field(..., description="OpenAI model name (ignored).")
    input: str = Field(..., description="Text to synthesize.")
    voice: str = Field(..., description="Voice name (used to select reference files).")
    instructions: Optional[str] = Field(None, description="Instructions (ignored).")


def resolve_output_format(request: Request) -> str:
    accept = request.headers.get("accept", "audio/mpeg").lower()
    if "audio/wav" in accept or "audio/wave" in accept:
        return "WAV"
    return "MP3"


@app.post("/v1/audio/speech")
def create_speech(payload: OpenAITtsRequest, http_request: Request) -> Response:
    if tts is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    if not payload.input.strip():
        raise HTTPException(status_code=400, detail="Input text must not be empty.")

    start_time = time.perf_counter()
    if VERBOSE:
        client_host = http_request.client.host if http_request.client else "unknown"
        print(
            f"[speech] client_ip={client_host} model={payload.model!r} "
            f"voice={payload.voice!r} instructions={payload.instructions!r} "
            f"input_len={len(payload.input)}"
        )

    try:
        ref_audio_path, ref_text_path = resolve_reference_paths(payload.voice)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ref_text = load_ref_text(ref_text_path)

    try:
        wavs, sr = tts.generate_voice_clone(
            text=payload.input,
            language="Auto",
            ref_audio=str(ref_audio_path),
            ref_text=ref_text,
            x_vector_only_mode=False,
            **COMMON_GEN_KWARGS,
        )
        if VERBOSE:
            elapsed = time.perf_counter() - start_time
            print(f"[speech] generation_time_sec={elapsed:.3f}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    output_buffer = io.BytesIO()
    output_format = resolve_output_format(http_request)
    sf.write(output_buffer, wavs[0], sr, format=output_format)
    media_type = "audio/mpeg" if output_format == "MP3" else "audio/wav"
    return Response(content=output_buffer.getvalue(), media_type=media_type)


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(
        description="Qwen3 TTS 12Hz base OpenAI-compatible server."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose request logging.",
    )
    args = parser.parse_args()
    VERBOSE = args.verbose
    uvicorn.run(app, host="0.0.0.0", port=8000)
