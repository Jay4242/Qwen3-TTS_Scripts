#!/usr/bin/env python3
import argparse
import os
import time

import soundfile as sf
import torch

# Dynamically set CPU thread limits based on available cores
max_threads = max(1, (os.cpu_count() or 1) - 2)
os.environ["OMP_NUM_THREADS"] = str(max_threads)
os.environ["MKL_NUM_THREADS"] = str(max_threads)
torch.set_num_threads(max_threads)

from qwen_tts import Qwen3TTSModel


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def next_available_take_path(out_dir: str, case_name: str, take_index: int) -> str:
    while True:
        filename = f"{case_name}_{take_index:03d}.wav"
        path = os.path.join(out_dir, filename)
        if not os.path.exists(path):
            return path
        take_index += 1


def run_case(tts: Qwen3TTSModel, out_dir: str, case_name: str, call_fn):
    # Synchronize if CUDA is available; otherwise no-op.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = call_fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    print(f"[{case_name}] time: {t1 - t0:.3f}s, n_wavs={len(wavs)}, sr={sr}")

    for i, w in enumerate(wavs):
        out_path = next_available_take_path(out_dir, case_name, i)
        sf.write(out_path, w, sr)


def load_syn_texts(syn_text: str) -> list[str]:
    if os.path.isfile(syn_text):
        with open(syn_text, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        return [line for line in lines if line]
    return [syn_text]


def main(
    ref_audio: str,
    ref_text: str,
    syn_text: str,
    syn_lang: str = "Auto",
    line: int | None = None,
):
    device = "cpu"
    model_path = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    out_dir = "qwen3_tts_lines"
    ensure_dir(out_dir)

    tts = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.float32,
        attn_implementation="eager",
    )

    ref_audio_single = ref_audio
    ref_text_single = ref_text
    syn_lang_single = syn_lang

    syn_text_lines = load_syn_texts(syn_text)
    if not syn_text_lines:
        raise ValueError("No synthesis text lines found.")
    if line is not None:
        if line < 1 or line > len(syn_text_lines):
            raise ValueError(
                f"Requested line {line} is out of range. Valid range: 1-{len(syn_text_lines)}."
            )
        syn_text_lines = [syn_text_lines[line - 1]]

    common_gen_kwargs = dict(
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

    line_start = line if line is not None else 1
    for line_index, syn_text_single in enumerate(syn_text_lines, start=line_start):
        case_name = f"case1_promptSingle_synSingle_direct_icl_line{line_index:04d}"
        run_case(
            tts,
            out_dir,
            case_name,
            lambda text=syn_text_single: tts.generate_voice_clone(
                text=text,
                language=syn_lang_single,
                ref_audio=ref_audio_single,
                ref_text=ref_text_single,
                x_vector_only_mode=False,
                **common_gen_kwargs,
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref-audio",
        type=str,
        required=True,
        help="Reference audio URL or file path.",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        required=True,
        help="Reference text for the audio. If this is a path to a file, the file will be read and its contents used as the transcript.",
    )
    parser.add_argument(
        "--syn-text",
        type=str,
        required=True,
        help="Synthesis target text. If this is a path to a file, each line is treated as a generation.",
    )
    parser.add_argument(
        "--syn-lang",
        type=str,
        default="Auto",
        help="Synthesis language.",
    )
    parser.add_argument(
        "--line",
        type=int,
        default=None,
        help="Only generate the specified 1-based line number.",
    )
    args = parser.parse_args()
    if os.path.isfile(args.ref_text):
        with open(args.ref_text, "r", encoding="utf-8") as f:
            args.ref_text = f.read().strip()
    try:
        main(args.ref_audio, args.ref_text, args.syn_text, args.syn_lang, args.line)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
