#!/usr/bin/env python3
import os
import time
import torch
import soundfile as sf
import argparse

# Dynamically set CPU thread limits based on available cores
max_threads = max(1, (os.cpu_count() or 1) - 2)
os.environ["OMP_NUM_THREADS"] = str(max_threads)
os.environ["MKL_NUM_THREADS"] = str(max_threads)
torch.set_num_threads(max_threads)

from qwen_tts import Qwen3TTSModel


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


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
        sf.write(os.path.join(out_dir, f"{case_name}_{i}.wav"), w, sr)


def main(ref_audio: str, ref_text: str, syn_text: str, syn_lang: str = "Auto"):
    device = "cpu"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    OUT_DIR = "qwen3_tts_test_voice_clone_output_wav_cpu"
    ensure_dir(OUT_DIR)

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.float32,
        attn_implementation="eager",
    )

    # Reference audio (single)
    ref_audio_single = ref_audio

    ref_text_single = ref_text

    # Synthesis target (single)
    syn_text_single = syn_text
    syn_lang_single = syn_lang

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

    for xvec_only in [False, True]:
        mode_tag = "xvec_only" if xvec_only else "icl"

        # Case 1: prompt single + synth single, direct
        run_case(
            tts,
            OUT_DIR,
            f"case1_promptSingle_synSingle_direct_{mode_tag}",
            lambda: tts.generate_voice_clone(
                text=syn_text_single,
                language=syn_lang_single,
                ref_audio=ref_audio_single,
                ref_text=ref_text_single,
                x_vector_only_mode=xvec_only,
                **common_gen_kwargs,
            ),
        )

        # Case 1b: prompt single + synth single, via create_voice_clone_prompt
        def _case1b():
            prompt_items = tts.create_voice_clone_prompt(
                ref_audio=ref_audio_single,
                ref_text=ref_text_single,
                x_vector_only_mode=xvec_only,
            )
            return tts.generate_voice_clone(
                text=syn_text_single,
                language=syn_lang_single,
                voice_clone_prompt=prompt_items,
                **common_gen_kwargs,
            )

        run_case(
            tts,
            OUT_DIR,
            f"case1_promptSingle_synSingle_promptThenGen_{mode_tag}",
            _case1b,
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
        help="Synthesis target text.",
    )
    parser.add_argument(
        "--syn-lang",
        type=str,
        default="Auto",
        help="Synthesis language.",
    )
    args = parser.parse_args()
    # If the provided ref-text argument is a path to a file, read its contents.
    if os.path.isfile(args.ref_text):
        with open(args.ref_text, "r", encoding="utf-8") as f:
            args.ref_text = f.read().strip()
    # If the provided syn-text argument is a path to a file, read its contents.
    if os.path.isfile(args.syn_text):
        with open(args.syn_text, "r", encoding="utf-8") as f:
            args.syn_text = f.read().strip()
    try:
        main(args.ref_audio, args.ref_text, args.syn_text, args.syn_lang)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
