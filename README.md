# Qwen3‑TTS Example Scripts

This repository contains a collection of example scripts for the **Qwen3‑TTS** models.  
Before running any of the scripts you must clone and install the Qwen3‑TTS library.

## Prerequisites

- Python 3.9 or newer
- Git
- A virtual environment (recommended)

## Setup Steps

1. **Clone the Qwen3‑TTS repository**

   ```bash
   git clone https://github.com/QwenLM/Qwen3-TTS.git
   cd Qwen3-TTS
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows use: .venv\Scripts\activate
   ```

3. **Install the package**

   ```bash
   pip install .
   ```

   This will install `qwen_tts` and its dependencies.

## Running the Example Scripts

Typical usage:

```bash
python test_model_12hz_base_server_openai.py
```

Each script provides its own `--help` output describing required arguments, e.g.:

```bash
python qwen-tts-client.py --help
```

## Notes

- The scripts assume the Qwen3‑TTS models are available via the Hugging Face hub.  
- For GPU usage ensure that CUDA is installed and that `torch` detects the device.  
- If you encounter import errors, verify that the virtual environment is activated.

Happy experimenting!
