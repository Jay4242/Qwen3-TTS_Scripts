# Voice Webchat (Node.js)

Record audio in the browser, transcribe it with a whisper.cpp server, send the chat history to an LLM, and synthesize a spoken reply with Qwen TTS.

## Prerequisites

- Node.js 18+
- A running whisper.cpp server (for transcription)
- A running LLM server compatible with the OpenAI SDK
- A running Qwen TTS server
- Reference voice files (configurable via `REF_AUDIO_PATH` / `REF_TEXT_PATH`,
  defaults to `../filename.wav` and `../filename.txt` from the `nodejs/` directory)

## Setup

```bash
cd nodejs
npm install
```

## Run

```bash
cd nodejs
npm start
```

Then open `http://localhost:3000`.

## Configuration

Environment variables (see `nodejs/.env` for defaults):

- `APP_TITLE`: Page title shown in the UI (default: `Voice Webchat`)
- `PORT`: Port to bind the web server (default: `3000`)
- `WHISPER_SERVER_URL`: whisper.cpp inference endpoint (default: `http://127.0.0.1:9191/inference`)
- `WHISPER_TRANSLATE`: Set to `true`/`1` to request translation to English (default: `false`)
- `WHISPER_TIMEOUT_SECONDS`: Whisper request timeout in seconds (default: `1800`)
- `LLM_BASE_URL`: Base URL of the LLM server (default: `http://127.0.0.1:9090/v1`)
- `LLM_API_KEY`: API key for the LLM server (default: `LAN`)
- `LLM_MODEL`: LLM model name (default: `qwen3`)
- `LLM_SYSTEM`: System prompt (default: `You are a helpful assistant. Use the following transcription to respond conversationally.`)
- `LLM_TIMEOUT_SECONDS`: LLM request timeout in seconds (default: `1800`)
- `QWEN_TTS_URL`: Qwen TTS base URL (default: `http://127.0.0.1:8000`)
- `QWEN_TTS_LANG`: Default synthesis language (default: `Auto`)
- `TTS_TIMEOUT_SECONDS`: TTS request timeout in seconds (default: `1800`)
- `REF_AUDIO_PATH`: Path to reference WAV audio (default: `../filename.wav`)
- `REF_TEXT_PATH`: Path to reference transcript (default: `../filename.txt`)

## Rolling chat behavior

The browser stores the full chat history and sends it to the backend on each request. This means the prompt context will grow over time. There is a **Clear chat** button in the UI to reset the conversation. A future improvement could cap or summarize context server-side.

## Notes

- Microphone access requires HTTPS or `localhost`.
- Ensure the TTS reference audio and text are accessible in the repository root before starting the server.
