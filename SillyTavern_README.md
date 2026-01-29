My personal solution to use the [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) model was to create [test_model_12hz_base_server_openai.py](https://github.com/Jay4242/Qwen3-TTS_Scripts/blob/main/test_model_12hz_base_server_openai.py).

The basic instructions to get it setup was:

```
git clone https://github.com/QwenLM/Qwen3-TTS/
cd Qwen3-TTS/
python3 -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate in Windows cmd?
pip install .
```
That installs the qwen_tts library to your Python venv at `Qwen3-TTS/.venv`

Then you can download my [test_model_12hz_base_server_openai.py](https://github.com/Jay4242/Qwen3-TTS_Scripts/blob/main/test_model_12hz_base_server_openai.py) somewhere like the [Qwen3-TTS/examples](https://github.com/QwenLM/Qwen3-TTS/tree/main/examples) directory and run it.  (while the `Qwen3-TTS/.venv` is still activated)

```
python examples/test_model_12hz_base_server_openai.py --verbose
```
Which should download the [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) from hugging face (saving it wherever your system stores the huggingface cache) and start a server at `http://0.0.0.0:8000`.  It falls back to CPU if you don't have a GPU that supports CUDA.

You have to prep the voices you want to clone.

1. Download a voice sample. (Youtube with [yt-dlp](https://github.com/yt-dlp/yt-dlp) `-x` option, etc.)
2. Use a program such as Audacity to edit it down to the single subject talking.  As clean speech as you can get.  Preferably with no extra noises like stuttering, "uh's", etc.  Export this as a .wav to whichever directory you will be running the `python examples/test_model_12hz_base_server_openai.py --verbose` from.  (that's the directory it checks for samples.  ie. in `Qwen3-TTS/` if that's where you're running the script from)  Apparently the length can be short, I go up to a minute.
3. Transcribe the .wav you created to a .txt in the same directory, with the same name as the .wav.  (I whipped up [video2text.py](https://github.com/Jay4242/Qwen3-TTS_Scripts/blob/main/video2text.py) to send the .wav to a [whisper-server](https://github.com/ggml-org/whisper.cpp/tree/master/examples/server) backend and create the .txt file.  It can then be checked for accuracy.)
4. Repeat for as many voices you need.

Now that you have the .wav/.txt pairing(s) & the script running the server you can hook it up to SillyTavern as an `OpenAI Compatible` endpoint.

The `Provider Endpoint` field is your server address at port & endpoint `:8000/v1/audio/speech`.
The `Available Voices (comma separated):` field is where you enter the voices you named the .wav/.txt, without spaces between them.  Without the .wav/.txt.  So if you named a pairing GlaDOS.wav & GlaDOS.txt it would simply be `GlaDOS`.

Once you have those entered you can hit 'Reload' next to the `TTS Provider` section and you should be able to select which voice is used for the character, the user, and the default voice.
