# Sonus — Speak freely

Multi-engine, offline text-to-speech (+ speech-to-text) on your Mac. No cloud. No API keys. No one listening.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](.python-version)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-black)](https://www.apple.com/mac/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guptasanchit90/sonus/blob/main/colab_quickstart.ipynb)

Runs on [MLX](https://github.com/ml-explore/mlx) and [ONNX Runtime](https://onnxruntime.ai/) — Apple Silicon for MLX engines, **any platform** for ONNX engines (Kokoro, Piper).

> **About the name:** *Sonus* is Latin for "sound" (/ˈsoː.nus/). Felt right for a project about making machines talk.
>
> **ℹ️** Not affiliated with any company, service, or organization named Sonus. Just a coincidence. We're an independent open-source thing.
>
> *[Opencode](https://opencode.ai) — vibecoded by AI, tested by humans.*

---

## What is this?

Sonus turns text into speech (and speech into text) using whatever engine you throw at it. Five engines, one API. Run it locally, hit the endpoint, get audio back. Zero data leaves your machine.

Think of it as a **local speech hub** — TTS via Qwen3, Kokoro, Piper, and Chatterbox Turbo; STT via Whisper MLX. All offline, all local.

---

## Engines at a glance

| Engine | Framework | Voices | Vibe |
|---|---|---|---|
| **Qwen3** | MLX | 11 built-in + custom cloning | Premium quality. Sounds almost human. 🍎 Apple Silicon only. |
| **Kokoro** | ONNX | 54 voices, 9 languages | The multilingual workhorse. Fast, reliable. ✅ Cross-platform. |
| **Piper** | ONNX | 40+ languages, ~80 MB/voice | The speed demon. 100+ languages, tiny footprint. ✅ Cross-platform. |
| **Chatterbox Turbo** | MLX | Voice cloning only | Best-in-class cloning. Feed it a WAV, get a twin. 🍎 Apple Silicon only. |
| **Whisper MLX** | MLX | 5 model sizes (tiny→large-v3) | Speech-to-text. Transcribe anything. 🍎 Apple Silicon only. |

More on each engine:
- [Qwen3](docs/engines/qwen.md) — custom voice, voice design, voice cloning
- [Kokoro](docs/engines/kokoro.md) — 54 built-in voices, 9 languages
- [Piper](docs/engines/piper.md) — fastest inference, widest language support
- [Chatterbox Turbo](docs/engines/chatterbox.md) — voice cloning via MLX
- [Whisper MLX](docs/engines/whisper.md) — speech-to-text via MLX
- [API Reference](docs/api.md) — full endpoint docs

---

## What you'll need

**Local (macOS Apple Silicon):**
- A Mac with Apple Silicon (M1, M2, M3, M4 — anything with Metal)
- Python 3.13+ (`brew install python@3.13`)
- [ffmpeg](https://ffmpeg.org/) — `brew install ffmpeg`

**Colab (any platform):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guptasanchit90/sonus/blob/main/colab_quickstart.ipynb)
- Kokoro and Piper engines only (MLX engines require Apple Silicon)
- No GPU needed — runs on CPU, but T4 GPU available if you want it
- No setup beyond clicking the badge above

> **Running in Docker?** Only Kokoro and Piper work there. Qwen3 and Chatterbox need the Metal GPU backend. See each engine's doc for details.

---

## Get started in 30 seconds

```bash
git clone https://github.com/YOUR_USERNAME/sonus.git
cd sonus

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

Download models for at least one engine (check the engine docs above), then:

```bash
source venv/bin/activate
python server.py
# Listening on http://0.0.0.0:8000
```

Interactive API docs: **http://localhost:8000/api-docs**

### Or run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guptasanchit90/sonus/blob/main/colab_quickstart.ipynb)

Click the badge above and follow the notebook. It installs everything, downloads models, starts the server, and gives you a public URL via localtunnel.

Includes engine selection (Kokoro, Piper, or both) — no Apple Silicon required.

---

## Try it

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "model": "kokoro-v1.0", "speaker_name": "af_heart"}' \
  --output hello.mp3
```

### Speaking OpenAI's language

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Hello world", "voice": "af_bella"}' \
  --output hello.mp3
```

Drop-in replacement for `POST /v1/audio/speech`. Your existing OpenAI TTS code works without changes — just point it at `http://localhost:8000`.

### Speech to text (OpenAI-compatible)

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@speech.mp3" \
  -F "model=whisper-base" \
  -F "language=en" \
  -F "temperature=0.0"
```

```json
{"text": "Hello world, this was transcribed locally."}
```

---

## Documentation

| Resource | What's inside |
|---|---|
| [API Reference](docs/api.md) | Every endpoint, schema, curl example |
| [Docs Home](docs/index.md) | Full docs index |
| [Development Guide](docs/development.md) | Setup, linting, testing, Docker |
| [Contributing](CONTRIBUTING.md) | Adding engines, code style, PRs |

---

## How it's built

```
server.py               # FastAPI — the brain
src/
  engines/
    base.py             # The contract every TTS engine signs
    qwen.py             # Qwen3 (MLX)
    chatterbox.py       # Chatterbox Turbo (MLX)
    kokoro.py           # Kokoro (ONNX)
    piper.py            # Piper (ONNX)
  stt/
    base.py             # The contract every STT engine signs
    whisper_mlx.py      # Whisper via MLX (Apple Silicon)
static/                 # Web UI — Vue 3 (CDN), no build step
docs/                   # Docs that don't suck
models/                 # Downloaded models (gitignored)
voices/                 # WAVs for voice cloning
outputs/                # Generated audio files
```

---

## Built with

| Tool | Role | How we use it |
|---|---|---|
| [Opencode](https://opencode.ai) | AI pair programmer | Vibecoded most of this thing |
| [VS Code](https://code.visualstudio.com) | Editor | Where the magic happens |
| [FastAPI](https://fastapi.tiangolo.com) | Web framework | Routes, validation, docs |
| [Uvicorn](https://www.uvicorn.org) | ASGI server | Serves it all up |
| [MLX](https://github.com/ml-explore/mlx) | ML framework | Apple Silicon superpowers |
| [mlx-audio](https://github.com/Blaizzy/mlx-audio) | Audio model loader | Loads Qwen3, Chatterbox models |
| [pydub](https://github.com/jiaaro/pydub) | Audio conversion | WAV ↔ MP3 magic |
| [soundfile](https://python-soundfile.readthedocs.io) | WAV I/O | Reads and writes WAVs |
| [Piper](https://github.com/rhasspy/piper) | TTS engine | Speed king, ONNX-powered |
| [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) | TTS engine | Multilingual, ONNX-powered |
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | TTS engine | Premium quality, MLX-powered |
| [Chatterbox Turbo](https://huggingface.co/mlx-community/Chatterbox-Turbo-TTS-fp16) | TTS engine | Cloning specialist, MLX-powered |
| [mlx-whisper](https://github.com/ml-explore/mlx-whisper) | STT engine | Speech-to-text, MLX-powered |
| [Whisper](https://github.com/openai/whisper) | STT model | OpenAI's transcription model |

---

## Disclaimer

Yes, this thing can make audio that sounds like real people. **You're responsible for what you do with it.**

- **Don't** impersonate people without their consent.
- **Don't** create deceptive, fraudulent, or misleading content.
- **Do** respect the laws where you live.
- The authors assume **zero liability** for misuse.

Use it wisely. Or don't — but that's on you.

---

## License

MIT — do what you want with it, just keep the notice.
