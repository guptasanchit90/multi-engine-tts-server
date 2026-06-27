# Local TTS Server — Apple Silicon

Run high-quality, offline text-to-speech on your Mac. No cloud. No API keys. Returns MP3.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](.python-version)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-black)](https://www.apple.com/mac/)

Powered by [MLX](https://github.com/ml-explore/mlx) and [ONNX Runtime](https://onnxruntime.ai/) on Apple Silicon.

---

## Features

- **4 engines** — Qwen3, Kokoro, Piper, Chatterbox Turbo
- **OpenAI-compatible endpoint** — drop-in replacement for `POST /v1/audio/speech`
- **Dark-themed web UI** — vanilla JS, no build step, served at `http://localhost:8000/`
- **Voice cloning & design** — clone from WAV, describe voice in plain English, or pick built-in speakers
- **Multi-language** — 40+ languages via Piper, 9 via Kokoro
- **Docker support** — Kokoro & Piper run on any platform via Docker

---

## Engines

| Engine | Framework | Voices | Best for |
|---|---|---|---|
| **Qwen3** | MLX | 11 built-in + custom cloning | Best quality, Apple Silicon only |
| **Kokoro** | ONNX | 54 voices, 9 languages | Fast, multilingual |
| **Piper** | ONNX | 40+ languages, ~80 MB/voice | Fastest, widest language coverage |
| **Chatterbox Turbo** | MLX | Voice cloning only | High-quality cloning, Apple Silicon only |

Engine documentation:

- [Qwen3](docs/engines/qwen.md) — custom voice, voice design, voice cloning
- [Kokoro](docs/engines/kokoro.md) — multilingual, 54 built-in voices
- [Piper](docs/engines/piper.md) — fastest inference, widest language support
- [Chatterbox Turbo](docs/engines/chatterbox.md) — voice cloning via MLX
- [API Reference](docs/api.md) — full endpoint documentation

---

## Requirements

- macOS with Apple Silicon (M1–M4)
- Python 3.13+ (`brew install python@3.13`)
- [ffmpeg](https://ffmpeg.org/) — `brew install ffmpeg`

> **Docker / Linux:** Qwen3 and Chatterbox require the MLX Metal backend and cannot run in Docker. Kokoro and Piper can — see the note in each engine's doc.

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/local-tts-server.git
cd local-tts-server

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

Download at least one engine's models (see engine docs), then:

```bash
source venv/bin/activate
python server.py
# Listening on http://0.0.0.0:8000
```

Interactive API docs: **http://localhost:8000/docs**

---

## Minimal Example

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "model": "kokoro-v1.0", "speaker_name": "af_heart"}' \
  --output hello.mp3
```

### OpenAI-compatible

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Hello world", "voice": "af_bella"}' \
  --output hello.mp3
```

---

## Documentation

| Resource | Description |
|---|---|
| [API Reference](docs/api.md) | All endpoints, request/response schemas, curl examples |
| [Docs Home](docs/index.md) | Full documentation index |
| [Development Guide](docs/development.md) | Setup, linting, testing, Docker |
| [Contributing](CONTRIBUTING.md) | Adding engines, code style, pull requests |
| [Postman Collection](docs/postman/collection.json) | Pre-configured API requests |

---

## Project Structure

```
server.py               # FastAPI entry point
src/
  engines/
    base.py             # TTSEngine protocol + @register
    qwen.py             # Qwen3 (MLX)
    chatterbox.py       # Chatterbox Turbo (MLX)
    kokoro.py           # Kokoro (ONNX)
    piper.py            # Piper (ONNX)
static/                 # Web UI (vanilla JS, no build)
docs/                   # API reference + per-engine docs
models/                 # Downloaded model files (gitignored)
voices/                 # WAV files for voice cloning
outputs/                # Generated MP3 files
```

---

## Related Projects

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Original model by Alibaba
- [MLX Audio](https://github.com/Blaizzy/mlx-audio) — MLX audio model library
- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) — Kokoro ONNX runtime wrapper
- [Piper](https://github.com/rhasspy/piper) — Fast neural TTS
- [Chatterbox-Turbo-TTS](https://huggingface.co/mlx-community/Chatterbox-Turbo-TTS-fp16) — MLX-converted Chatterbox

---

## License

MIT
