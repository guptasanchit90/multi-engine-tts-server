# Local TTS Server — Apple Silicon

Run high-quality, offline text-to-speech on your Mac. No cloud. No API keys. Returns MP3.

Powered by [MLX](https://github.com/ml-explore/mlx) and [ONNX Runtime](https://onnxruntime.ai/) on Apple Silicon.

---

## Engines

| Engine | Model | Voices | Best for |
|---|---|---|---|
| **Qwen3** (MLX) | 0.6B / 1.7B | 11 built-in + custom cloning | Best quality, Apple Silicon only |
| **Kokoro** (ONNX) | kokoro-v1.0 | 54 voices, 9 languages | Fast, multilingual |
| **Piper** (ONNX) | any voice | 40+ languages, ~80 MB/voice | Fastest, widest language coverage |

Engine-specific docs:

- [Qwen3 Engine](docs/engine-qwen.md) — custom voice, voice design, voice cloning
- [Kokoro Engine](docs/engine-kokoro.md) — multilingual, 54 built-in voices
- [Piper Engine](docs/engine-piper.md) — fastest inference, widest language support
- [API Reference](docs/api.md) — full endpoint documentation

---

## Requirements

- macOS with Apple Silicon (M1 / M2 / M3 / M4)
- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) — `brew install ffmpeg`

> **Docker / Linux:** Qwen3 requires the MLX Metal backend and cannot run in Docker.
> Kokoro and Piper can — see the note in each engine's doc.

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/local-tts-server.git
cd local-tts-server

pyenv local 3.13.12

python -m venv venv
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

---

## Project Structure

```
server.py               # FastAPI entry point
src/
└── engines/
    ├── base.py         # TTSEngine protocol
    ├── qwen.py         # Qwen3 engine
    ├── kokoro.py       # Kokoro engine
    └── piper.py        # Piper engine
models/                 # Downloaded model files (gitignored)
voices/                 # WAV files for voice cloning
outputs/                # Generated MP3 files
docs/                   # Engine-specific documentation
```

---

## Related Projects

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Original model by Alibaba
- [MLX Audio](https://github.com/Blaizzy/mlx-audio) — MLX audio model library
- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) — Kokoro ONNX runtime wrapper
- [Piper](https://github.com/rhasspy/piper) — Fast neural TTS
- [MLX Community](https://huggingface.co/mlx-community) — Pre-converted MLX models
