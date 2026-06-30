# Qwen3 Engine

Alibaba's Qwen3-TTS, running locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). Three modes: custom voice, voice design, voice cloning. Premium quality.

> **Apple Silicon only.** MLX needs the Metal GPU — no x86, no Docker.

---

## Models

Pick what you need, download it, drop it in `models/qwen/`.

### Pro (1.7B) — Best quality, ~6 GB RAM

| Model folder | Mode | Download |
|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | Custom voice | [HuggingFace](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit) |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` | Voice design | [HuggingFace](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit) |
| `Qwen3-TTS-12Hz-1.7B-Base-8bit` | Voice cloning | [HuggingFace](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit) |

### Pro (1.7B) — Additional options

| Model folder | Mode | Size | Alias |
|---|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-Base-4bit` | Voice cloning | ~1.6 GB | `qwen-clone` |

Quantized clone model — lower RAM, slightly lower quality.

### Lite (0.6B) — Faster, ~3 GB RAM

| Model folder | Mode | Alias |
|---|---|---|
| `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` | Custom voice | `qwen-lite` |
| `Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit` | Voice design | `qwen-lite-voice` |
| `Qwen3-TTS-12Hz-0.6B-Base-8bit` | Voice cloning | `qwen-lite-clone` |

Download via HF Hub:

```bash
source venv/bin/activate
pip install -U huggingface_hub
hf download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
  --local-dir models/qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
```

Expected layout:

```
models/
└── qwen/
    ├── Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit/
    ├── Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit/
    ├── Qwen3-TTS-12Hz-1.7B-Base-8bit/
    ├── Qwen3-TTS-12Hz-1.7B-Base-4bit/
    ├── Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit/
    ├── Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit/
    └── Qwen3-TTS-12Hz-0.6B-Base-8bit/
```

---

## MLX vs PyTorch — why MLX?

| | PyTorch | MLX |
|---|---|---|
| RAM | 10+ GB | 2–3 GB |
| CPU temp | 80–90 °C (ouch) | 40–50 °C (cozy) |

*Measured on an M4 MacBook Air (fanless) with 1.7B models. MLX keeps it cool.*

---

## Mode 1: Custom Voice

Pick a built-in speaker. Optionally set the vibe with `voice_description`.

**Built-in speakers:** `Ryan`, `Aiden`, `Ethan`, `Chelsie`, `Serena`, `Vivian`, `Uncle_Fu`, `Dylan`, `Eric`, `Ono_Anna`, `Sohee`

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-voice",
    "input": "Hello, welcome to the show.",
    "voice": "Ryan",
    "speed": 1.0
  }' \
  --output speech.mp3
```

| Field | Required | Notes |
|---|---|---|
| `voice` | No | Speaker name (defaults to `"Vivian"`); can include tone via `|` separator e.g. `"Ryan|calm and professional"` |

---

## Mode 2: Voice Design

Describe a voice in plain English. The model builds it on the fly.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-voice",
    "input": "Welcome to the evening news.",
    "voice": "Deep, authoritative male news anchor with a slight British accent",
    "speed": 1.0
  }' \
  --output speech.mp3
```

| Field | Required | Notes |
|---|---|---|
| `voice` | **Yes** | Plain English — describe the voice you want |

Try these:
- `"Warm, friendly female customer support agent"`
- `"Excited young child, speaking fast"`
- `"Elderly professor, slow and thoughtful"`

---

## Mode 3: Voice Cloning

Clone any voice from a WAV file. Drop it in `voices/`, call the API — that's it.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-clone",
    "input": "This will be spoken in the cloned voice.",
    "voice": "my_voice"
  }' \
  --output speech.mp3
```

| Field | Required | Notes |
|---|---|---|
| `voice` | **Yes** | Filename in `voices/` (`.wav` extension optional) |

**Pro tip for better clones:** Add a `.txt` sidecar with the exact words spoken:

```
voices/
├── my_voice.wav
└── my_voice.txt   ← transcript of what's in the WAV
```

**Performance:** Speaker embeddings are cached to disk as `.npy` files alongside WAVs. The first clone of a voice is slower; subsequent requests reuse the cached embedding.

---

## Reproducibility

Want the exact same audio every time?

```json
{ "temperature": 0.0, "seed": 42 }
```

The seed used is always in the `X-Seed` response header — handy for replaying specific outputs.

Want natural variation (different each call)?

```json
{ "temperature": 0.7 }
```

---

## Limitations

- Apple Silicon only — no x86, no Docker
- Voice Design needs a descriptive `voice` field (~10 words minimum recommended)
- Voice Cloning needs a WAV at least 5 seconds long
- `temperature` only works in Custom Voice mode; Design and Cloning are deterministic

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `mlx_audio not found` | Run `source venv/bin/activate` |
| `Model folder not found` | Check the folder name — use `GET /v1/models` |
| WAV-to-MP3 failed | Run `brew install ffmpeg` |
| Bad clone quality | Add a `.txt` transcript next to the `.wav` |
| High memory usage | Switch to 0.6B Lite models |
