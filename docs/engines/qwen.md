# Qwen3 Engine

The Qwen3 engine runs Alibaba's Qwen3-TTS models locally via [MLX](https://github.com/ml-explore/mlx) on Apple Silicon. It supports three distinct operating modes: custom voice, voice design, and voice cloning.

> **Apple Silicon only.** MLX requires the Metal GPU backend and cannot run on x86 or inside Docker.

---

## Models

Download only the models you need and place them inside `models/qwen/`.

### Pro (1.7B) — Best quality, ~6 GB RAM

| Model folder | Mode | HuggingFace |
|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | Custom voice | [link](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit) |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` | Voice design | [link](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit) |
| `Qwen3-TTS-12Hz-1.7B-Base-8bit` | Voice cloning | [link](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit) |

### Lite (0.6B) — Faster, ~3 GB RAM

| Model folder | Mode | HuggingFace |
|---|---|---|
| `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` | Custom voice | [link](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit) |
| `Qwen3-TTS-12Hz-0.6B-Base-8bit` | Voice cloning | [link](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit) |

Download via HuggingFace Hub (with venv active):

```bash
pip install huggingface_hub
huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
  --local-dir models/qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
```

Expected layout:

```
models/
└── qwen/
    ├── Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit/
    ├── Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit/
    └── Qwen3-TTS-12Hz-1.7B-Base-8bit/
```

---

## Why MLX vs PyTorch?

| | PyTorch | MLX |
|---|---|---|
| RAM | 10+ GB | 2–3 GB |
| CPU temperature | 80–90 °C | 40–50 °C |

*Tested on M4 MacBook Air (fanless) with 1.7B models.*

---

## Mode 1: Custom Voice

Use a named built-in speaker. Optionally set tone/emotion via `voice_description`.

**Built-in speakers:** `Ryan`, `Aiden`, `Ethan`, `Chelsie`, `Serena`, `Vivian`, `Uncle_Fu`, `Dylan`, `Eric`, `Ono_Anna`, `Sohee`

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, welcome to the show.",
    "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "speaker_name": "Ryan",
    "voice_description": "Excited and upbeat",
    "speed": "normal",
    "temperature": 0.0,
    "seed": 42
  }' \
  --output speech.mp3
```

| Field | Required | Notes |
|---|---|---|
| `speaker_name` | No | Defaults to `"Vivian"` |
| `voice_description` | No | Sets tone/emotion (e.g. `"calm and professional"`) |

---

## Mode 2: Voice Design

Describe a voice in plain English — the model generates it from scratch.

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welcome to the evening news.",
    "model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
    "voice_description": "Deep, authoritative male news anchor with a slight British accent",
    "speed": "normal"
  }' \
  --output speech.mp3
```

| Field | Required | Notes |
|---|---|---|
| `voice_description` | **Yes** | Plain-English description of the voice to generate |

Example descriptions:
- `"Warm, friendly female customer support agent"`
- `"Excited young child, speaking fast"`
- `"Elderly professor, slow and thoughtful"`

---

## Mode 3: Voice Cloning

Clone any voice from a WAV reference file.

**Setup:** place a WAV file inside `voices/`. The server picks it up immediately — no restart needed.

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This will be spoken in the cloned voice.",
    "model": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "sample_voice_file": "my_voice"
  }' \
  --output speech.mp3
```

| Field | Required | Notes |
|---|---|---|
| `sample_voice_file` | **Yes** | Filename in `voices/` with or without `.wav` extension |

**Improving clone quality:** place a `.txt` sidecar alongside the WAV containing the exact words spoken in the recording:

```
voices/
├── my_voice.wav
└── my_voice.txt   ← exact transcript of the WAV audio
```

---

## Reproducibility

Fix `temperature=0.0` and a `seed` to get identical audio on every call:

```json
{ "temperature": 0.0, "seed": 42 }
```

The seed actually used is always returned in the `X-Seed` response header — useful for logging and replaying specific outputs.

For natural variation (slightly different each call):

```json
{ "temperature": 0.7 }
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `mlx_audio not found` | Run `source venv/bin/activate` |
| `Model folder not found` | Check folder name matches exactly — use `GET /models` |
| `WAV-to-MP3 conversion failed` | Run `brew install ffmpeg` |
| Poor cloning quality | Add a `.txt` transcript sidecar next to the `.wav` |
| High memory usage | Use the 0.6B Lite models instead of 1.7B Pro |
