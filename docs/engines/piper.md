# Piper Engine

[Piper](https://github.com/rhasspy/piper) — the speed king. Each voice is a separate ~80 MB ONNX model. 40+ languages. Blazing fast.

> Piper runs everywhere: Apple Silicon, x86, Linux, Docker. No GPU needed.

---

## Model Download

Each voice needs two files (`.onnx` + `.onnx.json`) in `models/piper/`. Use the built-in downloader:

```bash
source venv/bin/activate

# English (US)
python -m piper.download_voices --download-dir models/piper en_US-lessac-medium

# English (UK)
python -m piper.download_voices --download-dir models/piper en_GB-alba-medium

# German
python -m piper.download_voices --download-dir models/piper de_DE-thorsten-low

# French
python -m piper.download_voices --download-dir models/piper fr_FR-siwis-medium
```

Expected layout:

```
models/
└── piper/
    ├── en_US-lessac-medium.onnx
    ├── en_US-lessac-medium.onnx.json
    ├── en_GB-alba-medium.onnx
    └── en_GB-alba-medium.onnx.json
```

New voices appear instantly — no restart needed.

Browse all voices: [huggingface.co/rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)

---

## Try it

The `model` field is the voice stem (filename without `.onnx`):

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello! This is Piper TTS running locally.",
    "model": "en_US-lessac-medium",
    "speed": "normal"
  }' \
  --output speech.mp3
```

No `speaker_name` needed — each Piper file is one voice.

---

## Voice naming

Piper voices follow `<locale>-<voice>-<quality>`:

| Quality | Size | Use for |
|---|---|---|
| `low` | ~30 MB | Speed runs |
| `medium` | ~65 MB | Daily driver |
| `high` | ~130 MB | Best quality |
| `x_low` | ~12 MB | Embedded / edge devices |

Examples: `en_US-lessac-medium`, `de_DE-thorsten-low`, `fr_FR-siwis-medium`

---

## Speed

Piper uses `length_scale` internally (longer = slower):

| Value | `length_scale` | Effective speed |
|---|---|---|
| `"slow"` | 1.4 | 0.71× |
| `"normal"` | 1.0 | 1.0× |
| `"fast"` | 0.75 | 1.33× |

---

## Limitations

- No voice cloning
- No voice design
- `temperature` and `seed` have no effect (ONNX is deterministic)
- One voice per model file — no multi-speaker models

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Voice not found | Run the download command, check `GET /models` |
| Voice not appearing | Verify both `.onnx` and `.onnx.json` exist |
| Low quality audio | Use `medium` or `high` instead of `low` |
