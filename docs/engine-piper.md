# Piper Engine

[Piper](https://github.com/rhasspy/piper) is a fast, local neural TTS engine using ONNX Runtime. Each voice is a separate ~80 MB model file. It offers the widest language coverage of the three engines and the fastest inference.

> Piper voices work on Apple Silicon, x86, and Linux. They can run inside Docker without performance loss.

---

## Model Download

Each voice requires two files (`.onnx` and `.onnx.json`) in `models/piper/`. Use the built-in downloader (with venv active):

```bash
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

New voices are picked up immediately without restarting the server.

Browse all 40+ supported languages: [huggingface.co/rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)

---

## Example Request

The `model` field is the exact voice stem (filename without `.onnx`):

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

No `speaker_name` needed — each Piper model file is already a single voice.

---

## Voice Naming Convention

Piper voice names follow the pattern `<lang_locale>-<voice>-<quality>`:

| Quality | File size | Use case |
|---|---|---|
| `low` | ~30 MB | Fastest, smallest |
| `medium` | ~65 MB | Good balance |
| `high` | ~130 MB | Best quality |
| `x_low` | ~12 MB | Embedded / edge |

Examples: `en_US-lessac-medium`, `de_DE-thorsten-low`, `fr_FR-siwis-medium`

---

## Speed Control

Piper uses `length_scale` internally (longer duration = slower speech):

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
- Each voice file is a single speaker — no multi-speaker models

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `Piper voice not found` | Run the download command above, check `GET /models` |
| Voice not appearing after download | Verify both `.onnx` and `.onnx.json` exist in `models/piper/` |
| Poor audio quality | Use `medium` or `high` quality instead of `low` |
