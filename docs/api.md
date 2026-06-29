# API Reference

The server lives at `http://0.0.0.0:8000`. Hit it with curl, Python, or whatever speaks HTTP.

- Interactive docs (Swagger UI): `http://localhost:8000/api-docs`
- Web UI: `http://localhost:8000/` — a dark-themed form that lets you play with everything

## GET /v1/voices — Who can speak?

Flat, searchable list of every voice from every engine. Add WAV files to `voices/` and they show up instantly — no restart needed.

```bash
curl http://localhost:8000/v1/voices
```

```json
{
  "object": "list",
  "data": [
    {"id": "Ryan", "engine": "qwen", "category": "built_in"},
    {"id": "Vivian", "engine": "qwen", "category": "built_in"},
    {"id": "af_heart", "engine": "kokoro", "category": "built_in", "language": "en-us"},
    {"id": "my_voice.wav", "engine": "qwen", "category": "cloneable", "size": 123456, "duration": 5.2, "url": "/voice/my_voice.wav"}
  ]
}
```

Cloneable entries include `size`, `duration`, `created_at`, and `url`.

---

## GET /v1/models — OpenAI-style model list

Returns friendly aliases and capabilities for the OpenAI-compatible endpoint.

```bash
curl http://localhost:8000/v1/models
```

```json
{"object": "list", "data": [
  {"id": "kokoro", "object": "model", "created": 1719000000, "owned_by": "kokoro"},
  {"id": "qwen-voice", "object": "model", "created": 1719000000, "owned_by": "qwen"}
]}
```

Add `?extras=true` for full detail (capabilities, voices, languages, install info).

---

## GET /v1/models/{model_id} — Model detail

Get a single model by ID. Returns the same shape as the list endpoint.

```bash
curl http://localhost:8000/v1/models/kokoro
```

```json
{"id": "kokoro", "object": "model", "created": 1719000000, "owned_by": "kokoro"}
```

Also supports `?extras=true` for full detail.

---

## POST /v1/audio/speech — OpenAI-compatible TTS

Drop-in replacement for [OpenAI's TTS endpoint](https://platform.openai.com/docs/api-reference/audio/createSpeech). Point your existing code at `http://localhost:8000` and it just works.

### Request body

| Field | Type | Default | What it does |
|---|---|---|---|
| `model` | string | **required** | Alias from `/v1/models` (e.g. `kokoro`, `qwen-voice`, `qwen-clone`) |
| `input` | string | **required** | Text to speak (max 4096 chars) |
| `voice` | string | `null` | Maps to `speaker_name`, `voice_description`, or `sample_voice_file` |
| `response_format` | string | `"mp3"` | `mp3` · `wav` · `pcm` |
| `speed` | number | `1.0` | `0.25` – `4.0` |

### How `voice` maps

| Model capability | `voice` becomes |
|---|---|
| `speaker` / `voice_blend` | Speaker name (e.g. `af_heart`, `Vivian`) |
| `voice_prompt` | Voice description (e.g. `"A warm, deep voice"`) |
| `voice_clone` | WAV filename in `voices/` (e.g. `my_voice.wav`) |

### Response

- **Win:** audio file with `Content-Type: audio/mpeg` (or `audio/wav`, `audio/L16`)
- **Header:** `X-Seed: <integer>` — the seed used
- **Lose:** JSON `{"detail": "..."}`

### Example

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Hello from the OpenAI-compatible endpoint!",
    "voice": "af_bella",
    "speed": 1.0
  }' --output speech.mp3
```

---

## DELETE /outputs — Clean up

Deletes all generated MP3s from `outputs/server/`. Returns what it nuked.

```bash
curl -X DELETE http://localhost:8000/outputs
```

```json
{"deleted": 5, "files": ["abc123.mp3", "def456.mp3", ...]}
```

---

## GET /health — Is it alive?

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

Yep.

---

## Errors

All errors come back as JSON:

```json
{"detail": "Something went wrong — here's what"}
```

| HTTP status | What it means |
|---|---|
| `422` | Bad request — unknown model, missing field, bad value |
| `500` | Server oops — model didn't load, ffmpeg not found, generation bombed |
