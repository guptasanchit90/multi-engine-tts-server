# Documentation

[← Back to README](../README.md)

---

## API Reference

- [Full API Reference](api.md) — all endpoints, request/response schemas, examples

## Engines

| Engine | Type | Doc |
|---|---|---|
| **Qwen3** | MLX (Apple Silicon) | [Guide](engines/qwen.md) |
| **Kokoro** | ONNX | [Guide](engines/kokoro.md) |
| **Piper** | ONNX | [Guide](engines/piper.md) |
| **Chatterbox Turbo** | MLX (Apple Silicon) | [Guide](engines/chatterbox.md) |

## Development

- [Contributing Guide](../CONTRIBUTING.md) — adding engines, code style, conventions
- [Development Setup](development.md) — environment setup, linting, testing

## Tools

- [Postman Collection](postman/collection.json) — pre-configured API requests

---

## Quick Links

| Endpoint | Description |
|---|---|
| `http://localhost:8000/docs` | Interactive Swagger UI |
| `http://localhost:8000/` | Web UI form |
| `GET /health` | Health check |
| `GET /models` | List available models |
| `GET /voices` | List voices per engine |
| `POST /tts` | Generate speech |
| `POST /v1/audio/speech` | OpenAI-compatible endpoint |
