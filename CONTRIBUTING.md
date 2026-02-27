# Contributing

Contributions are welcome — bug reports, new engines, documentation improvements, and pull requests.

---

## Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/local-tts-server.git
cd local-tts-server

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

Download at least one engine's models (see the engine docs in `docs/`) before running.

---

## Adding a New Engine

1. Create `src/engines/<name>.py` and implement the `TTSEngine` protocol from `src/engines/base.py`.
2. Import and append an instance to `ENGINES` in `server.py`.
3. Update `docs/api.md` and write a `docs/engine-<name>.md`.

The protocol requires five methods: `claims()`, `list_models()`, `list_voices()`, `validate()`, `generate()`. See `src/engines/base.py` for the full contract.

---

## Code Style

- Single-file rule for `server.py` — keep it cohesive. Engines belong in `src/engines/`.
- No classes outside engine files — procedural helpers are preferred.
- Imports: stdlib → third-party, in alphabetical order within each group.
- Wrap third-party imports in `try/except ImportError` with a helpful message.
- Line length: keep under ~100 characters.
- No trailing whitespace.

---

## Reporting Bugs

Open an issue with:
- macOS version and Apple Silicon chip (e.g. M2 Pro)
- Python version (`python3 --version`)
- The exact command or request that failed
- Full error output

---

## Pull Requests

- Keep changes focused — one logical change per PR.
- Update the relevant `docs/` file if you change behaviour.
- Test with at least one engine before submitting.
