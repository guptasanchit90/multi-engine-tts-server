import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import warnings

# Suppress harmless library warnings before any heavy imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel, field_validator
except ImportError:
    print("Error: 'fastapi' or 'uvicorn' not found.")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

from src.engines.kokoro import KokoroEngine
from src.engines.piper import PiperEngine
from src.engines.qwen import QwenEngine

# ---------------------------------------------------------------------------
# Engine registry
# To add a new TTS engine:
#   1. Implement the TTSEngine protocol in src/engines/<name>.py
#   2. Import and append an instance here
# ---------------------------------------------------------------------------

ENGINES = [
    QwenEngine(),
    KokoroEngine(),
    PiperEngine(),
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs", "server")
SPEED_MAP = {
    "slow":   0.8,
    "normal": 1.0,
    "fast":   1.3,
}

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Local TTS Server",
    description=(
        "Multi-engine, offline text-to-speech server.\n\n"
        "**Engines:** Qwen3 (MLX · Apple Silicon), Kokoro (ONNX), Piper (ONNX)\n\n"
        "All `/tts` requests return an `audio/mpeg` (MP3) file."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class TTSRequest(BaseModel):
    """
    Unified request schema for all TTS engines.

    Which optional fields are required depends on the engine and model:
      - speaker_name      — named speaker (e.g. Qwen CustomVoice, Kokoro)
      - voice_description — describe or style the voice (Qwen VoiceDesign / emotion)
      - sample_voice_file — filename in voices/ for cloning (Qwen Base)
    """

    text: str
    model: str

    speaker_name: str | None = None
    voice_description: str | None = None
    sample_voice_file: str | None = None

    speed: str = "normal"
    temperature: float = 0.0
    seed: int | None = None

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("'text' must not be empty")
        return v

    @field_validator("speed")
    @classmethod
    def speed_must_be_valid(cls, v: str) -> str:
        v = v.lower()
        if v not in SPEED_MAP:
            raise ValueError(f"'speed' must be one of: {list(SPEED_MAP)}")
        return v

    @field_validator("temperature")
    @classmethod
    def temperature_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("'temperature' must be >= 0")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_engine(model: str):
    """Return the engine that claims this model, or raise HTTP 422."""
    for engine in ENGINES:
        if engine.claims(model):
            return engine
    known = sorted(m["model"] for e in ENGINES for m in e.list_models())
    raise HTTPException(
        status_code=422,
        detail=f"Unknown model '{model}'. Known models: {known}",
    )


def _wav_to_mp3(wav_path: str, mp3_path: str) -> bool:
    """Convert a WAV file to MP3 via ffmpeg. Returns True on success."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", wav_path,
        "-codec:a", "libmp3lame", "-qscale:a", "2",
        mp3_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/tts", summary="Generate speech", response_description="MP3 audio file")
async def tts(req: TTSRequest):
    """
    Generate speech and return an MP3 file (`audio/mpeg`).

    The engine is selected automatically from the `model` field.
    Use `GET /models` to see all available models and which engine handles each.

    **Speed values:** `"slow"` (0.8×) · `"normal"` (1.0×) · `"fast"` (1.3×)

    **Reproducibility:** set `temperature=0` (default) and a fixed `seed` for
    identical output on every call. The seed actually used is echoed in the
    `X-Seed` response header.
    """
    engine = _find_engine(req.model)

    effective_seed = req.seed if req.seed is not None else int(time.time() * 1000) & 0xFFFFFFFF
    request_dict = {
        **req.model_dump(),
        "speed_value": SPEED_MAP[req.speed],
        "effective_seed": effective_seed,
    }

    # Engine validates its own fields before any heavy model loading
    engine.validate(request_dict)

    tmp_dir = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        wav_path = engine.generate(request_dict, tmp_dir)

        mp3_filename = f"{uuid.uuid4().hex}.mp3"
        mp3_path = os.path.join(OUTPUTS_DIR, mp3_filename)
        if not _wav_to_mp3(wav_path, mp3_path):
            raise HTTPException(
                status_code=500,
                detail="WAV-to-MP3 conversion failed — is ffmpeg installed?",
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return FileResponse(
        path=mp3_path,
        media_type="audio/mpeg",
        filename="speech.mp3",
        headers={"X-Seed": str(effective_seed)},
    )


@app.get("/models", summary="List available models")
def list_models():
    """List all models across all registered engines with on-disk availability."""
    result = []
    for engine in ENGINES:
        result.extend(engine.list_models())
    return JSONResponse(content=result)


@app.get("/voices", summary="List all voices grouped by engine")
def list_voices():
    """
    Return all voices grouped by engine and category.

    ```json
    {
      "qwen": {
        "built_in":  ["Aiden", "Chelsie", ...],
        "cloneable": ["my_voice.wav", ...]
      },
      "kokoro": {
        "en-us": ["af_heart", "af_bella", ...],
        "en-gb": ["bf_emma", ...]
      },
      "piper": {
        "en_US": ["en_US-lessac-medium"],
        "en_GB": ["en_GB-alba-medium"]
      }
    }
    ```

    Cloneable voice files and Piper voices are read at request time —
    drop a new file in the right directory and it appears immediately.
    """
    result = {}
    for engine in ENGINES:
        name = engine.__class__.__name__.replace("Engine", "").lower()
        voices = engine.list_voices()
        if voices:
            result[name] = voices
    return JSONResponse(content=result)


@app.delete("/outputs", summary="Delete all generated audio files")
def delete_outputs():
    """
    Delete all MP3 files from the `outputs/server/` directory.

    Returns the number of files deleted and a list of their filenames.
    The directory itself is preserved.
    """
    deleted = []
    errors = []
    try:
        entries = os.listdir(OUTPUTS_DIR)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not read outputs directory: {e}")

    for filename in entries:
        filepath = os.path.join(OUTPUTS_DIR, filename)
        if os.path.isfile(filepath):
            try:
                os.remove(filepath)
                deleted.append(filename)
            except OSError as e:
                errors.append({"file": filename, "error": str(e)})

    response: dict = {"deleted": len(deleted), "files": deleted}
    if errors:
        response["errors"] = errors
    return JSONResponse(content=response)


@app.get("/health", summary="Health check")
def health():
    """Returns `{"status": "ok"}` when the server is running."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
