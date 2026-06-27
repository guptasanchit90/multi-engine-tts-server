import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import uvicorn
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, field_validator
except ImportError:
    print("Error: 'fastapi' or 'uvicorn' not found.")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

from src.audio import trim_silence, wav_to_mp3, wav_to_pcm
from src.engines.base import discover
from src.utils import VOICES_DIR, convert_to_wav_24k, get_audio_duration, resolve_voice, scan_wav_voices

import src.engines  # noqa: F401

ENGINES = discover()

OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs", "server")
SPEED_MAP = {
    "slow":   0.8,
    "normal": 1.0,
    "fast":   1.3,
}

os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI(
    title="Local TTS Server",
    description="Multi-engine, offline text-to-speech server with OpenAI-compatible endpoint.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Model manifest — friendly aliases for OpenAI-compatible endpoint
# Each entry maps an alias to an internal engine+model pair and declares
# capabilities that the web UI uses to show the relevant voice fields.
# ---------------------------------------------------------------------------

MODEL_MANIFEST: dict[str, dict] = {
    "kokoro": {
        "id": "kokoro",
        "name": "Kokoro 82M",
        "engine": "kokoro",
        "model": "kokoro-v1.0",
        "mode": "speaker",
        "capabilities": ["speaker", "voice_blend"],
        "description": "Fast ONNX-based TTS with 9 languages and voice blending",
    },
    "qwen-voice": {
        "id": "qwen-voice",
        "name": "Qwen3 Pro Voice Design",
        "engine": "qwen",
        "model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        "mode": "design",
        "capabilities": ["voice_prompt"],
        "description": "Synthesize voices from natural language descriptions",
    },
    "qwen-clone": {
        "id": "qwen-clone",
        "name": "Qwen3 Pro Voice Clone",
        "engine": "qwen",
        "model": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "mode": "clone",
        "capabilities": ["voice_clone"],
        "description": "Clone a voice from a reference WAV sample",
    },
    "qwen-lite": {
        "id": "qwen-lite",
        "name": "Qwen3 Lite",
        "engine": "qwen",
        "model": "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
        "mode": "custom",
        "capabilities": ["speaker"],
        "description": "Lightweight 0.6B model with preset speakers",
    },
    "chatterbox": {
        "id": "chatterbox",
        "name": "Chatterbox Turbo",
        "engine": "chatterbox",
        "model": "chatterbox-turbo-fp16",
        "mode": "clone",
        "capabilities": ["voice_clone"],
        "description": "High-quality voice cloning via MLX (5s+ reference recommended)",
    },
    "piper": {
        "id": "piper",
        "name": "Piper TTS",
        "engine": "piper",
        "model": "",
        "mode": "speaker",
        "capabilities": ["speaker"],
        "description": "Lightweight ONNX-based TTS, 100+ downloadable voices",
    },
}

# Optional mapping from OpenAI standard names to our aliases
OPENAI_MODEL_ALIASES: dict[str, str] = {
    "tts-1": "kokoro",
    "tts-1-hd": "qwen-voice",
}


def _with_availability(entries: list[dict]) -> list[dict]:
    result = []
    for e in entries:
        entry = dict(e)
        entry["available"] = False
        for engine in ENGINES:
            if engine.engine_name == entry["engine"]:
                if entry["model"]:
                    for m in engine.list_models():
                        if m["model"] == entry["model"]:
                            entry["available"] = m["available"]
                            break
                else:
                    entry["available"] = any(m["available"] for m in engine.list_models())
                break
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class TTSRequest(BaseModel):
    text: str
    model: str

    speaker_name: str | None = None
    voice_description: str | None = None
    sample_voice_file: str | None = None

    speed: str = "normal"
    temperature: float = 0.0
    seed: int | None = None
    add_pauses: bool = True

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


class OpenAIRequest(BaseModel):
    model: str
    input: str
    voice: str | None = None
    response_format: str = "mp3"
    speed: float = 1.0

    @field_validator("input")
    @classmethod
    def input_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("'input' must not be empty")
        return v

    @field_validator("response_format")
    @classmethod
    def format_valid(cls, v: str) -> str:
        if v not in ("mp3", "wav", "pcm"):
            raise ValueError(f"Unsupported response_format '{v}' — must be mp3, wav, or pcm")
        return v

    @field_validator("speed")
    @classmethod
    def speed_in_range(cls, v: float) -> float:
        if v < 0.25 or v > 4.0:
            raise ValueError("'speed' must be between 0.25 and 4.0")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_engine(model: str):
    for engine in ENGINES:
        if engine.claims(model):
            return engine
    known = sorted(m["model"] for e in ENGINES for m in e.list_models())
    raise HTTPException(
        status_code=422,
        detail=f"Unknown model '{model}'. Known models: {known}",
    )


def _resolve_openai_model(model_input: str) -> dict:
    resolved = OPENAI_MODEL_ALIASES.get(model_input, model_input)
    entry = MODEL_MANIFEST.get(resolved)
    if not entry:
        known = sorted(MODEL_MANIFEST)
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{model_input}'. Known: {known}",
        )
    return entry


def _openai_to_internal(req: OpenAIRequest, manifest: dict) -> dict:
    if req.speed <= 0.8:
        speed_key = "slow"
    elif req.speed >= 1.21:
        speed_key = "fast"
    else:
        speed_key = "normal"

    d: dict = {
        "text": req.input,
        "speed": speed_key,
        "speed_value": req.speed,
        "temperature": 0.0,
        "seed": None,
        "add_pauses": True,
        "speaker_name": None,
        "voice_description": None,
        "sample_voice_file": None,
    }

    caps = manifest["capabilities"]
    is_piper = manifest["engine"] == "piper"

    if is_piper:
        d["model"] = req.voice or ""
    else:
        d["model"] = manifest["model"]

    if req.voice:
        if "voice_clone" in caps:
            d["sample_voice_file"] = req.voice
        elif "voice_prompt" in caps:
            d["voice_description"] = req.voice
        elif "speaker" in caps or "voice_blend" in caps:
            d["speaker_name"] = req.voice

    return d


# ---------------------------------------------------------------------------
# Routes — original API (backward compatible)
# ---------------------------------------------------------------------------


@app.post("/tts", summary="Generate speech", response_description="MP3 audio file")
async def tts(req: TTSRequest):
    engine = _find_engine(req.model)

    effective_seed = req.seed if req.seed is not None else int(time.time() * 1000) & 0xFFFFFFFF
    request_dict = {
        **req.model_dump(),
        "speed_value": SPEED_MAP[req.speed],
        "effective_seed": effective_seed,
    }

    engine.validate(request_dict)

    tmp_dir = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        wav_path = await asyncio.to_thread(engine.generate, request_dict, tmp_dir)
        trim_silence(wav_path)

        mp3_filename = f"{uuid.uuid4().hex}.mp3"
        mp3_path = os.path.join(OUTPUTS_DIR, mp3_filename)
        if not wav_to_mp3(wav_path, mp3_path):
            raise HTTPException(
                status_code=500,
                detail="WAV-to-MP3 conversion failed — is ffmpeg installed?",
            )
        params = {"model": req.model, "input": req.text, "speed": req.speed, "seed": effective_seed}
        try:
            with open(mp3_path + ".json", "w") as fh:
                json.dump(params, fh)
        except OSError:
            pass
    except HTTPException:
        raise
    except Exception as e:
        print(f"[server] Error in /tts ({req.model}): {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
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
    result = []
    for engine in ENGINES:
        result.extend(engine.list_models())
    return JSONResponse(content=result)


@app.get("/voices", summary="List all voices grouped by engine")
def list_voices():
    result = {}
    for engine in ENGINES:
        name = engine.engine_name
        voices = engine.list_voices()
        if voices:
            result[name] = voices
    return JSONResponse(content=result)


@app.delete("/outputs", summary="Delete all generated audio files")
def delete_outputs():
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
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Routes — voice management
# ---------------------------------------------------------------------------

MAX_UPLOAD_SIZE = 50 * 1024 * 1024


@app.post("/voice", summary="Upload a voice file (any audio format)")
async def upload_voice(file: UploadFile = File(...), name: str | None = Form(None)):
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=422, detail=f"File exceeds {MAX_UPLOAD_SIZE // (1024*1024)} MB limit")

    stem = name if name else (file.filename or "voice")
    stem = "".join(c for c in stem.rsplit(".", 1)[0] if c.isalnum() or c in "-_.").rstrip(".") or "voice"
    safe_name = stem + ".wav"

    target = os.path.join(VOICES_DIR, safe_name)
    if os.path.exists(target):
        raise HTTPException(
            status_code=409,
            detail=f"Voice '{safe_name}' already exists. Use DELETE /voice/{safe_name} first to replace it.",
        )

    is_wav = len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WAVE"

    if is_wav:
        try:
            with open(target, "wb") as f:
                f.write(content)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    else:
        fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename or ".dat")[1])
        os.close(fd)
        try:
            with open(tmp_path, "wb") as f:
                f.write(content)
            if not convert_to_wav_24k(tmp_path, target):
                if os.path.exists(target):
                    os.unlink(target)
                raise HTTPException(
                    status_code=422,
                    detail="Could not convert file to WAV — is it a valid audio file? ffmpeg must be installed.",
                )
        finally:
            os.unlink(tmp_path)

    duration = get_audio_duration(target)
    size = os.path.getsize(target)
    created_at = os.path.getmtime(target)
    url = f"/voice/{safe_name}"

    return {
        "name": safe_name,
        "duration": round(duration, 1),
        "size": size,
        "created_at": created_at,
        "url": url,
    }


@app.get("/voice/{name:path}", summary="Get a voice file")
def get_voice(name: str):
    path = resolve_voice(name)
    if not path:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    return FileResponse(path, media_type="audio/wav")


@app.put("/voice/{name:path}", summary="Rename a voice file")
def rename_voice(name: str, new_name: str):
    path = resolve_voice(name)
    if not path:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")

    safe_new = "".join(c for c in new_name if c.isalnum() or c in "-_.").rstrip(".") or "voice"
    if not safe_new.endswith(".wav"):
        safe_new += ".wav"

    new_path = os.path.join(VOICES_DIR, safe_new)
    if os.path.exists(new_path):
        raise HTTPException(status_code=409, detail=f"Voice '{safe_new}' already exists")

    os.rename(path, new_path)
    old_emb = path + ".npy"
    if os.path.exists(old_emb):
        os.rename(old_emb, new_path + ".npy")
    return {"name": safe_new, "url": f"/voice/{safe_new}"}


@app.delete("/voice/{name:path}", summary="Delete a voice file")
def delete_voice(name: str):
    path = resolve_voice(name)
    if not path:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
    os.remove(path)
    embedding = path + ".npy"
    if os.path.exists(embedding):
        os.remove(embedding)
    return {"deleted": name}


@app.get("/voices/detail", summary="List all cloneable voices with metadata")
def list_voices_detail():
    files = scan_wav_voices(VOICES_DIR)
    result = []
    for f in files:
        path = os.path.join(VOICES_DIR, f)
        size = os.path.getsize(path)
        duration = get_audio_duration(path)
        created_at = os.path.getmtime(path)
        result.append({
            "name": f,
            "size": size,
            "duration": round(duration, 1),
            "created_at": created_at,
            "url": f"/voice/{f}",
        })
    result.sort(key=lambda e: e["created_at"], reverse=True)
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Routes — output management
# ---------------------------------------------------------------------------

AUDIO_EXTS = {".mp3", ".wav", ".pcm"}


@app.get("/outputs/detail", summary="List generated outputs with metadata")
def list_outputs_detail():
    if not os.path.exists(OUTPUTS_DIR):
        return JSONResponse(content=[])

    entries = []
    for fname in os.listdir(OUTPUTS_DIR):
        fpath = os.path.join(OUTPUTS_DIR, fname)
        if not os.path.isfile(fpath) or fname.startswith("."):
            continue
        if fname.endswith(".json"):
            continue

        ext = os.path.splitext(fname)[1].lower()
        if ext not in AUDIO_EXTS:
            continue

        size = os.path.getsize(fpath)
        created_at = os.path.getmtime(fpath)

        params = {}
        json_path = fpath + ".json"
        if os.path.exists(json_path):
            try:
                with open(json_path) as fh:
                    params = json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass

        duration = get_audio_duration(fpath) if ext in (".wav", ".mp3") else 0.0
        entries.append({
            "name": fname,
            "size": size,
            "duration": round(duration, 1),
            "created_at": created_at,
            "url": f"/output/{fname}",
            "params": params,
        })

    entries.sort(key=lambda e: e["created_at"], reverse=True)
    return JSONResponse(content=entries)


@app.get("/output/{filename:path}", summary="Get a generated output file")
def get_output(filename: str):
    fpath = os.path.join(OUTPUTS_DIR, filename)
    real = os.path.realpath(fpath)
    if not real.startswith(os.path.realpath(OUTPUTS_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.isfile(real):
        raise HTTPException(status_code=404, detail=f"Output '{filename}' not found")

    ext = os.path.splitext(filename)[1].lower()
    media_type = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".pcm": "audio/L16"}.get(ext, "application/octet-stream")
    return FileResponse(real, media_type=media_type)


@app.delete("/output/{filename:path}", summary="Delete a generated output file")
def delete_output(filename: str):
    fpath = os.path.join(OUTPUTS_DIR, filename)
    real = os.path.realpath(fpath)
    if not real.startswith(os.path.realpath(OUTPUTS_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.isfile(real):
        raise HTTPException(status_code=404, detail=f"Output '{filename}' not found")
    os.remove(real)
    json_path = real + ".json"
    if os.path.exists(json_path):
        os.remove(json_path)
    return {"deleted": filename}


# ---------------------------------------------------------------------------
# Routes — OpenAI-compatible API
# ---------------------------------------------------------------------------


@app.get("/v1/models", summary="List available models (OpenAI-compatible)")
def list_v1_models():
    entries = _with_availability(list(MODEL_MANIFEST.values()))
    return JSONResponse(content=entries)


@app.post("/v1/audio/speech", summary="Generate speech (OpenAI-compatible)")
async def openai_speech(req: OpenAIRequest):
    manifest = _resolve_openai_model(req.model)

    caps = manifest.get("capabilities", [])

    if not req.voice and ("voice_clone" in caps or "voice_prompt" in caps):
        detail_parts = []
        if "voice_clone" in caps:
            detail_parts.append(
                "'voice' is required for voice cloning models. "
                "Specify a .wav filename from the voices/ directory."
            )
        if "voice_prompt" in caps:
            detail_parts.append(
                "'voice' is required for voice design models. "
                "Specify a natural language voice description."
            )
        raise HTTPException(status_code=422, detail=" ".join(detail_parts))

    request_dict = _openai_to_internal(req, manifest)

    engine = _find_engine(request_dict["model"])

    effective_seed = int(time.time() * 1000) & 0xFFFFFFFF
    request_dict["effective_seed"] = effective_seed

    engine.validate(request_dict)

    tmp_dir = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        wav_path = await asyncio.to_thread(engine.generate, request_dict, tmp_dir)
        trim_silence(wav_path)

        output_id = uuid.uuid4().hex
        content_type = "audio/mpeg"
        filename = "speech.mp3"

        if req.response_format == "wav":
            output_path = os.path.join(OUTPUTS_DIR, f"{output_id}.wav")
            shutil.copy2(wav_path, output_path)
            content_type = "audio/wav"
            filename = "speech.wav"
        elif req.response_format == "pcm":
            output_path = os.path.join(OUTPUTS_DIR, f"{output_id}.pcm")
            if not wav_to_pcm(wav_path, output_path):
                raise HTTPException(
                    status_code=500,
                    detail="WAV-to-PCM conversion failed — is ffmpeg installed?",
                )
            content_type = "audio/L16"
            filename = "speech.pcm"
        else:
            output_path = os.path.join(OUTPUTS_DIR, f"{output_id}.mp3")
            if not wav_to_mp3(wav_path, output_path):
                raise HTTPException(
                    status_code=500,
                    detail="WAV-to-MP3 conversion failed — is ffmpeg installed?",
                )

        params = {"model": req.model, "input": req.input, "voice": req.voice, "speed": req.speed, "seed": effective_seed}
        try:
            with open(output_path + ".json", "w") as fh:
                json.dump(params, fh)
        except OSError:
            pass
    except HTTPException:
        raise
    except Exception as e:
        print(f"[server] Error in openai_speech ({req.model}): {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return FileResponse(
        path=output_path,
        media_type=content_type,
        filename=filename,
        headers={"X-Seed": str(effective_seed)},
    )


app.mount("/", StaticFiles(directory="static", html=True), name="ui")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
