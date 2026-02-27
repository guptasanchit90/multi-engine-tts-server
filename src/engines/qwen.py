import gc
import os
import subprocess
import threading
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# MLX is Apple Silicon only and unavailable inside Docker / on x86.
# We import at module level so type checkers and linters are satisfied,
# but the guard `if not _MLX_AVAILABLE` in generate() ensures these are
# never called when absent.
_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    from mlx_audio.tts.generate import generate_audio
    from mlx_audio.tts.utils import load_model
    _MLX_AVAILABLE = True
except ImportError:
    pass

try:
    from fastapi import HTTPException
except ImportError as exc:
    raise ImportError("fastapi is not installed. Run: pip install -r requirements.txt") from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.getcwd(), "models", "qwen")
VOICES_DIR = os.path.join(os.getcwd(), "voices")
SAMPLE_RATE = 24000

# Maps model folder name → generation mode
_MODELS: dict[str, str] = {
    # Pro (1.7B)
    "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit": "custom",
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit": "design",
    "Qwen3-TTS-12Hz-1.7B-Base-8bit":        "clone",
    # Lite (0.6B)
    "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit": "custom",
    "Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit": "design",
    "Qwen3-TTS-12Hz-0.6B-Base-8bit":        "clone",
}

_SPEAKERS: set[str] = {
    "Ryan", "Aiden", "Ethan", "Chelsie", "Serena", "Vivian",
    "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _model_path(folder_name: str) -> str | None:
    """Resolve the on-disk path for a model folder (handles HF snapshot layout)."""
    full = os.path.join(MODELS_DIR, folder_name)
    if not os.path.exists(full):
        return None
    snapshots = os.path.join(full, "snapshots")
    if os.path.exists(snapshots):
        subs = [f for f in os.listdir(snapshots) if not f.startswith(".")]
        if subs:
            return os.path.join(snapshots, subs[0])
    return full


def _resolve_voice(filename: str) -> str | None:
    """Find a voice WAV in VOICES_DIR by bare name or full filename."""
    for candidate in [filename, f"{filename}.wav"]:
        full = os.path.join(VOICES_DIR, candidate)
        if os.path.exists(full):
            return full
    return None


def _convert_to_wav_24k(input_path: str, output_path: str) -> bool:
    """Re-encode any audio to 24 kHz mono WAV via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", input_path,
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _clean_memory() -> None:
    gc.collect()


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

_MODEL_CACHE_TTL = 10  # seconds to keep model in memory after last request

_cached_model = None  # loaded mlx model, or None
_cached_model_name: str | None = None
_cache_lock = threading.Lock()
_eviction_timer: threading.Timer | None = None


def _evict_model() -> None:
    """Called by the eviction timer to free the cached model."""
    global _cached_model, _cached_model_name, _eviction_timer
    with _cache_lock:
        if _cached_model is not None:
            print(f"[qwen] Evicting cached model '{_cached_model_name}' after {_MODEL_CACHE_TTL}s idle")
            del _cached_model
            _cached_model = None
            _cached_model_name = None
            _eviction_timer = None
            _clean_memory()


def _reschedule_eviction() -> None:
    """Cancel any pending eviction and start a fresh countdown. Must be called under _cache_lock."""
    global _eviction_timer
    if _eviction_timer is not None:
        _eviction_timer.cancel()
    _eviction_timer = threading.Timer(_MODEL_CACHE_TTL, _evict_model)
    _eviction_timer.daemon = True
    _eviction_timer.start()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class QwenEngine:
    """TTS engine wrapping Qwen3-TTS models via mlx-audio."""

    # --- TTSEngine protocol ---

    def claims(self, model: str) -> bool:
        return model in _MODELS

    def list_models(self) -> list[dict]:
        return [
            {
                "engine": "qwen",
                "model": folder,
                "mode": mode,
                "available": _model_path(folder) is not None,
            }
            for folder, mode in _MODELS.items()
        ]

    def list_voices(self) -> dict:
        """Return built-in speakers and cloneable WAV files, structured by category."""
        cloneable = sorted(
            f for f in os.listdir(VOICES_DIR)
            if f.lower().endswith(".wav") and not f.startswith(".")
        ) if os.path.exists(VOICES_DIR) else []

        return {
            "built_in": sorted(_SPEAKERS),
            "cloneable": cloneable,
        }

    def validate(self, request: dict) -> None:
        mode = _MODELS[request["model"]]

        if mode == "custom":
            speaker = request.get("speaker_name")
            if speaker and speaker not in _SPEAKERS:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown speaker '{speaker}'. Valid: {sorted(_SPEAKERS)}",
                )

        elif mode == "design":
            if not request.get("voice_description"):
                raise HTTPException(
                    status_code=422,
                    detail="'voice_description' is required for VoiceDesign models",
                )

        elif mode == "clone":
            if not request.get("sample_voice_file"):
                raise HTTPException(
                    status_code=422,
                    detail="'sample_voice_file' is required for Base (cloning) models",
                )
            if not _resolve_voice(request["sample_voice_file"]):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Voice file '{request['sample_voice_file']}' not found in {VOICES_DIR}. "
                        "Place a .wav file there and retry."
                    ),
                )

    def generate(self, request: dict, tmp_dir: str) -> str:
        """Run inference; return path to audio_000.wav inside tmp_dir."""
        if not _MLX_AVAILABLE:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Qwen3 engine requires MLX (Apple Silicon only). "
                    "Use the Kokoro or Piper engine instead."
                ),
            )

        model_name = request["model"]
        mode = _MODELS[model_name]
        speed = request["speed_value"]          # float, resolved by server
        temperature = request["temperature"]
        text = request["text"]

        model_path = _model_path(model_name)
        if not model_path:
            raise HTTPException(
                status_code=422,
                detail=f"Model folder '{model_name}' not found in {MODELS_DIR}",
            )

        # Seed the MLX PRNG for reproducibility
        mx.random.seed(request["effective_seed"])  # type: ignore[union-attr]

        # --- acquire cached model (load only if model changed or not yet loaded) ---
        with _cache_lock:
            global _cached_model, _cached_model_name
            if _cached_model_name != model_name:
                # Different model requested — evict the existing one first
                if _cached_model is not None:
                    print(f"[qwen] Switching model: '{_cached_model_name}' → '{model_name}'")
                    del _cached_model
                    _cached_model = None
                    _cached_model_name = None
                    _clean_memory()
                try:
                    print(f"[qwen] Loading model '{model_name}' into cache")
                    _cached_model = load_model(Path(model_path))  # type: ignore[operator]
                    _cached_model_name = model_name
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Model load failed: {e}")
            else:
                print(f"[qwen] Using cached model '{model_name}'")
            model = _cached_model  # type: ignore[assignment]

        try:
            if mode == "custom":
                generate_audio(  # type: ignore[operator]
                    model=model,
                    text=text,
                    voice=request.get("speaker_name") or "Vivian",
                    instruct=request.get("voice_description") or "Normal tone",
                    speed=speed,
                    temperature=temperature,
                    output_path=tmp_dir,
                )

            elif mode == "design":
                generate_audio(  # type: ignore[operator]
                    model=model,
                    text=text,
                    instruct=request["voice_description"],
                    speed=speed,
                    temperature=temperature,
                    output_path=tmp_dir,
                )

            elif mode == "clone":
                voice_path = _resolve_voice(request["sample_voice_file"])
                assert voice_path is not None  # guaranteed by validate()

                ref_wav = os.path.join(tmp_dir, "ref_converted.wav")
                if not _convert_to_wav_24k(voice_path, ref_wav):
                    raise HTTPException(status_code=500, detail="Failed to convert reference audio")

                txt_path = os.path.splitext(voice_path)[0] + ".txt"
                ref_text = "."
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as fh:
                        ref_text = fh.read().strip() or "."

                generate_audio(  # type: ignore[operator]
                    model=model,
                    text=text,
                    ref_audio=ref_wav,
                    ref_text=ref_text,
                    speed=speed,
                    temperature=temperature,
                    output_path=tmp_dir,
                )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

        # --- reschedule eviction timer after successful generation ---
        with _cache_lock:
            _reschedule_eviction()

        wav_path = os.path.join(tmp_dir, "audio_000.wav")
        if not os.path.exists(wav_path):
            raise HTTPException(status_code=500, detail="TTS produced no output file")

        return wav_path
