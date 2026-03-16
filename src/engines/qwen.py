import gc
import os
import subprocess
import threading
import warnings
from pathlib import Path

import numpy as np

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


def _load_audio_for_embedding(audio_path: str):
    """Load audio and convert to format needed for speaker embedding extraction."""
    try:
        from mlx_audio.utils import load_audio
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="mlx-audio not installed. Run: pip install mlx-audio",
        )

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    if not _convert_to_wav_24k(audio_path, tmp_path):
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail="Failed to convert voice for embedding")

    try:
        audio = load_audio(tmp_path, sample_rate=SAMPLE_RATE)
    finally:
        os.unlink(tmp_path)

    return audio


def _embedding_path(voice_file: str) -> str:
    """Return path to the cached embedding file for a voice."""
    return voice_file + ".npy"


def _load_embedding_from_disk(voice_file: str) -> mx.array | None:  # type: ignore[union-attr]
    """Load pre-computed embedding from disk if it exists."""
    emb_path = _embedding_path(voice_file)
    if os.path.exists(emb_path):
        try:
            return mx.load(emb_path)  # type: ignore[union-attr]
        except Exception:
            os.remove(emb_path)
    return None


def _save_embedding_to_disk(voice_file: str, embedding: mx.array) -> None:  # type: ignore[union-attr]
    """Save embedding to disk for future use."""
    emb_path = _embedding_path(voice_file)
    try:
        mx.save(emb_path, embedding)  # type: ignore[union-attr]
    except Exception:
        pass


def _get_or_compute_speaker_embedding(
    model, voice_file: str
) -> mx.array:  # type: ignore[union-attr]
    """Get cached speaker embedding (memory or disk) or compute and cache it."""
    cache_key = os.path.abspath(voice_file)

    with _speaker_embedding_lock:
        if cache_key in _speaker_embedding_cache:
            return _speaker_embedding_cache[cache_key]

    embedding = _load_embedding_from_disk(voice_file)
    if embedding is None:
        audio = _load_audio_for_embedding(voice_file)
        embedding = model.extract_speaker_embedding(audio, SAMPLE_RATE)
        _save_embedding_to_disk(voice_file, embedding)

    with _speaker_embedding_lock:
        _speaker_embedding_cache[cache_key] = embedding

    return embedding


def _inject_speaker_embedding(model, embedding: mx.array) -> None:  # type: ignore[union-attr]
    """Temporarily patch model to use pre-computed embedding instead of extracting from audio."""
    original_method = model.extract_speaker_embedding

    def patched_extract_speaker_embedding(audio, sr=24000):  # type: ignore[union-attr,arg-type]
        del audio, sr
        return embedding

    model.extract_speaker_embedding = patched_extract_speaker_embedding
    model._original_spk_embed_method = original_method


def _restore_speaker_embedding(model) -> None:  # type: ignore[union-attr]
    """Restore original speaker embedding method after generation."""
    if hasattr(model, "_original_spk_embed_method"):
        model.extract_speaker_embedding = model._original_spk_embed_method
        delattr(model, "_original_spk_embed_method")


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

_MODEL_CACHE_TTL = 10  # seconds to keep model in memory after last request

_cached_model = None  # loaded mlx model, or None
_cached_model_name: str | None = None
_cache_lock = threading.Lock()
_eviction_timer: threading.Timer | None = None

# Speaker embedding cache for clone mode (voice file -> mx.array embedding)
_speaker_embedding_cache: dict[str, mx.array] = {}  # type: ignore[union-attr]
_speaker_embedding_lock = threading.Lock()


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
            _speaker_embedding_cache.clear()
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

    def preload_speaker_embeddings(self) -> dict:
        """Pre-compute and cache speaker embeddings for all cloneable voices.

        Call this after model is loaded to eagerly cache embeddings for faster
        first-time clone requests. Returns dict with voice names and status.
        """
        global _cached_model
        result: dict = {"preloaded": [], "failed": []}

        if _cached_model is None:
            result["error"] = "Model not loaded"  # type: ignore[literal]
            return result

        cloneable = self.list_voices().get("cloneable", [])
        for voice_file in cloneable:
            voice_path = _resolve_voice(voice_file)
            if voice_path:
                try:
                    _get_or_compute_speaker_embedding(_cached_model, voice_path)
                    result["preloaded"].append(voice_file)
                except Exception as e:
                    result["failed"].append({"voice": voice_file, "error": str(e)})

        return result

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

                embedding = _get_or_compute_speaker_embedding(model, voice_path)
                _inject_speaker_embedding(model, embedding)

                try:
                    generate_audio(  # type: ignore[operator]
                        model=model,
                        text=text,
                        ref_audio=ref_wav,
                        ref_text=ref_text,
                        speed=speed,
                        temperature=temperature,
                        output_path=tmp_dir,
                    )
                finally:
                    _restore_speaker_embedding(model)

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
