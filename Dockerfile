# ---------------------------------------------------------------------------
# Local TTS Server — Docker image (Kokoro + Piper only)
#
# The Qwen3 engine requires the Apple Silicon Metal GPU via MLX and cannot
# run inside Docker. Kokoro and Piper use CPU-based ONNX Runtime and work
# on any platform (x86, ARM, Linux).
#
# Build:
#   docker build -t tts-server .
#
# Run (mount your models/ and voices/ directories):
#   docker run -p 8000:8000 \
#     -v "$(pwd)/models:/app/models" \
#     -v "$(pwd)/voices:/app/voices" \
#     -v "$(pwd)/outputs:/app/outputs" \
#     tts-server
#
# The server starts automatically. Access it at http://localhost:8000
# Interactive docs: http://localhost:8000/docs
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# System dependencies: ffmpeg for WAV→MP3 conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies — exclude MLX (Apple Silicon only)
COPY requirements.txt .
RUN pip install --no-cache-dir \
        fastapi \
        uvicorn \
        kokoro-onnx \
        piper-tts \
        soundfile \
        numpy \
        scipy \
    && pip install --no-cache-dir \
        "pydantic>=2.0"

# Copy application source
COPY server.py .
COPY src/ src/

# Runtime directories (models/voices/outputs are expected to be mounted)
RUN mkdir -p models/kokoro models/piper voices outputs/server

# Environment: disable Qwen engine at runtime by setting a flag
# The engine gracefully skips itself when models are not present,
# but we also suppress MLX import errors in Docker explicitly.
ENV TOKENIZERS_PARALLELISM=false \
    TTS_DISABLE_QWEN=1

EXPOSE 8000

CMD ["python", "server.py"]
