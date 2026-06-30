#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${1:-models}"

download_kokoro() {
  local dir="$MODELS_DIR/kokoro"
  mkdir -p "$dir"
  echo "[kokoro] Downloading model files..."
  curl -fsSL "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" \
    -o "$dir/kokoro-v1.0.onnx"
  curl -fsSL "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" \
    -o "$dir/voices-v1.0.bin"
  echo "[kokoro] Done — $dir"
}

download_piper() {
  local voice="${1:-en_US-lessac-medium}"
  local dir="$MODELS_DIR/piper"
  mkdir -p "$dir"
  echo "[piper] Downloading $voice..."
  curl -fsSL "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/$voice/$voice.onnx" \
    -o "$dir/$voice.onnx"
  curl -fsSL "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/$voice/$voice.onnx.json" \
    -o "$dir/$voice.onnx.json"
  echo "[piper] Done — $dir/$voice"
}

case "${2:-kokoro}" in
  kokoro) download_kokoro ;;
  piper)  download_piper "${3:-en_US-lessac-medium}" ;;
  both)
    download_kokoro
    download_piper "${3:-en_US-lessac-medium}"
    ;;
  *)
    echo "Usage: $0 [models_dir] {kokoro|piper|both} [piper_voice]"
    echo ""
    echo "Examples:"
    echo "  $0 models kokoro"
    echo "  $0 models piper en_US-lessac-medium"
    echo "  $0 models both"
    exit 1
    ;;
esac
