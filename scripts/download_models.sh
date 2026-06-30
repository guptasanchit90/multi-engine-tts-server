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

  # Parse voice slug: en_US-lessac-medium → locale=en_US, voice_name=lessac, quality=medium
  local locale="${voice%%-*}"
  local rest="${voice#*-}"
  local quality="${rest##*-}"
  local voice_name="${rest%-*}"
  local lang="${locale%%_*}"

  hf download "rhasspy/piper-voices" \
    "$lang/$locale/$voice_name/$quality/$voice.onnx" \
    --local-dir "$dir"
  hf download "rhasspy/piper-voices" \
    "$lang/$locale/$voice_name/$quality/$voice.onnx.json" \
    --local-dir "$dir"

  # Flatten: hf preserves subdirs, but engine expects files in models/piper/
  find "$dir" -name "*.onnx" -maxdepth 3 -exec mv {} "$dir" \;
  find "$dir" -name "*.onnx.json" -maxdepth 3 -exec mv {} "$dir" \;
  find "$dir" -type d -not -path "$dir" -exec rm -rf {} + 2>/dev/null || true

  echo "[piper] Done — $dir/$voice"
}

download_all_piper() {
  local dir="$MODELS_DIR/piper"
  mkdir -p "$dir"
  echo "[piper] Fetching voice list..."

  local voices
  voices=$(curl -sf "https://huggingface.co/api/models/rhasspy/piper-voices" \
    | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data.get('siblings', []):
    f = s['rfilename']
    if f.endswith('.onnx') and not f.endswith('.json'):
        print(f)
" 2>/dev/null) || { echo "[piper] Failed to fetch voice list"; exit 1; }

  local count
  count=$(echo "$voices" | wc -l | tr -d ' ')
  echo "[piper] Downloading all $count voices..."

  echo "$voices" | while IFS= read -r voice_path; do
    local voice_file
    voice_file=$(basename "$voice_path")
    local voice_name="${voice_file%.onnx}"

    echo "  Downloading $voice_name..."
    hf download "rhasspy/piper-voices" "$voice_path" --local-dir "$dir" >/dev/null 2>&1
    hf download "rhasspy/piper-voices" "${voice_path}.json" --local-dir "$dir" >/dev/null 2>&1
  done

  # Flatten
  find "$dir" -name "*.onnx" -maxdepth 3 -exec mv {} "$dir" \;
  find "$dir" -name "*.onnx.json" -maxdepth 3 -exec mv {} "$dir" \;
  find "$dir" -type d -not -path "$dir" -exec rm -rf {} + 2>/dev/null || true

  local final_count
  final_count=$(find "$dir" -maxdepth 1 -name "*.onnx" | wc -l | tr -d ' ')
  echo "[piper] Done — $final_count voices in $dir"
}

case "${2:-kokoro}" in
  kokoro)   download_kokoro ;;
  piper)    download_piper "${3:-en_US-lessac-medium}" ;;
  piper-all) download_all_piper ;;
  both)
    download_kokoro
    download_piper "${3:-en_US-lessac-medium}"
    ;;
  *)
    echo "Usage: $0 [models_dir] {kokoro|piper|piper-all|both} [piper_voice]"
    echo ""
    echo "Examples:"
    echo "  $0 models kokoro"
    echo "  $0 models piper en_US-lessac-medium"
    echo "  $0 models piper-all"
    echo "  $0 models both"
    exit 1
    ;;
esac
