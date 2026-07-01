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
  find "$dir" -name "*.onnx" -exec mv {} "$dir" \;
  find "$dir" -name "*.onnx.json" -exec mv {} "$dir" \;
  find "$dir" -type d -not -path "$dir" -exec rm -rf {} + 2>/dev/null || true

  echo "[piper] Done — $dir/$voice"
}

download_all_piper() {
  local filter="${1:-}"
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

  # Filter by language/locale if given
  if [[ -n "$filter" ]]; then
    local pattern
    if [[ "$filter" =~ ^[a-z][a-z]$ ]]; then
      # 2-letter code → match language prefix (en → en_US, en_GB, etc.)
      pattern="^${filter}/"
    elif [[ "$filter" =~ ^[a-z]{2}_[A-Z]{2}$ ]]; then
      # Full locale → match lang/locale/ prefix (en_US → en/en_US/)
      local lang="${filter%%_*}"
      pattern="^${lang}/${filter}/"
    else
      pattern="$filter"
    fi
    voices=$(echo "$voices" | grep "$pattern" || true)
  fi

  local count
  count=$(echo "$voices" | grep -c . || true)
  if [[ "$count" -eq 0 ]]; then
    echo "[piper] No voices matched filter '${filter:-<all>}'"
    return 1
  fi
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
  find "$dir" -name "*.onnx" -exec mv {} "$dir" \;
  find "$dir" -name "*.onnx.json" -exec mv {} "$dir" \;
  find "$dir" -type d -not -path "$dir" -exec rm -rf {} + 2>/dev/null || true

  local final_count
  final_count=$(find "$dir" -maxdepth 1 -name "*.onnx" | wc -l | tr -d ' ')
  echo "[piper] Done — $final_count voices in $dir"
}

case "${2:-kokoro}" in
  kokoro)   download_kokoro ;;
  piper)    download_piper "${3:-en_US-lessac-medium}" ;;
  piper-all) download_all_piper "${3:-}" ;;
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
    echo "  $0 models piper-all                    # all voices"
    echo "  $0 models piper-all en                 # all English voices (en_US, en_GB, etc.)"
    echo "  $0 models piper-all en_US              # only US English voices"
    exit 1
    ;;
esac
