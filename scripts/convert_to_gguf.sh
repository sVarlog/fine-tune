set -euo pipefail
IFS=$'\n\t'

#
# convert_to_gguf.sh
#
# Usage:
#   ./convert_to_gguf.sh [--outtype <q4_0|q8_0>] [--model-dir <merged-models/deepseek-merged>]
#   bash scripts/convert_to_gguf.sh --outtype <q8_0> --model-dir <merged-models/deepseek-merged>
#

# Default parameters:
OUTTYPE="q8_0"
MERGED_MODEL_BASE="merged-models/deepseek-merged"

# Helper: print usage
usage() {
  echo "Usage: $0 [--outtype <q4_0|q8_0>] [--model-dir <path>]"
  echo
  echo "  --outtype    quant type for GGUF (default: $OUTTYPE)"
  echo "  --model-dir  merged‐model root (default: $MERGED_MODEL_BASE)"
  exit 1
}

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --outtype)
      OUTTYPE="$2"; shift 2;;
    --model-dir)
      MERGED_MODEL_BASE="$2"; shift 2;;
    -h|--help)
      usage;;
    *)
      echo "Unknown option: $1"; usage;;
  esac
done

# Locate latest merging-*
if [[ ! -d "$MERGED_MODEL_BASE" ]]; then
  echo "❌ Merged models base directory not found: $MERGED_MODEL_BASE" >&2
  exit 1
fi

# Use sort -V to handle numeric suffixes correctly
latest=$(find "$MERGED_MODEL_BASE" -maxdepth 1 -type d -name 'merging-*' | sort -V | tail -n1)
if [[ -z "$latest" ]]; then
  echo "❌ No 'merging-*' directories found under $MERGED_MODEL_BASE" >&2
  exit 1
fi

echo "🔄 Converting merged model in: $latest (type=$OUTTYPE)"

GGUF_DIR="$latest/gguf-output"
mkdir -p "$GGUF_DIR"

# Build the output filename
base_name="deepseek-${OUTTYPE}.gguf"
outfile="$GGUF_DIR/$base_name"

if [[ -f "$outfile" ]]; then
  timestamp=$(date +'%Y%m%dT%H%M%S')
  outfile="$GGUF_DIR/deepseek-${OUTTYPE}-${timestamp}.gguf"
fi

# Locate the conversion script in your repo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONVERTER="${REPO_ROOT}/tools/llama/convert_hf_to_gguf.py"

if [[ ! -f "$CONVERTER" ]]; then
  echo "❌ Converter not found at $CONVERTER" >&2
  exit 1
fi

# Finally, run it
echo "➡️  Running: python \"$CONVERTER\" …"
if python "$CONVERTER" "$latest" \
     --outfile "$outfile" \
     --outtype "$OUTTYPE"; then
  echo "✅ Conversion successful!"
  echo "   Saved to: $outfile"
else
  echo "❌ Conversion failed." >&2
  exit 1
fi