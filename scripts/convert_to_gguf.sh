#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "🔄 Converting merged model to GGUF..."

# Base merged model path
MERGED_MODEL_DIR="merged-models/deepseek-merged"

# Ensure the base merged model directory exists
mkdir -p "$MERGED_MODEL_DIR"

# Find the latest merging directory
latest_merging_dir=$(ls "$MERGED_MODEL_DIR" | grep "merging-" | sort -V | tail -n 1)
if [ -z "$latest_merging_dir" ]; then
  echo "❌ No merging directories found in $MERGED_MODEL_DIR." >&2
  exit 1
fi

GGUF_OUTPUT_DIR="$MERGED_MODEL_DIR/$latest_merging_dir/gguf-output"

# Ensure output directory exists
mkdir -p "$GGUF_OUTPUT_DIR"

# Generate unique output file name
OUTPUT_FILE="$GGUF_OUTPUT_DIR/deepseek-q4.gguf"
if [ -f "$OUTPUT_FILE" ]; then
  suffix=$(date +%Y%m%d%H%M%S)
  OUTPUT_FILE="$GGUF_OUTPUT_DIR/deepseek-q4-$suffix.gguf"
fi

# Conversion command
if python tools/llama/transformers-to-gguf.py \
  "$MERGED_MODEL_DIR/$latest_merging_dir" \
  --outfile "$OUTPUT_FILE" \
  --outtype q8_0; then
  echo "✅ Conversion done. File saved as:"
  echo "   $OUTPUT_FILE"
else
  echo "❌ Conversion failed." >&2
  exit 1
fi
