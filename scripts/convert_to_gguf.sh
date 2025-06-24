#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "🔄 Converting merged model to GGUF..."

# Set paths
MODEL_PATH="merged-models/deepseek-merged"
OUTPUT_DIR="$MODEL_PATH/gguf-output"
OUTPUT_FILE="$OUTPUT_DIR/deepseek-q4.gguf"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run conversion
if python tools/llama/transformers-to-gguf.py \
  "$MODEL_PATH" \
  --outfile "$OUTPUT_FILE" \
  --outtype q8_0; then
  echo "✅ Conversion done. File saved as:"
  echo "   $OUTPUT_FILE"
else
  echo "❌ Conversion failed." >&2
  exit 1
fi
