#!/bin/bash

# Fast-ACVNet Model Download Script
# This script downloads all 4 ONNX models for Fast-ACVNet and Fast-ACVNet+

BASE_URL="https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/338_Fast-ACVNet"

# Array of models to download
declare -a MODELS=(
    "fast_acvnet_onnx_no_gridsample.tar.gz:Fast-ACVNet without GridSample"
    "fast_acvnet_onnx_gridsample.tar.gz:Fast-ACVNet with GridSample"
    "fast_acvnet_plus_onnx_no_gridsample.tar.gz:Fast-ACVNet+ without GridSample"
    "fast_acvnet_plus_onnx_gridsample.tar.gz:Fast-ACVNet+ with GridSample"
)

echo "=========================================="
echo "Downloading all Fast-ACVNet models"
echo "=========================================="
echo ""

# Download and extract each model
for model_info in "${MODELS[@]}"; do
    IFS=':' read -r filename description <<< "$model_info"
    URL="${BASE_URL}/${filename}"
    
    echo "----------------------------------------"
    echo "Downloading: $description"
    echo "URL: $URL"
    echo "----------------------------------------"
    
    curl "$URL" -o "$filename"
    
    if [[ $? -ne 0 ]]; then
        echo "Error: Download failed for $description"
        continue
    fi
    
    echo "Extracting $filename..."
    tar -zxvf "$filename"
    
    if [[ $? -ne 0 ]]; then
        echo "Error: Extraction failed for $description"
        continue
    fi
    
    rm "$filename"
    echo "✓ $description completed"
    echo ""
done

echo "=========================================="
echo "All downloads finished!"
echo "=========================================="
