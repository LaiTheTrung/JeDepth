#!/bin/bash

set -e

echo "=== Downloading models ==="

OUT_DIR="."
mkdir -p $OUT_DIR
cd $OUT_DIR

download () {
    ID=$1
    NAME=$2

    echo ">>> Downloading $NAME"

    for i in {1..3}; do
        gdown "https://drive.google.com/uc?id=$ID" -O "$NAME" && break
        echo "Retry $i..."
        sleep 2
    done

    echo "✓ Done: $NAME"
}


# =========================
# Download files
# =========================

download "1W1V1H64l9bAi97boEQQ2ueNzzGmSMz-E" "model_best_bp2_serialize.pth"

download "1GDBRYL-ZaLpXEtWfGFRJvkBc_2sywjgj" "cfg.yaml"

echo "=== Done all downloads ==="