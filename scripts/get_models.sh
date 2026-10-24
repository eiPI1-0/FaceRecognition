#!/bin/sh
# Fetch model weights into build/models/ (gitignored).
#
#   ./scripts/get_models.sh [models_dir] [--with-yunet-320]
#
# By default: the 160x120 YuNet export this project is built and measured against.
# --with-yunet-320 additionally fetches opencv_zoo's 320x320 export (per-level head; the
# code recognizes it from the model's own layer names). It is not the default because it
# is not a free upgrade — see README.md, "Swapping models".
#
# The embedder (models/mobilefacenet.onnx) and the registration photos (data/) are
# NOT downloaded here: the weights are not redistributed by this repository. Place
# them yourself — see README.md, "Assets you must supply".
set -eu

# Pinned to the upstream commit that shipped this file. The path no longer exists on
# master (the repo was restructured), so a branch-name URL 404s; a commit SHA does not.
YUNET_160_URL="https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/1688402dbd9b9fc4a3a6793810f558b7407ff384/tasks/task1/onnx/yunet_120x160.onnx"
YUNET_160_SHA256="50ef07f702a31741ca46a4c0d947773b64143b9362780237bf0d427d6c79bab7"
YUNET_160_BYTES=345478

# Also pinned to the commit that shipped it ("update yunet to v2", 2023-06-06).
# Git-LFS hosted upstream: go through github.com/<repo>/raw/<commit>/<path>, which serves
# the LFS content. raw.githubusercontent.com would hand back the 131-byte pointer file.
YUNET_320_URL="https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_320_SHA256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_320_BYTES=232589

usage() {
    echo "usage: $0 [models_dir] [--with-yunet-320]"
    echo "  models_dir        where to write (default: build/models)"
    echo "  --with-yunet-320  also fetch the 320x320 per-level YuNet export"
}

MODELS_DIR=build/models
WITH_320=0
for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        --with-yunet-320) WITH_320=1 ;;
        *) MODELS_DIR=$arg ;;
    esac
done

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

fetch() {
    dest="$MODELS_DIR/$1"
    url=$2
    want=$3
    bytes=$4

    if [ -f "$dest" ] && [ "$(sha256 "$dest")" = "$want" ]; then
        echo "already present and verified: $dest"
        return 0
    fi

    echo "downloading $1 ($bytes bytes)..."
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "$dest.part" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$dest.part" "$url"
    else
        echo "error: need curl or wget" >&2
        exit 1
    fi

    if [ "$(sha256 "$dest.part")" != "$want" ]; then
        rm -f "$dest.part"
        echo "error: checksum mismatch for $1 (expected $want)" >&2
        exit 1
    fi

    mv "$dest.part" "$dest"
    echo "ok: $dest"
}

mkdir -p "$MODELS_DIR"

fetch yunet_120x160.onnx "$YUNET_160_URL" "$YUNET_160_SHA256" "$YUNET_160_BYTES"

if [ "$WITH_320" = 1 ]; then
    fetch face_detection_yunet_2023mar.onnx "$YUNET_320_URL" "$YUNET_320_SHA256" "$YUNET_320_BYTES"
    cat <<'EOF'
to use it, put this in your config file (see README.md, "Swapping models"):
  paths.detector_model: ./models/face_detection_yunet_2023mar.onnx
  detector.input_w: 320
  detector.input_h: 320
  display.rfactor: 3
EOF
fi
