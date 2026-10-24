#!/bin/sh
# Fetch the detector this project is built against into build/models/.
#
# The embedder (models/mobilefacenet.onnx) and the registration photos (data/) are
# NOT downloaded here: the weights are not redistributed by this repository. Place
# them yourself — see README.md, "Assets you must supply".
set -eu

# Pinned to the upstream commit that shipped this file. The path no longer exists on
# master (the repo was restructured), so a branch-name URL 404s; a commit SHA does not.
YUNET_URL="https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/1688402dbd9b9fc4a3a6793810f558b7407ff384/tasks/task1/onnx/yunet_120x160.onnx"
YUNET_SHA256="50ef07f702a31741ca46a4c0d947773b64143b9362780237bf0d427d6c79bab7"
YUNET_BYTES=345478

MODELS_DIR=${1:-build/models}
DEST="$MODELS_DIR/yunet_120x160.onnx"

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

mkdir -p "$MODELS_DIR"

if [ -f "$DEST" ] && [ "$(sha256 "$DEST")" = "$YUNET_SHA256" ]; then
    echo "already present and verified: $DEST"
    exit 0
fi

echo "downloading YuNet detector (160x120, $YUNET_BYTES bytes)..."
if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$DEST.part" "$YUNET_URL"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$DEST.part" "$YUNET_URL"
else
    echo "error: need curl or wget" >&2
    exit 1
fi

if [ "$(sha256 "$DEST.part")" != "$YUNET_SHA256" ]; then
    echo "error: checksum mismatch for $DEST.part (expected $YUNET_SHA256)" >&2
    exit 1
fi

mv "$DEST.part" "$DEST"
echo "ok: $DEST"
