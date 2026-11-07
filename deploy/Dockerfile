# FaceRecognition server image.
#
# Two stages: the builder carries the whole devel toolchain, the runtime image
# carries only the binary + OpenCV's runtime package + assets.
#
# Build (from the repo root):
#   podman build -f deploy/Dockerfile -t facerec:1.2.0 .
#   docker build -f deploy/Dockerfile -t facerec:1.2.0 .
#
# Run (tokens file is REQUIRED: a non-loopback bind without auth refuses to start):
#   docker run -d -p 8080:8080 \
#       -v "$PWD/build/models:/opt/facerec/models:ro" \
#       -v "$PWD/build/data:/opt/facerec/data:ro" \
#       -v facerec-index:/data \
#       -v "$PWD/deploy/tokens:/etc/facerec/tokens:ro" \
#       facerec:1.2.0
#
# Verify with:  docker run --rm facerec:1.2.0 --help

# ---------------------------------------------------------------- build stage
FROM opensuse/leap:16.0 AS builder

RUN zypper -n install --no-recommends cmake make gcc-c++ opencv-devel && \
    zypper clean -a

WORKDIR /src
COPY . .

RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j"$(nproc)"

# -------------------------------------------------------------- runtime stage
FROM opensuse/leap:16.0

# On openSUSE Leap 16, one package (`libopencv413`) carries every module .so —
# there are no per-module `libopencv_core413`-style packages (verified with
# `rpm -qf /usr/lib64/libopencv_core.so.413` on the build host).
RUN zypper -n install --no-recommends libopencv413 && \
    zypper clean -a && \
    useradd -r -U -s /sbin/nologin -d /opt/facerec facerec

WORKDIR /opt/facerec

# The CMake target is named `main`; it is installed as `facerec` (the CLI's own
# usage text already calls itself `facerec serve ...`).
COPY --from=builder /src/build/main ./facerec
COPY --from=builder /src/scripts/get_models.sh ./scripts/get_models.sh
COPY --from=builder /src/config.example.yml ./config.example.yml

# Default config. Relative asset paths resolve against paths.root, so the
# process can be started from anywhere as long as this file is passed.
#   - gallery.idx lives on a volume so enrollment survives restarts
#   - auth.tokens_file is deliberately set: binding 0.0.0.0 without a token
#     table is refused by design, so the image fails loud instead of running open
RUN mkdir -p models data /data && \
    printf '%s\n' \
        'paths.root: /opt/facerec' \
        'paths.gallery_index: /data/gallery.idx' \
        'auth.tokens_file: /etc/facerec/tokens' \
        > /etc/facerec.yml && \
    chown -R facerec:facerec /opt/facerec /data && \
    chmod 0644 /etc/facerec.yml

USER facerec

# 8080 = API + /mcp; the same port serves /healthz, /readyz and /metrics.
EXPOSE 8080

# --workers defaults to CPU count; override with `docker run -e` style args or
# an explicit `--workers N` after the image name. Assets must be mounted at
# /opt/facerec/models and /opt/facerec/data (see the run example above).
ENTRYPOINT ["/opt/facerec/facerec", "serve",
            "--config", "/etc/facerec.yml",
            "--host", "0.0.0.0",
            "--port", "8080",
            "--reuseport",
            "--quiet"]
