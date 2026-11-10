# FaceRecognition — a lightweight, high-concurrency face recognition server in C++

**Lightweight. High concurrency. Accurate at a stated operating point. MCP-native.** All on CPU,
with one dependency (OpenCV's `dnn`) — no dlib, no ncnn, no TensorFlow / PyTorch / ONNX Runtime.

| | Measured on this machine (16-core x86_64, CPU only, OpenCV 4.13.0, `-O3`) |
|---|---|
| **Lightweight** | **1.3 MiB** of our own code, **8.1 MiB** of weights (YuNet 0.33 + MobileFaceNet 7.8), **5.7 ms** for one face in one frame (detect 0.93 + align/embed 4.47 + letterbox 0.27), **112 MB** RSS at 4 workers. No GPU, no model runtime, no Python at runtime. The *linked* closure is OpenCV's, not ours: 298 shared objects on this distro, almost all of them FFmpeg/GStreamer I/O that a `core`/`imgproc`/`dnn`-only OpenCV build does not pull in. |
| **High concurrency** | p50 **74 ms** at 4 workers (≈135 req/s), ≈193 req/s at 8, and **1347 req/s** on repeated requests through the SHA-256 result cache. One pipeline per worker *is* the backpressure: overload answers `503 + Retry-After` inside a single end-to-end budget instead of queueing without bound. |
| **Accurate, at an operating point you choose** | **1.3%** false accepts against a 610-person gallery at `recognize.thresh = 0.45` (at the previous default `0.2`: **99.7%** — the reason it changed), 3.0% at 1680 identities; 1:1 EER at threshold ≈ 0.18 / rate ≈ 4.1%. Held-out photos, the production code path. |
| **MCP-native** | `./main mcp` serves JSON-RPC 2.0 tools over stdio or HTTP — four read-only tools with JSON Schemas, in-process so a tool call costs one forward pass rather than a cold model load. Enrollment is deliberately **not** exposed to a prompt-injectable caller. |

The accuracy belongs to the models. What this repository adds is that the number is **real**:
5-point similarity alignment onto the ArcFace template, letterboxed preprocessing shared by
registration and inference, and a threshold picked from held-out photos rather than from a 1:1
separability harness (the two differ by two orders of magnitude at 1:N). Every figure above names
its dataset, its gallery size and its threshold, and every property below is backed by a build
target, a startup check or an endpoint — not by prose.

```
camera frame ──▶ YuNet detector ──▶ 5-point similarity alignment ──▶ MobileFaceNet
                   (160×120)              (112×112 crop)                (512-d embedding)
                                                  │
              tracker (two-gate association, stable IDs, gap filling)
                                                  │
                                    cosine similarity vs. per-person fused feature
```

Prior-box decoding, NMS, alignment, the tracker, the matcher, the HTTP server and the MCP server
are all written here: **55 files / 8.7k lines of C++17**, and every layer boundary is also a
CMake target — the build graph is the architecture, not a description of it.

## Properties, and what enforces each one

| Property | Enforced by |
|---|---|
| A `matched` answer is a statement about **your** gallery, not a bare cosine. At startup, `serve`/`mcp` compute what the configured threshold does to *the gallery they loaded* and log a warning when it is too loose; `/readyz` answers the same numbers. | `check_recognize_threshold` (`src/app/api.cpp`) → startup audit line + `/readyz` (`far_1n_estimate`, `far_1n_resolution`) |
| Accuracy claims come from photos the gallery never saw. | `./main evaluate <dataset>` (first N photos enrolled, next M held out); the report's own figures are asserted in `tests/evaluation.cpp` |
| A `nan` score or a phantom box cannot reach a client, even though the pinned YuNet export computes 6 of its 1076 anchors unreliably. | two load-bearing gates in `PriorBox::decode` (a score gate written so NaN *fails* it; a geometry gate, because `iou_score` is clamped and garbage confidence can forge 0.99) — mutation-tested in `tests/decode_gates.cpp` |
| Overload returns `503 + Retry-After` inside **one** end-to-end budget; it never queues without bound, and health probes are never shed. | one admission budget applied at both waiting points (`src/serve/admission.hpp`, `timed_queue.hpp`), accounting asserted in `tests/admission.cpp` |
| Predicted boxes are drawn but never recognized: a face the detector did not see this frame cannot acquire a label from an embedding of a guess. | `TrackView::measured` contract; labels cached per track id and overwritten — including "no match" — on every measured frame |
| The architecture cannot quietly rot back into a tangle. | the `layers` ctest target (`scripts/check_layers.py`) fails on an upward include, on an unqualified project include, or on a `src/**/*.cpp` missing from `CMakeLists.txt`, quoting the offending line |
| A prompt-injectable caller cannot enroll a biometric template. | MCP exposes four read-only tools; `tools/call enroll_person` answers `-32602 unknown tool` (`src/serve/mcp.cpp`) |
| Credentials on disk are useless to anyone who reads them. | tokens stored only as SHA-256, compared in constant time, issued from a 32-byte CSPRNG; binding to a non-loopback address with an empty token table **refuses to start** (`src/serve/auth.{hpp,cpp}`, `./main token`) |
| The audit trail never contains biometrics. | request audit lines carry `method`/`path`/`status`/`ms`/`caller`/`bytes` — never image bytes, embeddings or names — and a `401` is audited even under `--quiet`, with peer and whether a token was presented |
| Every tunable is declared exactly once. | the `knobs()` table (`src/app/config.cpp`) generates the file parser, `--set`, range validation, `--dump-config` and `config.example.yml`. A key it does not recognize is reported with a nearest-name suggestion — silently ignored in a config file, a hard stop from `--set` |

---

## Numbers

This machine: 16-core x86_64, CPU only, OpenCV 4.13.0, `-O3`. Every figure is reproducible with
the commands in this file; none of them is a benchmark I ran once and forgot.

| Pipeline cost (per frame / per face) | measured |
|---|---|
| detect (YuNet, 160×120) | **0.93 ms** |
| embed, per face | **4.47 ms** |
| letterbox onto 960×720 | 0.27 ms |

| HTTP service (one 1.3 MB JPEG, fresh connection per request) | measured |
|---|---|
| 4 workers | p50 **74 ms**, ≈135 req/s |
| 8 workers | p50 100 ms, ≈193 req/s — past here the CPU is the wall |
| identical requests, result cache on | 205 → **1347 req/s** |

| Recognition quality (LFW deepfunneled, through the production path, held-out photos) | measured |
|---|---|
| false accepts at the default `recognize.thresh = 0.45` | **1.3%** against 610 people, 3.0% against 1680 |
| the same at the old default `0.2` | **99.7%** against 610 people — the reason 0.45 is the default |
| 1:1 EER | threshold ≈ 0.18, rate ≈ 4.1% |

| Build & checks | measured |
|---|---|
| clean build, `-j16` | **23 s** |
| `ctest` (5 targets) | **0.7 s**, needs no model weights |
| recompile after editing one subcommand | 1.4 – 1.9 s |

---

## What this project deliberately does not do

- **No TLS and no reverse proxy inside the binary.** `serve` binds `127.0.0.1` by default; terminate TLS in front of it. A single-binary "HTTPS server" is how certificate handling quietly becomes someone's problem.
- **No CUDA / OpenVINO promise.** The tokens exist and the code path is there, but they need an OpenCV build with those plugins; the OpenCV this project is developed against has neither. CPU is the tested default.
- **No encryption at rest for templates.** `paths.gallery_index` is plaintext 512-d templates plus names, created `0600`; the mitigation is the file mode and the flag that turns the cache off, not a security claim. Stated in [Limitations](#limitations) rather than hidden.
- **No enrollment over a network transport.** `enroll` is a local command; the MCP server omits it on purpose, because writing a biometric template is not retractable.
- **No accuracy claim beyond what is measured here.** `data/` holds two identities — a regression signal, not a benchmark — and the augmentation harness measures 1:1 separability, so it is never used to pick a threshold that a 1:N shell applies.
- **No CI runner, no linter, no formatter config.** Verification is `make && ctest` plus the hand-run recipes written out in this file, with their numbers, so a reviewer can repeat them instead of trusting them.

---

## Documentation

The front of this file is the pitch and the evidence. Each deep dive lives in `docs/` and is
written to the same standard: every claim names the measurement behind it.

| | |
|---|---|
| [docs/interfaces.md](docs/interfaces.md) | `identify` · `compare` · `verify` · `demo` — flags, the exit-code contracts, why `verify` exits non-zero on "no" |
| [docs/serving.md](docs/serving.md) | `serve` — endpoints, request-level detector size, the SHA-256 result cache, one capacity budget and what shedding buys (measured), auth, audit, `/metrics`, `deploy/` |
| [docs/mcp.md](docs/mcp.md) | the four MCP tools and their schemas, and why enrollment is deliberately not one |
| [docs/models.md](docs/models.md) | the two pinned ONNX exports (sha256, upstream commit), what a model swap can silently change, the two YuNet heads measured against each other |
| [docs/configuration.md](docs/configuration.md) | precedence, the flat file format, how bad input is handled, `paths.root`, registration semantics, the gallery index, the full knobs table |
| [docs/evaluation.md](docs/evaluation.md) | the 1:1 vs 1:N arithmetic, two LFW runs over disjoint identities, `unreachable` vs measurement resolution, how the startup self-check estimates FAR |
| [docs/architecture.md](docs/architecture.md) | the layer chain and the namespace rule, the traps that cost real debugging time, how a change is validated (with the harness numbers) |
| [docs/limitations.md](docs/limitations.md) | the complete list, including the pinned detector export's unreliable last level and how `PriorBox::decode` contains it |

---

## Quick start

### Requirements

| | |
|---|---|
| OpenCV | 4.x, built with the `dnn` module (developed and tested against **4.13.0**) |
| Compiler | C++17 (g++ 15 used here; clang++ should work) |
| Build | CMake ≥ 3.10, Make |
| Hardware | Any x86_64 or Arm board. Runs on CPU with no accelerator. |

Verified build + run environment: **openSUSE Leap 16.0**, OpenCV 4.13.0 (system
`opencv4` devel package), g++ 15.3.0, CMake 3.31.7.

> The **CUDA / OpenVINO** tokens below require an OpenCV build that includes those
> plugins. A distro `opencv` package usually does not. Plain CPU is the default and
> always works.

---

### Build

```sh
git clone https://github.com/eiPI1-0/FaceRecognition
cd FaceRecognition && mkdir -p build && cd build
cmake .. && make          # produces ./main
```

---

**The weights and the enrollment set are not in this repository** (both gitignored), so a fresh
clone compiles and then loads nothing:

```sh
./scripts/get_models.sh                       # the pinned YuNet export, sha256-checked → build/models/
# put photos under build/data/<person>/ — one folder per person, any number of photos
cd build && ./main identify --image data/BruceLi/*.jpeg
```

A missing, zero-byte or Git-LFS-pointer weight is caught before the ONNX importer runs, with a
pasteable hint about the working directory. Pins, hashes and what a swap may change:
[docs/models.md](docs/models.md).

---

## The interfaces

Six commands, one binary — `demo` is the original interactive window, the rest are for scripts,
services and LLM tool calls.

```sh
./main identify --image <path> [options]                     # one image in, JSON out
./main enroll   --name <person> [--dry-run|--yes] <photo>…   # add a person, validated first
./main serve    [--host H] [--port P] [--workers N]          # HTTP service
./main mcp      [--transport stdio|http] [--port P]          # MCP tools for LLM clients
./main compare  <img-a> <img-b>                              # cosine between two images' faces
./main verify   <img> --name <person>                        # is this image this person?
./main evaluate <dataset> [--gallery-photos N] [--probe-photos N]
                                                           # FAR/FRR + EER at 1:1 and 1:N
./main demo     <source> [cpu|cuda|openvino] [batch|nobatch] # the interactive window
```

`./main <source>` without a command is still treated as `./main demo <source>`, so older
invocations keep working.

### identify — JSON in, JSON out

```sh
./main identify --image photo.jpg --top-k 3 --include landmarks,crops
```

```json
{"status":"ok","source":"photo.jpg","image_size":[640,607],"detector_input":[160,120],
 "timing_ms":{"detect":1.03,"embed":5.22,"total":6.25},
 "faces":[{"index":0,"bbox":[299.1,71.8,111.3,126.5],"det_score":1.0,"aligned":true,
           "matched":{"person_id":"BruceLi","score":1.0},
           "top_k":[{"person_id":"BruceLi","score":1.0},{"person_id":"IvanYang","score":0.0226}]}]}
```

stdout is **only** JSON (logs go to stderr), and exit status 0 also covers *"no face"*
(`"faces": []`) and *"face found but unknown"* (`"matched": null`) — those are answers, not
failures. `bbox` and `landmarks` come back in **original-image pixels**, not the detector's
160×120 space.

### serve — HTTP

```sh
./main serve --port 8080 --workers 8
curl -X POST --data-binary @photo.jpg -H 'Content-Type: image/jpeg' localhost:8080/v1/identify
```

| | |
|---|---|
| `POST /v1/identify` | raw image body or JSON with `top_k` / `min_score` / `include` / per-request `detector_input` |
| `POST /v1/embed` · `POST /v1/compare` | 512-d embedding · cosine between two images |
| `GET /v1/people` | enrolled ids — never embeddings or photos |
| `GET /healthz` · `/readyz` · `/metrics` | probes and Prometheus text; `/readyz` also reports the cache and the gallery's threshold estimate |

Two facts belong up here: **one `Pipeline` per worker is both the concurrency model and the
backpressure** (past that, one end-to-end capacity budget returns `503 + Retry-After`), and **a
non-loopback `--host` with no token table refuses to start** — binding `0.0.0.0` cannot be a silent
open door. Endpoints, cache, auth, audit, metrics, `deploy/`: [docs/serving.md](docs/serving.md).

### mcp — four tools an LLM can call

| Tool | Returns |
|---|---|
| `identify_faces` | per-face bbox, `matched` or `null`, ranked candidates |
| `compare_faces` | cosine of the most confident face in each image |
| `list_people` | enrolled person ids |
| `get_face_crop` | aligned 112×112 PNG |

**Enrollment is deliberately not a tool**: writing a biometric template is not retractable, so it
must not be reachable from a prompt-injectable caller (`tools/call enroll_person` → `-32602`).
The tool description the model actually reads states that `"matched": null` means *unknown*.
[docs/mcp.md](docs/mcp.md).

### enroll — local only, and validated before it writes

```sh
./main enroll --name BruceLi --dry-run a.jpg b.jpg   # validate, write nothing (the default)
./main enroll --name BruceLi --yes a.jpg b.jpg       # copy the usable photos into data/BruceLi/
```

Only photos containing a usable face are copied, an existing file is never clobbered (`_1` suffix),
and `--name` is restricted to one safe path component. **Enrollment has no network transport** —
neither `serve` nor `mcp` can do it — because writing a biometric template is the irreversible act.

### compare / verify / demo

```sh
./main compare a.jpg b.jpg      # {"cosine":0.022648,…} — bit-identical to /v1/compare and to MCP
./main verify x.jpg --name BruceLi && echo present || echo absent   # exit 1 = "no", 2 = "cannot answer"
./main demo 0                   # the interactive window on camera 0 (needs a display)
```

`verify` is an authentication-style decision, so it **exits non-zero when the answer is no** —
unlike `identify`, which exits 0 even with no face. The full exit-code contract, and why the
face each image contributes is the *most confident* detection rather than the largest:
[docs/interfaces.md](docs/interfaces.md).

---

## Configuration, in one screen

Precedence is **built-in defaults < `--config` file < command line**, and one table (`knobs()` in
`src/app/config.cpp`) declares every key — the file parser, `--set`, range validation,
`--dump-config` and `config.example.yml` are all generated from it, so they cannot drift.
`./main --dump-config` prints the whole table with defaults and comments.

| key | default | why you would touch it |
|---|---|---|
| `recognize.thresh` | `0.45` | the match threshold — a gallery-sized number, see the next section |
| `detector.input_w/h` | `160×120` | larger buys *recall* on small or distant faces, roughly quadratically in cost |
| `detector.conf_thresh` / `nms_thresh` | `0.6` / `0.3` | detection sensitivity and box suppression |
| `tracker.iou` / `dist_ratio` | `0.30` / `0.60` | the two association gates — IoU alone splits a face at the frame edge (measured adjacent-frame IoU 0.37) |
| `tracker.detect_every` / `max_lost` / `min_hits` | `1` / `5` / `2` | detection cadence and track lifecycle |
| `align.border` | `constant` | how the alignment window is filled for edge-clipped faces |
| `display.rfactor` | `6` | display canvas = detector input × 6 (the letterbox target) |
| `log.sink` / `log.level` | `stderr` / `warn` | audit lines ignore the level; only `sink: off` silences them |
| `paths.root` / `data_dir` / `gallery_index` | cwd / `./data/` / `./gallery.idx` | asset resolution and the persistent feature index |

Bad input is handled loudly: out-of-range or wrong-typed values stop the run naming `file:line`
and the allowed range; an unknown key in a file warns with the nearest valid key, and after `--set`
it is a hard stop. [docs/configuration.md](docs/configuration.md).

---

## What the threshold actually means

`identify` scores the query against **every** enrolled person and keeps the maximum, so the
false-accept rate is a property of the gallery, not of the pair: `1 − (1−p)^(N−1)`.

| `recognize.thresh` | against 610 people | against 1680 people |
|---|---|---|
| `0.20` — the previous default | **99.7%** of strangers matched *somebody* | 100% |
| **`0.45` — the default** | **1.3%** | **3.0%** |

Measured on LFW (deepfunneled) through the production path on **held-out** photos, reproducible with
`./main evaluate <dataset>`. `serve` and `mcp` recompute the estimate for *the gallery they loaded*
at startup and answer it at `/readyz` (`far_1n_estimate`, `far_1n_resolution`) — because a false
accept raises no error, returns no 503 and adds no latency spike: it looks exactly like success.
The arithmetic, both LFW runs and why a 1:1 number (EER, the augmentation harness) must never be
used to pick a threshold a 1:N shell applies: [docs/evaluation.md](docs/evaluation.md).

---

## Deployment

| | |
|---|---|
| `deploy/Dockerfile` | multi-stage: build stage carries `opencv-devel`, runtime carries the single `libopencv413` plus the binary; non-root, assets and index on volumes |
| `deploy/systemd/facerec@.service` | `systemctl start facerec@{1..4}` — four processes, one port, `--reuseport` lets the kernel distribute connections |
| `deploy/nginx/facerec.conf` | TLS termination; `/metrics` restricted to private ranges |

**Scale out, not up**: one process already uses every core (one pipeline per worker), so the axis
that matters is processes — and that is a one-flag change, no coordinator and no shared memory.
Verified here: two processes on one port, two `LISTEN` lines in `ss -lnt`, 60 requests split
21 / 39 in the access log. Caveats that bite on first deploy (probes are not logged on success,
`/metrics` is per process): [docs/serving.md](docs/serving.md).

---

## Limitations

- Detection runs at a fixed 160×120, so **very small or very distant faces** are missed; there is no
  pyramid or tiling pass (the per-request `detector_input` buys recall instead).
- Recognition is an `O(N)` scan, which stays cheap — the real ceiling at large N is **accuracy, not
  compute**: FAR=1e-3 needs ≈0.71, which already rejects 29% of genuine faces at 610 people. A gallery
  of thousands needs a stronger embedder and a two-stage search, not just an index.
- `data/` ships 2 identities with 1 photo each: a regression signal, **not** an accuracy benchmark.
- Landmarks are coarse (3–5 px fit residual, ~5 px box jitter): identity is robust to it, box edges
  are not perfectly still.
- There is no batch mode that writes annotated images; the only visual output is the `demo` window.
- The pinned fused YuNet export computes 6 of its 1076 anchors unreliably under OpenCV 4.13;
  `PriorBox::decode` filters those candidates out, so they never surface in an API.

The full list, with the measurements: [docs/limitations.md](docs/limitations.md).

---

## Repo layout

```
common  →  core  →  app  →  serve  →  cli          (src/main.cpp sits on top)
```

`src/` is the only source root; an include names the layer it reaches into
(`#include "core/engine.hpp"`) and may only point downward. Each layer is also a CMake target, so
**the build graph is the architecture**, and the fifth ctest target (`scripts/check_layers.py`)
fails the build on an upward include, an unqualified project include, or a `src/**/*.cpp` that is
not in `CMakeLists.txt`. Layers 0–3 are one `facerec` namespace; only `src/cli/` and the entry
point are outside it.

The layer table, the namespace rule, the traps (`cv::Size` order,
detector-space coordinates, `copyMakeBorder` on a submatrix, the cosine contract) and how a change
is validated: [docs/architecture.md](docs/architecture.md).

---

## Changes

See [CHANGELOG.md](CHANGELOG.md) for what this revision fixed, added and measured
(older revisions are in `git log`; releases are tagged, latest `v1.3.0`).

## License

Apache License 2.0 — see [LICENSE](LICENSE). Model weights are third-party works with
their own terms; check the upstream repositories before redistributing them.