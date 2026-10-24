# FaceRecognition — real-time face detection + recognition in C++

Real-time face **detection and recognition** built on the **OpenCV DNN module only** —
no dlib, no ncnn, no TensorFlow/PyTorch runtime. One dependency, one binary.

```
camera frame ──▶ YuNet detector ──▶ 5-point similarity alignment ──▶ MobileFaceNet
                   (160×120)              (112×112 crop)                (512-d embedding)
                                                  │
                              IoU tracker (IDs, smoothing, gap filling)
                                                  │
                                    cosine similarity vs. per-person fused feature
```

Everything — including prior-box decoding, NMS and face alignment — is implemented in
this repository, so the whole pipeline is inspectable and portable to any platform that
builds OpenCV.

---

## Requirements

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

## Build

```sh
git clone https://github.com/eiPI1-0/FaceRecognition
cd FaceRecognition && mkdir -p build && cd build
cmake .. && make          # produces ./main
```

---

## Assets you must supply (not in this repository)

`build/models/` and `build/data/` are **gitignored** — a fresh clone compiles but has
nothing to load, so `main` prints `Loaded 0 person(s)`. Place the assets yourself:

```
build/
├── models/
│   ├── yunet_120x160.onnx        # detector
│   └── mobilefacenet.onnx        # embedder
└── data/
    ├── <person_name>/*.jpg       # one folder per person, their photos
    └── ...
```

The detector can be fetched automatically:

```sh
./scripts/get_models.sh                    # → build/models/yunet_120x160.onnx, checksum-verified
./scripts/get_models.sh --with-yunet-320   # + the 320x320 per-level export (see "Swapping models")
```

**Detector** — YuNet, 160×120 input. Upstream restructured `ShiqiYu/libfacedetection.train`
and deleted the old `tasks/task1/onnx/` path, so a branch-name URL 404s; the file is pinned
to the commit that shipped it:

```
https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/1688402dbd9b9fc4a3a6793810f558b7407ff384/tasks/task1/onnx/yunet_120x160.onnx
sha256 50ef07f702a31741ca46a4c0d947773b64143b9362780237bf0d427d6c79bab7   345478 bytes
```

opencv_zoo's exports (`2023mar`, `2026may`) work as alternate detectors but are **not drop-in
defaults** — they are 320×320 with a per-level head, so they need the config keys listed under
"Swapping models"; `--with-yunet-320` fetches the first one, pinned and checksummed. A
community re-export of the 120×160 shape is a different matter: at least one fails to parse in
OpenCV 4.13's ONNX importer, so do not swap it in for the default.

Its contract, as consumed by `PriorBox::decode`:

| tensor | shape | notes |
|---|---|---|
| `input` | 1×120×160×3 | BGR, no mean subtraction, scale 1.0 |
| `loc`   | 1076×14 | 4 box deltas + **10 landmark values**, order `[right_eye, left_eye, nose, mouth_left, mouth_right]` |
| `conf`  | 1076×2  | background / face |
| `iou`   | 1076×1  | objectness |

Final score is `sqrt(conf_score · iou)`. Landmarks are **person-side** named
(`right_eye` is the person's right eye, i.e. on the left of the image).

**Embedder** — MobileFaceNet, 112×112 input, **512-d** output:
`input` 1×3×112×112, RGB order, normalized as `(x/255 − 0.5) / 0.5`, output `1×512`.
This repository does not redistribute the weights; supply any MobileFaceNet ONNX
matching that contract. It must be a model trained with **ArcFace-style 5-point
alignment** — the aligner here targets the standard ArcFace 112×112 reference template,
so a differently-aligned model will silently degrade.

`build/models/openface.nn4.small2.v1.t7` is a leftover from an earlier revision and is
**unused** by `main.cpp`.

---

## Run

```sh
./main <source> [cpu|cuda|openvino] [batch|nobatch]
             [--config <path>] [--set key=value] [--dump-config]
```

`<source>` is either a **camera index** (any integer) or anything `cv::VideoCapture`
accepts — an image file, a video file, a camera device path, a URL/stream. A still image
is processed as its single frame; for non-camera sources the window stays open on the last
frame until you press `Esc`, instead of exiting immediately.

```sh
./main 0                          # camera 0, CPU backend
./main 1 cuda                     # CUDA backend (needs OpenCV built with CUDA)
./main 0 openvino                 # Inference Engine backend
./main 0 nobatch                  # force per-face forward passes
./main photo.jpg                  # single image
./main clip.mp4                   # video file
./main "rtsp://host/stream"       # stream
./main 0 --config config.yml      # load a config file (see Configuration)
./main 0 --set recognize.thresh=0.35 --set tracker.detect_every=3
./main --dump-config              # print the effective config and exit
```

A display is required (`cv::imshow`); on a headless machine use `xvfb-run` or an `Xvfb`
display. Model/data paths are relative to the working directory, so by default run the
binary from `build/`; set `paths.root` (in a config file or with `--set`) to anchor them
elsewhere (see [Configuration](#configuration)).

Headless sanity check (no camera attached):

```sh
./main 1
# Loaded 2 person(s) from ./data/
# ...
# Cannot open source: 1        (exit status != 0)
```

`Loaded N person(s)` confirms the models and the registration set loaded. Everything
OpenCV prints between those two lines (FFmpeg "Camera index out of range", GStreamer
video-position warnings) is backend noise from probing devices, not a bug in this
program.

---

## Configuration

Every tunable is one entry in a single table (`knobs()` in `config.cpp`) with a dotted key.
The **same key** is used in the config file, in `--set`, and in `--dump-config` output —
there is one list, generated from that table, so it cannot drift out of sync.

Precedence: **built-in defaults < `--config` file < command line** (a later flag wins).

```sh
./main --dump-config > config.example.yml    # the shipped example is generated, never hand-written
./main 0 --config config.example.yml
./main 0 --set recognize.thresh=0.35 --set tracker.detect_every=3
./main 0 cuda batch                          # shorthand for --set dnn.backend=cuda --set dnn.batch_mode=1
```

The file format is flat `key: value` lines with `#` comments — no YAML parser, no nesting:

```yaml
paths.data_dir: ./data/        # one subfolder per enrolled person
recognize.thresh: 0.20
detector.conf_thresh: 0.60
tracker.detect_every: 1
```

Run `./main --dump-config` for the complete list of keys, their defaults and what each one
does; it is generated from the table, so it is always current and self-documenting.

How bad input is handled, deliberately:

| input | result |
|---|---|
| unknown key in a file | warning naming the nearest valid key (`recogn_thresh` → “did you mean `recognize.thresh`”), then ignored |
| wrong type / out-of-range value | hard stop naming `file:line`, the key and the allowed range |
| unknown key after `--set` | hard stop — you typed it, you should hear about it |
| missing data dir | warning (`no one will be recognized`) |
| bad backend or missing model file | readable `Fatal: …`, non-zero exit — not a core dump |

**Paths.** Relative `paths.*` values are resolved against `paths.root`, which defaults to
the **working directory** — so a config file that only changes thresholds never moves your
assets. Anchoring is explicit; say so in the file:

```yaml
paths.root: .            # relative to this file's directory
paths.data_dir: ./data/
```

With that line `./build/main 0 --config build/config.yml` works from the repo root, and
`--set paths.root=<dir>` does the same from the command line (relative to your shell's cwd).
Absolute values are used as given.

### Swapping models

Configuration covers the **shape** axis completely; it cannot cover the **graph contract**
axis. Measured behaviour:

Works by editing the yml only:

- **any input size** for a YuNet-160×120-style graph — verified at 320×320, 224×224 and
  128×96. The graph is shape-dynamic, and the prior grid (strides 8/16/32/64 over
  floor-halved feature maps) equals the graph's own anchor count at every size. Set
  `detector.input_w/h`, and `display.rfactor` so the canvas stays a sane size.
- **the tensor names** of a fused head — `detector.out_loc` / `out_conf` / `out_iou`.
- **the other YuNet head layout** — `face_detection_yunet_2023mar.onnx` (320×320, outputs split
  per level as `cls_8 … kps_32`) is decoded too: the head type is inferred from the model's own
  layer names, so the only keys to change are `paths.detector_model`, `detector.input_w/h` (a
  multiple of the largest stride, 32) and `display.rfactor` (3 gives a 960×960 canvas). No key
  says which head it is — an unrecognized head is an error listing the names the model exposes.
  Fetch it with `./scripts/get_models.sh --with-yunet-320`. Because the strides come from the
  model rather than a hardcoded grid, other per-level exports run unchanged: zoo's newer
  `face_detection_yunet_2026may.onnx` was tried with no code or config change beyond the model
  path, and detects and self-matches (1.000) at both 320×320 and 160×160 — it is not fetched by
  the script and has not been put through the numbers below.
- **any ArcFace-convention embedder, at its native shape** — `paths.embedder_model` plus
  `embedder.input_w/h`. The aligned crop is warped to exactly that size, so the model must
  have been trained on that crop. Feature length is irrelevant (fusion and cosine are
  dimension-agnostic): verified by dropping in opencv_zoo SFace, whose output is 128-d
  instead of 512-d — self-match still 2/2.

Two head layouts are understood: **fused** (one `loc` of 14 floats per anchor — 4 box + 5 × 2
landmarks — plus `conf` (2 per anchor) and `iou` (1 per anchor), 4 levels at strides
8/16/32/64) and **per-level** (`cls_{s}/obj_{s}/bbox_{s}/kps_{s}`, one anchor per cell, strides
taken from the names). Anything else is an error at startup.

Don't expect a free accuracy win from the bigger export — measured on the same scene, with the
face placed at equal size **in detector space** (not equal size in the scene):

| face size in detector pixels | pinned YuNet 120×160 (fused) | zoo 2023mar 320×320 (per-level) |
|---|---|---|
| ~76 px | — | detected, score 0.84 |
| ~38 px | detected, score 0.51 | not detected |
| ~19 px | score 0.20 | not detected |

The per-level export's advantage is that a 320×320 input gives a 4:3 scene ~2.7× more detector
pixels per face, so on the *same* scene it detects faces the small export misses; at equal
detector pixels this export is the weaker one below ~50 px. On the augmentation harness (26
samples, seed `12345`) the two are a wash: within 0.902 / between −0.039 / margin 0.941 for
fused@160×120 vs 0.901 / −0.034 / 0.935 for per-level@320×320, zero detection misses either way.

What stays fixed regardless of head:

- **normalization is fixed per stage**: detector scale 1.0, no mean, BGR; embedder `1/255`
  then `(x − 0.5) / 0.5` on RGB — which is what InsightFace-family exports expect
  (mean/std 127.5).
- **alignment is the ArcFace 5-point template** with a square ROI of 3.2 × eye distance. An
  embedder trained on a different crop (MTCNN/RetinaFace) runs without any error and scores
  badly — the one failure mode that stays silent, so check your scores after a swap.

Both decoders check their layout contract instead of trusting it: when the tensors do not match
the grid implied by `detector.input_w/h` they stop and name the sizes, which turns a wrong
model/shape pair into a message rather than an out-of-bounds read.

---

## Registration

`build/data/<name>/` containing that person's photos. At startup every photo is
detected, aligned and embedded; the per-person feature is the mean of the
**L2-normalized** embeddings, re-normalized. Photos with no detectable face are skipped.

More photos per person ⇒ more stable fused feature. Accuracy is dominated by how close
the recognition environment (distance, lighting, camera) is to the registration photos.
The gallery is rebuilt from the photos on every startup — there is no cached feature file,
so changes to the preprocessing convention need no migration step.

---

## Tuning

Tunables are config keys (defaults in `config.hpp`, full list via `--dump-config`); the
ones worth knowing:

| key | default | effect |
|---|---|---|
| `recognize.thresh` | `0.2` | cosine threshold above which a match is reported. After the alignment fix, same-identity scores sit near 1.0 and cross-identity near 0, so there is wide headroom. |
| `display.rfactor` | `6` | display frame = `detector input × rfactor` = 960×720; the letterbox canvas and the coord scale for `draw()`/`putText()`. |
| `detector.conf_thresh` / `detector.nms_thresh` / `detector.keep_top_k` | `0.6` / `0.3` / `750` | detection sensitivity and box suppression. |
| `tracker.iou` / `tracker.detect_every` / `tracker.max_lost` / `tracker.min_hits` | `0.45` / `1` / `15` / `1` | see below. |

`detect_every` is the main performance/accuracy dial. Measured here (CPU, 960×720,
one face):

| `detect_every` | max box-centre error |
|---|---|
| 1 (default) | ~6 px |
| 3 | ~9 px |
| 5 | ~18 px |

Detection is cheap relative to embedding, so raising it buys almost nothing:

```
YuNet detect         0.93 ms/frame
MobileFaceNet embed  4.47 ms/face     ← this dominates
```

---

## Performance (this machine, CPU, single face)

```
detect                    0.93 ms
embed per face            4.47 ms
letterbox onto 960×720    0.27 ms   (a plain stretch-resize was 0.21 ms)
```

Batch embedding is **auto-disabled on CPU targets**: OpenCV's CPU backend executes the
graph per sample and rebuilds internal blobs whenever the batch size changes, so
batching measured *slower* (1.31× slower over a realistic varying-face-count sequence,
−22% at N=8). GPU/NPU targets get batching automatically. Results are bit-identical
either way (verified N=1…16). Override with `batch` / `nobatch` to A/B it yourself.

---

## Source layout

| File | Role |
|---|---|
| `main.cpp` | CLI parsing, config application, capture loop |
| `config.hpp` / `config.cpp` | `PipelineConfig` + the `knobs()` table — every tunable described once, driving file parsing, `--set`, validation and `--dump-config` |
| `FaceRecognition.h` / `.cpp` | `DNNConfig`, `Detector`, `FeatureExtractor`, `Similarity`, header-only `FeatureDatabase<T>`; alignment (`align_face`, `align_detection`), `extract_face_feature(s)`, `list_person_dirs` |
| `face_tracker.hpp` / `.cpp` | `FaceTracker` — IoU greedy matching + constant-velocity prediction |
| `priorbox.hpp` / `.cpp` | YuNet prior-box generation and raw-output decoding |
| `detector_head.hpp` / `.cpp` | head strategy: fused (via `PriorBox`) vs per-level YuNet-G, inferred from the model's layer names |
| `utils.hpp` / `.cpp` | `nms()`, `draw()`, `letterbox()` |
| `face_def.hpp` | POD types: `BndBox_xyxy` / `BndBox_xywh`, `Landmarks_5`, `Face` |

### Contributor notes (the traps in this codebase)

- **`cv::Size` is `{width, height}`.** `img_shape = {160, 120}` means 160 wide, 120 tall.
- **Detection coordinates are always in detector-input space (160×120)**, never in the
  image you passed to `detect()`. `align_detection(img, det, detection_shape)` maps them
  for you; pass the *same image* you detected on.
- **Every input frame is letterboxed onto `display_shape`** (`letterbox()` in `utils.cpp`)
  before tracking and drawing, and registration goes through the same helper — the two
  paths must preprocess identically or the features are not comparable. The padding is
  centred and the canvas is exactly `input_shape × rfactor`, so `draw()`/`putText()`
  scaling detection coords by `rfactor` stays exact with no extra offset.
- **`CosSimilarity::similarity` returns `‖ref‖ · cos(query, ref)`**, not a cosine — it
  divides only by `norm(feature)` and assumes the *stored* side is unit-length (the
  fusion normalizes it). Comparing two raw embeddings returns ≈15, not a value in [0,1].
- **Tracker velocity is per frame**, `v = (det_center − last_det_center) / frames_since_update`,
  with `last_det_center` anchored at the last *matched detection*. Assigning the whole
  inter-detection displacement, or advancing the anchor by prediction, both produce boxes
  that leap ahead of the face and then lose association.
- **Alignment ROI must be square and ≈3.2 × eye distance**, centred on the landmark
  centroid. The ArcFace 112×112 window spans much more than a tight detector bbox.
- **The aligned crop size comes from `FeatureExtractor::inputSize()`**, not from a literal:
  `embedder.input_w/h` is a geometry knob (the warp produces exactly the size the model
  wants — one interpolation, not two). Changing it only makes sense together with a model
  trained on that crop.
- **`PriorBox::decode` asserts the head contract** (`loc.total() == priors*14`,
  `conf.total() == priors*2`, `iou.total() == priors`). Before that check the decode loop
  indexed past the end of the tensors whenever the model's anchor count differed from the
  prior grid, and a type mismatch returned an empty face list that looked like "no faces".
- **The decode scheme is inferred, never configured.** `make_detector_head()` looks at
  `net.getLayerNames()`: the configured `detector.out_*` names → fused; a complete
  `cls_{s}/obj_{s}/bbox_{s}/kps_{s}` family → per-level with strides from the names. Adding a
  third layout means adding a class, not a key. The per-level grid is `input / stride`, so input
  dims must be divisible by the largest stride (OpenCV's own implementation pads instead).
- **The prior grid agreeing with the graph is an empirical fact**, not a theorem: strides
  8/16/32/64 over floor-halved feature maps matched this YuNet export at 160×120, 320×320,
  224², 200², 150², 128² and 640². A different export can use the same strides and still
  split its head per level — then only the decoder is wrong, not the config.
- **`knobs()` in `config.cpp` is the only list of tunables.** File parsing, `--set`,
  range validation and `--dump-config` are all generated from it, so adding a tunable
  means adding a struct field *and* a table entry — never a bespoke parse branch.
  `config.hpp` is app-level: it must not be pulled into `FaceRecognition.h`, because
  `FeatureDatabase<T>` is a header-only template and every consumer would recompile.
- **Relative `paths.*` values are resolved against `paths.root`** (default: the config
  file's directory, else the working directory), in one place, after defaults, file and
  `--set` have all been applied. Resolving earlier — or per source — produces the
  half-resolved mix that made this project unrunnable outside `build/`.

### How changes are validated

There is no test harness, so verification is: `make`, then `./main 1` (expect
`Loaded N person(s)` followed by `Cannot open source: 1` and a non-zero exit), then a throwaway program in `build/` linked against
`../FaceRecognition.cpp ../priorbox.cpp ../utils.cpp` that registers from `./data/` and
re-recognizes each registration photo (expect that photo's own person as top match); an
`hconcat` of two people's photos exercises multi-face index mapping. For any
alignment/embedding change, embed ~12 warped variants of each identity (±25° rotation,
0.85–1.15 scale, small translations, brightness ±40) and compare mean **within-identity**
vs **between-identity** true cosine (`a.dot(b)/(‖a‖‖b‖)`, *not* `CosSimilarity`).
Current baseline (letterbox pipeline, warp seed `12345`, 13 samples per identity):
**within 0.845 / between −0.029 / margin 0.874**; the same harness run against the old
stretch preprocessing scored 0.805 / −0.019 / 0.824. These numbers are harness-specific —
compare only against a re-run of the same generator (an earlier, different augmentation
draw produced 0.848 / −0.068 / 0.916, which is not comparable).

---

## Limitations

- Frames from any source are letterboxed onto a fixed 960×720 canvas, so **aspect ratio is
  preserved**; non-4:3 sources get black bars in the window. The price of that simplicity:
  the detector still runs on the whole padded frame at 160×120, so on an extreme aspect
  (portrait phone frame) the face occupies fewer detector pixels than a crop-aware
  pipeline would give it.
- Requires a display: the only output is an `imshow` window (no headless/batch mode that
  writes annotated files).
- YuNet's landmarks are coarse: fit residuals are 3–5 px and detection boxes jitter by
  ~5 px frame to frame. Identity assignment is robust to this; box edges are not perfectly
  still.
- Detection runs at a fixed 160×120, so **very small or very distant faces** are missed;
  there is no image pyramid / tiling pass.
- Recognition scales linearly with the enrolled population (`O(N)` scan per face). Fine
  for tens of people; hundreds would need an index.
- The shipped sample data contains 2 identities with 1 photo each — useful as a regression
  signal, **not** an accuracy benchmark.
- No tests, CI, or lint config.

---

## Changes

See [CHANGELOG.md](CHANGELOG.md) for what this revision fixed, added and measured
(older revisions are in `git log`; releases are tagged, latest `v1.0.0`).

## License

Apache License 2.0 — see [LICENSE](LICENSE). Model weights are third-party works with
their own terms; check the upstream repositories before redistributing them.
