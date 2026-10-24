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
./scripts/get_models.sh          # → build/models/yunet_120x160.onnx, checksum-verified
```

**Detector** — YuNet, 160×120 input. Upstream restructured `ShiqiYu/libfacedetection.train`
and deleted the old `tasks/task1/onnx/` path, so a branch-name URL 404s; the file is pinned
to the commit that shipped it:

```
https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/1688402dbd9b9fc4a3a6793810f558b7407ff384/tasks/task1/onnx/yunet_120x160.onnx
sha256 50ef07f702a31741ca46a4c0d947773b64143b9362780237bf0d427d6c79bab7   345478 bytes
```

Other YuNet exports exist (opencv_zoo `2023mar`, and community re-exports of the 120×160
shape). They are **not drop-in here**: the zoo model is 320×320 with a different prior-box
layout, and at least one 120×160 re-export fails to parse in OpenCV 4.13's ONNX importer.

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
```

All model/data paths are **relative to the working directory**, so run the binary from
`build/`. A display is required (`cv::imshow`); on a headless machine use `xvfb-run` or an
`Xvfb` display.

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

## Registration

`build/data/<name>/` containing that person's photos. At startup every photo is
detected, aligned and embedded; the per-person feature is the mean of the
**L2-normalized** embeddings, re-normalized. Photos with no detectable face are skipped.

More photos per person ⇒ more stable fused feature. Accuracy is dominated by how close
the recognition environment (distance, lighting, camera) is to the registration photos.

---

## Tuning

Tunables live in `main.cpp`, plus tracker constructor arguments:

| knob | default | effect |
|---|---|---|
| `recogn_thresh` | `0.2` | cosine threshold above which a match is reported. After the alignment fix, same-identity scores sit near 1.0 and cross-identity near 0, so there is wide headroom. |
| `rfactor` | `6` | display frame = `input_shape × rfactor` = 960×720. Also the scale the camera frame is resized to. |
| detector `conf_thresh` / `nms_thresh` / `keep_top_k` | `0.6` / `0.3` / `750` | detection sensitivity and box suppression. |
| `FaceTracker(iou, detect_every, max_lost, min_hits)` | `0.45, 1, 15, 1` | see below. |

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
detect              0.93 ms
embed per face      4.47 ms
resize 960×720      0.05 ms
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
| `main.cpp` | CLI parsing, tunables, camera loop |
| `FaceRecognition.h` / `.cpp` | `DNNConfig`, `Detector`, `FeatureExtractor`, `Similarity`, header-only `FeatureDatabase<T>`; alignment (`align_face`, `align_detection`), `extract_face_feature(s)`, `list_person_dirs` |
| `face_tracker.hpp` / `.cpp` | `FaceTracker` — IoU greedy matching + constant-velocity prediction |
| `priorbox.hpp` / `.cpp` | YuNet prior-box generation and raw-output decoding |
| `utils.hpp` / `.cpp` | `nms()`, `draw()` |
| `face_def.hpp` | POD types: `BndBox_xyxy` / `BndBox_xywh`, `Landmarks_5`, `Face` |

### Contributor notes (the traps in this codebase)

- **`cv::Size` is `{width, height}`.** `img_shape = {160, 120}` means 160 wide, 120 tall.
- **Detection coordinates are always in detector-input space (160×120)**, never in the
  image you passed to `detect()`. `align_detection(img, det, detection_shape)` maps them
  for you; pass the *same image* you detected on.
- **`main.cpp` must resize the camera frame to `display_shape`** before tracking and
  drawing: `draw()`/`putText()` scale detection coords by `rfactor`, which is only valid
  when the frame is exactly `input_shape × rfactor`. The resize also keeps inference
  preprocessing identical to registration, which matters because `blobFromImage` does
  **not** preserve aspect ratio.
- **`CosSimilarity::similarity` returns `‖ref‖ · cos(query, ref)`**, not a cosine — it
  divides only by `norm(feature)` and assumes the *stored* side is unit-length (the
  fusion normalizes it). Comparing two raw embeddings returns ≈15, not a value in [0,1].
- **Tracker velocity is per frame**, `v = (det_center − last_det_center) / frames_since_update`,
  with `last_det_center` anchored at the last *matched detection*. Assigning the whole
  inter-detection displacement, or advancing the anchor by prediction, both produce boxes
  that leap ahead of the face and then lose association.
- **Alignment ROI must be square and ≈3.2 × eye distance**, centred on the landmark
  centroid. The ArcFace 112×112 window spans much more than a tight detector bbox.

### How changes are validated

There is no test harness, so verification is: `make`, then `./main 99` (expect
`Loaded N person(s)`), then a throwaway program in `build/` linked against
`../FaceRecognition.cpp ../priorbox.cpp ../utils.cpp` that registers from `./data/` and
re-recognizes each registration photo (expect that photo's own person as top match); an
`hconcat` of two people's photos exercises multi-face index mapping. For any
alignment/embedding change, embed ~12 warped variants of each identity (±25° rotation,
0.85–1.15 scale, small translations, brightness ±40) and compare mean **within-identity**
vs **between-identity** true cosine (`a.dot(b)/(‖a‖‖b‖)`, *not* `CosSimilarity`).
Current baseline to beat: **within 0.848 / between −0.068 / margin 0.916**.

---

## Limitations

- Frames from **every** source are resized to a fixed 960×720 **without preserving aspect
  ratio**; a portrait photo is stretched, which costs detection quality. Letterboxing to a
  4:3 canvas is the fix and is not implemented.
- Requires a display: the only output is an `imshow` window (no headless/batch mode that
  writes annotated files).
- `blobFromImage` resizes without preserving aspect ratio; a portrait photo squashed into
  160×120 detects noticeably worse than a 4:3 face. Letterboxing would fix this.
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
(older revisions are in `git log`; nothing is tagged yet).

## License

Apache License 2.0 — see [LICENSE](LICENSE). Model weights are third-party works with
their own terms; check the upstream repositories before redistributing them.
