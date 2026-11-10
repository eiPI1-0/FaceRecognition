# Model assets and model swaps

What the two pinned ONNX files are, where they come from, and what a swap can silently change.

Deep dive split out of [README.md](../README.md), which keeps the short version.
Related: [configuration.md](configuration.md) (`detector.input_w/h`, `paths.*`), [limitations.md](limitations.md) (the pinned export's defect)

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

A model path that points at nothing, at an empty file (what a failed `curl -o` leaves behind),
or at a Git LFS pointer is reported by the knob that names it — `Invalid config:
paths.detector_model: model file … is a Git LFS pointer (131 bytes), not weights` — instead of
surfacing as an ONNX importer stack.

**Detector** — YuNet, 160×120 input. Upstream restructured `ShiqiYu/libfacedetection.train`
and deleted the old `tasks/task1/onnx/` path, so a branch-name URL 404s; the file is pinned
to the commit that shipped it:

```
https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/1688402dbd9b9fc4a3a6793810f558b7407ff384/tasks/task1/onnx/yunet_120x160.onnx
sha256 50ef07f702a31741ca46a4c0d947773b64143b9362780237bf0d427d6c79bab7   345478 bytes
```

opencv_zoo's exports (`2023mar`, and `2026may` — the same weights re-exported with dynamic
input dims, for OpenCV 5's ONNX Runtime engine) work as alternate detectors but are **not drop-in
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

---

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
  the script and has not been put through the numbers below. OpenCV 4's DNN is shape-agnostic,
  so either of those runs at **any** input size divisible by 32 (`detector.input_w/h`); the model
  was trained at 320×320, so quality away from that size is unmeasured, not guaranteed.
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
- **a face at the frame edge keeps its geometry.** The similarity transform comes from the
  landmarks, so clipping the ROI never changes the warp — the only question is what to put
  where the window sticks out of the frame. `align.border` chooses that fill; measured over
  312 edge-clipped cases (mean cosine to the unclipped reference embedding):
  `constant` 0.507, `reflect` 0.509, `replicate` 0.489 — copying the edge pixels inward is
  measurably *worse* than black, so the default stayed black. The other idea worth closing:
  a second embedding from a horizontally mirrored crop ("TTA") scored +0.006 mean cosine and
  rescued 1 case in 312 — implemented, measured, removed. Mirroring moves the hole instead of
  filling it; the model+aligner are already mirror-invariant at 0.97/0.98 cosine.

Both decoders check their layout contract instead of trusting it: when the tensors do not match
the grid implied by `detector.input_w/h` they stop and name the sizes, which turns a wrong
model/shape pair into a message rather than an out-of-bounds read.

---
