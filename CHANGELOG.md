# Changelog

Notable changes to this repository. The format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses release tags rather
than a strict versioning scheme.

## v1.1.0 — 2026-10-27

### Added

- **A configuration layer.** 15 tunables lived as loose locals in `main.cpp`, plus 6 bare
  literals inside constructor calls, so nothing represented "this pipeline's settings":
  nothing to pass, nothing to validate in one place, no A/B without recompiling. They are
  now fields of `PipelineConfig`, each described **once** in `knobs()` (`config.cpp`) with
  its dotted key, member pointer, range and documentation. The config-file parser, the
  `--set` overrides, range validation and `--dump-config` are all generated from that one
  table, so there is no second key list that could drift.
- `--config <path>`: flat `key: value` file with `#` comments. An unknown key warns and
  names the nearest valid key (`recogn_thresh` → "did you mean `recognize.thresh`"); a
  wrong type or out-of-range value stops with `file:line`, the key and the allowed range.
- `--set key=value`: repeatable, one-off override (the tuning loop this project actually
  runs). An unknown key there is a hard stop rather than a warning.
- `--dump-config`: prints the effective configuration after file and CLI inputs.
  `config.example.yml` is its output and is committed, so the shipped example can never
  disagree with the built-in defaults. Dumping then loading the same file changes nothing.
- `paths.root`: all relative `paths.*` values resolve against it, defaulting to the working
  directory — deliberately *not* the config file's directory, so a file that only tweaks
  thresholds cannot silently move your assets. Anchoring is opt-in: `paths.root: .` inside
  the file (that value is file-relative) makes `./build/main 0 --config build/config.yml`
  work from anywhere, retiring the least discoverable constraint this project had ("you must
  run from `build/`").
- Loud failure where there used be core dumps: an unavailable backend
  (`--set dnn.backend=openvino` on an OpenCV without that plugin) and a missing model file
  now print `Fatal: …` and exit non-zero; a missing data directory warns instead of
  loading 0 persons in silence.
- `detector.out_loc` / `out_conf` / `out_iou`: the fused head's output tensor names are config
  keys, so a differently-*named* fused YuNet head is a config change.
- **A second detector head: per-level YuNet-G** (`cls_{s}/obj_{s}/bbox_{s}/kps_{s}`, one anchor
  per cell) — i.e. opencv_zoo `face_detection_yunet_2023mar.onnx` now works from the config file
  alone (`paths.detector_model`, `detector.input_w/h=320`, `display.rfactor=3`). Which head a
  model uses is **inferred from its own layer names**, not configured: `make_detector_head()`
  picks fused if the configured names are present, per-level if a complete `cls/obj/bbox/kps`
  family is, and errors listing the model's outputs otherwise. The decode formulas follow
  OpenCV 4.13 `modules/objdetect/src/face_detect.cpp` (`FaceDetectorYNImpl::postProcess`).
  Measured with the same augmentation harness (seed `12345`, 26 samples, zero detection misses
  both): fused@160×120 within 0.902 / between −0.039 / margin 0.941; per-level@320×320
  0.901 / −0.034 / 0.935 — the bigger input is not a free win. On a small-face ladder at equal
  *detector-space* size, the pinned 120×160 export wins (score 0.51 at 38 px vs nothing), while
  at equal *scene* size the 320 export detects faces the small one misses (0.84 vs nothing);
  that is the tradeoff, and it is documented in the README. This harness is a re-implementation
  and is only comparable to itself, not to the older 0.845 figure.
- `scripts/get_models.sh --with-yunet-320`: fetches that 320×320 export too, pinned to the
  opencv_zoo commit that shipped it and sha256-checked like the default model (a mismatch now
  deletes the partial file and exits non-zero). Re-pinning note: zoo's files are Git-LFS
  objects, so `raw.githubusercontent.com/<repo>/<commit>/<path>` returns the 131-byte pointer —
  `github.com/<repo>/raw/<commit>/<path>` serves the content.
- Model files are sniffed before the network is built: a missing path, a 0-byte file (a failed
  `curl -o`) or a 131-byte Git LFS pointer each used to arrive as an ONNX importer assertion.
  They now stop as `Invalid config: paths.<key>: model file <path> is a Git LFS pointer …` —
  named by the knob that points at them, and deliberately *not* checked by `--dump-config`,
  which should work on a clone with no weights yet.
- **Both head decoders check their layout contract.** `PriorBox::decode` used to return an
  empty face list when the outputs had the wrong type/layout — indistinguishable from "no
  faces in this frame" — and when the model's anchor count differed from the prior grid the
  decode loop read past the end of the tensors (UB). It now names both counts and stops
  (`prior grid has 5875 anchors … the model returned loc=457 conf=800 iou=1600 …`).
- **Zero new dependencies.** The loader is ~35 lines of flat parsing rather than
  `cv::FileStorage`, which refuses any file that does not begin with a `%YAML:1.0`
  signature (it sniffs content and ignores the `.yml` extension) and throws when it
  disagrees — and this way file and `--set` share one parse-and-validate path.

- **`tracker.dist_ratio`, the second association gate.** A detection whose centre moved less
  than this times the mean box size counts as the same track; the budget also widens with
  `0.5 × speed × frames-since-last-detection`, because the error of a predicted position grows
  with how long it has been since a real measurement.
- **Tracks carry their own lifecycle: `TrackView {face, id, measured}`.** `measured` says
  whether this frame's geometry came from a detection or is pure extrapolation, so the app can
  recognize one and not the other (see Fixed). `draw()` paints predicted boxes in a different
  colour and thinner, so a carried box is visible as such.

- **Edge-clipped faces have a fill policy: `align.border`.** When a face sits against the
  frame edge, the square alignment window (`3.2 × eye distance`) sticks out of the image. The
  similarity transform comes from the landmarks, so clipping never changed the geometry — all
  it decided was what the missing pixels were: black. That choice is now a key
  (`constant` | `reflect` | `replicate`), and it is measured rather than assumed: over 312
  edge-clipped cases (2 identities × 13 augmented frames × 4 edges × 25/50/75 % of the window
  cut away; metric = mean cosine to the unclipped reference embedding) `constant` 0.507,
  `reflect` 0.509, `replicate` 0.489. So the default **stays** `constant`, and the default
  default path is identical to the previous implementation: registration plus query through
  the whole app path on three images (one of them edge-clipped) matched to the bit, and
  aligned crops of heavily clipped faces differ by at most ±1 LSB (6 crops checked).
  The competing idea was implemented, measured and deleted rather than kept as a knob: a
  second embedding from a horizontally mirrored crop, gated on edge-clipped faces, beat the
  direct embedding by **+0.006** mean cosine and rescued **1 case in 312** — mirroring moves
  the hole instead of filling it, and the model+aligner are already mirror-invariant
  (self vs self-flip cosine 0.97/0.98).

### Fixed

- **One face could become two boxes that never went away.** A face near the frame edge
  alternates between the whole-face box and the visible-part-only box; the measured IoU between
  those two on adjacent frames is **0.37**, below the tracker's single 0.45 association gate.
  So the next detection was declared a *new* track while the old one kept being predicted —
  both drawn every frame, each with its own label, and the pair was permanent. The real defect
  was the single gate: box IoU conflates "same object" with "same box geometry", and those
  separate exactly at the frame edge, which is where this was observed. Association now scores
  two gates (IoU **or** normalized centre displacement, `tracker.dist_ratio`) over all candidate
  pairs taken in descending score order — detections no longer claim tracks in arrival order,
  which used to make the result depend on the detector's output permutation — and a dedup pass
  (same two quantities) makes "at most one track per face" a hard property rather than luck.
- **Prediction distorted boxes.** `advance()` clamped `x` into `[0, frame_width − box_width]`;
  once a box is wider than the space left, that bound goes negative and the box gets pinned to
  x = 0 *and squeezed*, which made the next association harder and spawned yet another duplicate
  track. Prediction is now a rigid translation of box and landmarks together, and a track that
  leaves the frame entirely is dropped instead of being pinned to the border.
- **Extrapolated geometry was fed to the embedder.** A carried track holds the last *measured*
  landmarks offset by velocity: good enough to draw, meaningless to embed — and the embedding
  comes back looking perfectly normal, which is how a phantom box gets somebody's name on it.
  The app now embeds only `measured` faces and carries the last real label on the track's `id`
  for at most `tracker.max_lost` frames; a measured frame's verdict always replaces the cache,
  *including* "nobody matches", so names no longer linger after the person walks off.
- **One false positive stayed on screen for 16 frames** (`max_lost 15`, `min_hits 1`) — and was
  embedded 16 times. Defaults are `max_lost 5` and `min_hits 2`: a detection that exists for a
  single frame is never shown at all. `min_hits` cannot corroborate the first `min_hits` frames
  of a stream, so those are emitted as measured regardless — otherwise a single still-image
  source would print nothing.

  Measured on one synthetic harness with both versions linked against it (one face unless
  noted): face at the edge with alternating box geometry, 40 frames — **1.98 → 1.00**
  boxes/frame, doubled frames **39/40 → 0**; two faces 1.5 face-widths apart — **2.63 → 2.00**
  boxes/frame, peak **3 → 2**; target vanishes — ghost drawn **14 → 5** frames, embeddings
  **14 → 0**; false positive present for one frame — drawn **11 more frames → never**; motion
  through a 3-frame detection miss — boxes far from the truth **5 → 0**; box-centre error at
  7 px/frame motion **1.70 px mean / 2.92 peak, unchanged**; `detect_every 3` embeddings
  **1.06 → 0.39** per frame. On a 12-frame clip built from a real photo plus a one-frame false
  detection in the background, the old code drew 2 boxes and embedded 2 faces from frame 4 to
  the end; the new one draws 1, keeps the right label through the prediction-only tail, and
  embeds nothing there.

### Changed

- **The alignment fill lives on `FeatureExtractor`, not a library-wide global.** A global
  setter was called *after* the `FeatureDatabase` constructor, i.e. after registration —
  gallery features were built with one fill and queries with another, which is exactly the
  "registration and inference must preprocess identically" rule this project already had.
  Both paths take the same extractor, so they cannot disagree now: `align_detection` takes
  the fill as a parameter, `extract_face_feature(s)` read it from the extractor.
- **`align_detection` dropped its bbox fallback.** It was unreachable (`align_face` only fails
  on degenerate landmarks, which the eye-distance check rejects first) and, if it had run, it
  would have handed the embedder a crop outside the ArcFace convention — a silent score
  degradation with a different distribution than the gallery's. Degenerate landmarks mean an
  empty feature and an `Unknown` label again.
- **`copyMakeBorder` is now called with `cv::BORDER_ISOLATED`.** Without it, OpenCV fills the
  border from the *parent* `Mat` when the source is a submatrix and silently ignores the
  requested border type (verified on 4.13 with `BORDER_CONSTANT` returning parent pixels).
  `align_detection`'s ROI is such a submatrix, so a caller passing an ROI used to get pixels
  from outside the image it handed over.
- **The recognition-label loop is per face**: a local `label` replaces the `max_name`/
  `max_similar` pair that used to leak across iterations, and the dead `i < dets.size()`
  guard is gone (`extract_face_features` returns exactly one entry per detection, empty
  where alignment failed). Behaviour unchanged.
- Docs: `config.example.yml` regenerated with the new keys; the README no longer advertises
  the unused `openface.nn4.small2.v1.t7` leftover.

- `embedder.input_w/h` is now a *geometry* knob, not just a blob resize: the aligned crop is
  warped to exactly the size the extractor declares, instead of always 112×112 and being
  resampled again. `extract_face_feature(s)` take their output size from
  `FeatureExtractor::inputSize()`, so there is no second place where the 112 lives. At 112 the
  change is bit-identical (measured `maxabsdiff = 0` against the previous path on both
  identities), and a foreign embedder at its native shape works (opencv_zoo SFace, 128-d,
  self-match 2/2).
- Precedence is built-in defaults < `--config` file < command line; the
  `cpu|cuda|openvino` and `batch|nobatch` tokens are folded into the same ordered override
  list instead of carrying dedicated variables.
- `main()` is argument parsing and configuration only; the pipeline moved into
  `run_pipeline()`, with a single `cv::Exception` catch at the boundary.
- Documentation and the tuning table now refer to config keys instead of `main.cpp` lines.

---

- `Detector` no longer owns a `PriorBox`; it owns a `DetectorHead` resolved at construction, so
  adding a third layout is one class, not another config key. `PriorBox::decode` became `const`.

---

## v1.0.0 — 2026-10-24

### Fixed

- **Face alignment was a no-op.** `face_alignment` was called with the reference eye
  positions equal to the detected ones, so `scale = ref_dist / dist` evaluated to `0` and
  every embedded face was warped down to a point. Replaced with `align_face`, which fits a
  similarity transform (rotation + uniform scale + translation, closed-form least squares)
  from all five landmarks onto the standard ArcFace 112×112 reference template.
- **`PriorBox::decode` swapped two landmarks.** YuNet emits
  `[right_eye, left_eye, nose, mouth_left, mouth_right]`, but offsets 8/9 were read as
  `mouth_left` and 10/11 as `nose`. Detected by geometry: the "mouth" point sat vertically
  between the eyes and the mouth corners, at the eye midpoint horizontally. Landmark fit
  residuals went from 13–23 px to 3–5 px once corrected.
- **Unscaled detection coordinates when drawing.** `main` no longer left the camera frame
  at camera resolution while `draw()`/`putText()` still multiplied detector coordinates by
  `rfactor` (valid only for a `960×720` frame), so boxes landed in the wrong place with
  proportionally amplified motion. The frame is now resized to `display_shape` up front,
  which also keeps inference preprocessing identical to registration.
- **Tracker boxes leapt past the face.** Velocity was assigned the *total* displacement
  between two detections and then applied once per frame; a fresh track sat still for
  `detect_every` frames and then jumped the whole gap, overshot, lost IoU association and
  died — boxes flickered and lurched. Velocity is now estimated per frame
  (`Δcenter / frames_since_update`) with the anchor pinned to the last matched detection,
  tracks are advanced every frame so matching is temporally aligned, and the first
  estimate is not smoothed.
- **New faces were invisible for ~15 frames.** Track confirmation required 3 detection
  matches while detection itself ran every 5th frame. `min_hits` now defaults to 1.
- **`free()` used on a `new`-allocated object.** `FeatureDatabase`'s destructor called
  `free(similar_calculator)`, and `Similarity` had no virtual destructor. Now
  `std::unique_ptr<Similarity>` with `virtual ~Similarity() = default`.
- **`main` segfaulted with no arguments** by dereferencing `argv[1]` unguarded.
- **Registration never found any photos.** `cv::glob(path + name)` matched the directory
  itself rather than its contents; now `path + "/" + name + "/*"`.
- **Registration could mis-attribute features to the wrong person.** Directory listing
  assumed the first two entries were exactly `.` and `..` and paired `names[i + 2]` with
  `feature_database[i]`, which breaks on any extra dot-entry or readdir reordering. Now
  `list_person_dirs()` (dot/non-directory filtering + sorting) with explicit
  name→feature pairing.
- **`PriorBox::decode` trusted raw `float*` buffers**; input type and continuity are now
  checked, returning no detections on malformed tensors instead of reading out of bounds.

### Added

- **Face tracker** (`face_tracker.{hpp,cpp}`): IoU greedy matching + constant-velocity
  prediction, per-track IDs, and gap filling for frames where detection misses a face
  (occlusion), so boxes stay continuous.
- **Backend configuration** (`DNNConfig`): `./main <id> [cpu|cuda|openvino]`, with CPU as
  the zero-config default.
- **Any input source, not just a camera index**: the first argument is an integer camera
  index (no longer truncated to one digit) or anything `cv::VideoCapture` accepts — image
  file, video file, device path, URL/stream. For non-camera sources the window holds on the
  last frame instead of exiting instantly. Verified: a still image and a 3-frame MP4 both
  run through the full pipeline; a two-person composite resolves both identities correctly.
- **`scripts/get_models.sh`**: fetches the exact YuNet detector this project is built
  against (pinned to the upstream commit that shipped it — the old branch path 404s after
  upstream restructured) and verifies its sha256. Idempotent.
- `-h` / `--help`; unknown options now print usage instead of a bare message.
- **Batch embedding** (`FeatureExtractor::extract_batch`) — one forward pass for all faces
  in a frame. Auto-disabled on CPU targets because OpenCV's CPU backend runs the graph per
  sample and rebuilds internal blobs on batch-size changes (measured 1.31× *slower* on a
  realistic varying-face-count sequence); enabled automatically for GPU/NPU. Overridable
  via `[batch|nobatch]`. Results are bit-identical either way.
- **Pure-function alignment/extraction API**: `align_detection`,
  `extract_face_feature`, `extract_face_features` (the latter keeps
  `features[i] ↔ dets[i]` correspondence and picks batch vs per-face internally), replacing
  duplicated coordinate/index arithmetic in the call sites.
- Startup log of the number of enrolled people, for headless verification.

### Changed

- **Preprocessing now preserves aspect ratio**: `letterbox()` (in `utils.cpp`) scales a
  frame into the 960×720 canvas and centres it on black padding, replacing the straight
  stretch. Both paths use it — the live loop and `features_from_img` (registration) — so
  the two stay comparable. Downscales use `INTER_AREA`; upscaling a 4:3 camera frame is
  byte-for-byte the old behaviour.
  Measured (paired A/B, same 12 warps per identity, one run):
  within-identity 0.805 → **0.845**, between −0.019 → **−0.029**, margin 0.824 → **0.874**.
  Attribution: swapping only the interpolation (`INTER_LINEAR` → `INTER_AREA`) gave
  0.800 / 0.824, i.e. no gain — the improvement comes from not distorting the face.
  A two-person composite (its query feature is preprocessing-invariant) now matches its
  own identities at **0.959 / 0.843** where the stretched gallery gave 0.586 / 0.696,
  while wrong-identity scores stay near 0. Cost: 0.21 → 0.27 ms/frame.
  Consequence: enrollment features move (cosine 0.83 between the two conventions), but the
  gallery is rebuilt from photos at every startup, so nothing needs migrating.
- Failing to open the input source now exits with a non-zero status and names the source
  (previously it printed "Cannot open the camera." and returned success).
- `features_from_img` no longer mutates the caller's `cv::Mat` (it clones internally).
- The unused, unimplemented `FaceRecognition` shell class and `read_directory` were
  removed.
- `nms` rewritten with an explicit kept-vector instead of an `erase`/`remove_if`
  counter trick.
- Build: `cmake_minimum_required` 2.8 → 3.10, C++17 enabled, header no longer listed as a
  source file.
- **Recognition quality (augmentation A/B, true cosine, 12 warped variants per identity):**
  within-identity 0.820 → **0.848**, between-identity 0.534 → **−0.068**, margin
  0.286 → **0.916**. The old eye-only alignment produced badly aligned crops whose features
  looked spuriously similar across identities, which is why the `0.2` match threshold had
  so little headroom before.
- **Tracking accuracy (translation simulation, face moving 6 px/frame):** worst box-centre
  error 31.5 px → **5.9 px**; worst inter-frame box jump 35.1 px → **11.8 px**; no more
  dropped frames.

### Known limitations

Frames from any source are stretched to a fixed 960×720 (`blobFromImage` and the display
resize do not preserve aspect ratio, so portrait inputs lose detection quality); detection
runs at a fixed 160×120, so small/distant faces are missed; `O(N)` linear scan over enrolled
identities; a display is required for the only output path; and there are no automated tests.
The bundled sample data (2 identities × 1 photo) is a regression signal, not an accuracy
benchmark.
