# Changelog

Notable changes to this repository. The format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses release tags rather
than a strict versioning scheme.

## [Unreleased]

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
- **Zero new dependencies.** The loader is ~35 lines of flat parsing rather than
  `cv::FileStorage`, which refuses any file that does not begin with a `%YAML:1.0`
  signature (it sniffs content and ignores the `.yml` extension) and throws when it
  disagrees — and this way file and `--set` share one parse-and-validate path.

### Changed

- Precedence is built-in defaults < `--config` file < command line; the
  `cpu|cuda|openvino` and `batch|nobatch` tokens are folded into the same ordered override
  list instead of carrying dedicated variables.
- `main()` is argument parsing and configuration only; the pipeline moved into
  `run_pipeline()`, with a single `cv::Exception` catch at the boundary.
- Documentation and the tuning table now refer to config keys instead of `main.cpp` lines.

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
