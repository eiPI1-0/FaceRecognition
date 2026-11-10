# Source layout, traps and validation

The layer chain is a build-graph fact and a ctest target; the traps are the ones that cost real debugging time.

Deep dive split out of [README.md](../README.md), which keeps the short version.
Related: [README.md](../README.md) for the measured numbers, [configuration.md](configuration.md) for the knobs table

---

## Source layout

`src/` is the only source root, and its subdirectories are layers. An include names the layer
it reaches into — `#include "core/engine.hpp"` — and may only point at that layer or a lower
one:

```
common  →  core  →  app  →  serve  →  cli          (src/main.cpp sits on top)
```

| Layer | Directory | Contents |
|---|---|---|
| 0 | `src/common/` | `log`, `json_util`, `encoding`, `sha256`, `metrics` — leaves: nothing in this repo depends on them |
| 1 | `src/core/` | the recognition core: `face_def` (`BndBox_xyxy` / `BndBox_xywh`, `Landmarks_5`, `Face`), `priorbox` (YuNet prior-box generation + raw-output decoding), `detector_head` (fused vs per-level, inferred from layer names), `utils` (`nms()`, `draw()`, `letterbox()`), `engine` (`DNNConfig`, `Detector`, `FeatureExtractor`, `Similarity`, header-only `FeatureDatabase<T>`, `align_face`/`align_detection`, `extract_face_feature(s)`, `list_person_dirs`), `face_tracker` (`FaceTracker` — constant-velocity prediction, two-gate association, dedup, lifecycle; emits `{face, id, measured}`). OpenCV only: no GUI, no config, no JSON, no network |
| 2 | `src/app/` | `config` (`PipelineConfig` + the `knobs()` table — every tunable described once, driving file parsing, `--set`, validation and `--dump-config`), `evaluation` (held-out FAR/FRR/EER, the pair→1:N conversion), `gallery` (feature index + persistence), `api` (`Pipeline` = detector + extractor + gallery, `IdentifyResult`, JSON serialization). All three shells go through this layer, so they cannot disagree |
| 3 | `src/serve/` | transports and capacity: `auth` (bearer tokens, audit, one access-log hook), `admission` (the single end-to-end capacity-wait budget), `timed_queue` (`TimedTaskQueue` around httplib's own `ThreadPool`), `pipeline_pool` (`PipelinePool` + `Lease` — the unit of concurrency and backpressure), `result_cache` (LRU keyed by image sha256 + params + fingerprint), `server` (cpp-httplib, `/v1/*`, `/healthz`, `/readyz`, `/metrics`), `mcp` (JSON-RPC 2.0, stdio + HTTP transports, four read-only tools) |
| 4 | `src/cli/` | the subcommand shells — one file per command (`identify`, `enroll`, `serve`, `mcp`, `token`, `compare`, `verify`, `evaluate`, plus `demo`), with the plumbing shared by them in `args.{hpp,cpp}` |
| 5 | `src/main.cpp` | argv → subcommand dispatch, usage text, exit codes (the demo capture loop lives in `src/cli/demo.cpp`) |

Everything in layers 0–3 is in `namespace facerec`; only `src/cli/` and the entry point are
outside it, and they reach the library through `using namespace facerec;` in a **`.cpp`** —
headers spell the qualified name (`facerec::PipelineConfig`) because a using-directive in a
header would export the whole namespace to every includer.

| Path | Role |
|---|---|
| `tests/` | `ctest` targets — `decode_gates`, `tracker_scenarios`, `admission`, `evaluation` — plus the `augmentation_ab` harness (needs `models/` and `data/`, so it is not in `ctest`); the fifth target, `layers`, is the script below |
| `scripts/check_layers.py` | the `layers` test: include direction above, and every `.cpp` under `src/` listed in `CMakeLists.txt` |
| `third_party/httplib.h` | vendored cpp-httplib v0.15.3 (header-only, pthread only), reached through the `facerec_httplib` INTERFACE target so its two compile-time limits cannot drift per target |

### Contributor notes (the traps in this codebase)

- **`cv::Size` is `{width, height}`.** `img_shape = {160, 120}` means 160 wide, 120 tall.
- **Detection coordinates are always in detector-input space (160×120)**, never in the
  image you passed to `detect()`. `align_detection(img, det, detection_shape)` maps them
  for you; pass the *same image* you detected on.
- **Any hand-written coord transform on a `Face` has to stay in that Face's own space.**
  `x' = img.cols − x` on a detection in 160×120 space (with the display frame 960×720) puts
  every landmark outside the image, and the failure is silent: the alignment returns an
  empty Mat and the face simply stays Unknown. Scale/mirror and the `detection_shape` you
  pass alongside have to be converted together — which is why callers normally just hand
  the raw detection to `align_detection()`.
- **`cv::copyMakeBorder` on a submatrix reads the parent image.** Without
  `cv::BORDER_ISOLATED`, OpenCV "fills" the border with real pixels from the parent Mat if
  they are available — the requested border type is ignored (verified on 4.13, even with
  `BORDER_CONSTANT`). `align_detection` passes the flag; any code cropping a `Mat` before
  aligning needs it too.
- **Every input frame is letterboxed onto `display_shape`** (`letterbox()` in `src/core/utils.cpp`)
  before tracking and drawing, and registration goes through the same helper — the two
  paths must preprocess identically or the features are not comparable. The padding is
  centred and the canvas is exactly `input_shape × rfactor`, so `draw()`/`putText()`
  scaling detection coords by `rfactor` stays exact with no extra offset.
- **One association gate is not enough, and predicted geometry must not be recognized.**
  A face at the frame edge alternates between the whole-face box and the visible-part box;
  measured adjacent-frame IoU is 0.37, so a 0.45 IoU gate split one face into two tracks
  that were both drawn every frame. Association therefore takes two gates (IoU *or*
  normalized centre displacement) over candidate pairs sorted by score, plus a dedup pass.
  `TrackView::measured` is the contract that keeps velocity-offset landmarks out of the
  embedder; `advance()` must stay a rigid translation — clamping box corners squeezed boxes
  at the edge and made the next association harder, which spawned more duplicate tracks.
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
- **`knobs()` in `src/app/config.cpp` is the only list of tunables.** File parsing, `--set`,
  range validation and `--dump-config` are all generated from it, so adding a tunable
  means adding a struct field *and* a table entry — never a bespoke parse branch.
  `app/config.hpp` is app-level: no `core/` header may include it (the `layers` test enforces
  this), because `FeatureDatabase<T>` is a header-only template and every consumer would
  recompile.
- **Relative `paths.*` values are resolved against `paths.root`** (default: the working
  directory; anchoring to the config file is opt-in via `paths.root: .` inside it), in one place, after defaults, file and
  `--set` have all been applied. Resolving earlier — or per source — produces the
  half-resolved mix that made this project unrunnable outside `build/`.

### How changes are validated

`make && ctest` is the cheap gate (no weights, under a second). Beyond it, verification is:
`./main 1` (expect `Loaded N person(s)` followed by `Cannot open source: 1` and a non-zero
exit), then a throwaway program in `build/` linked against
`../src/core/{engine,priorbox,detector_head,utils}.cpp` that registers from `./data/` and
re-recognizes each registration photo (expect that photo's own person as top match); an
`hconcat` of two people's photos exercises multi-face index mapping. For any
alignment/embedding change, embed ~12 warped variants of each identity (±25° rotation,
0.85–1.15 scale, small translations, brightness ±40) and compare mean **within-identity**
vs **between-identity** true cosine (`a.dot(b)/(‖a‖‖b‖)`, *not* `CosSimilarity`).
For tracking changes, compile `src/core/face_tracker.cpp` and the previous version (from git) against
one synthetic harness: paste a real face onto a blurred frame at controlled positions,
then compare *boxes drawn* (`tracker.tracks()`) against *detections returned*
(`detector.faces()`) over scripted sequences — geometry alternating at the frame edge, two
faces at 0.7/1.0/1.5 face-widths apart, a target that vanishes, single- and two-frame false
positives, motion with a 3-frame detection miss, motion at 7 px/frame for box error.
Baseline (defaults `iou 0.30` / `dist_ratio 0.60` / `max_lost 5` / `min_hits 2`): edge case
1.00 boxes/frame with 0 doubled frames, 0 ghost boxes after a miss burst, single-frame false
positives never drawn, mean box-centre error 1.70 px (peak 2.92) at 7 px/frame. The old
tracker scored 1.98 / 5 / drawn-11-frames / 1.70 on the same sequences.

Current baseline (letterbox pipeline, warp seed `12345`, 13 samples per identity):
**within 0.845 / between −0.029 / margin 0.874**; the same harness run against the old
stretch preprocessing scored 0.805 / −0.019 / 0.824. These numbers are harness-specific —
compare only against a re-run of the same generator (an earlier, different augmentation
draw produced 0.848 / −0.068 / 0.916, which is not comparable).

---
