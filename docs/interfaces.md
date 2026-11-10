# CLI interfaces: identify, compare, verify, demo

One binary, six commands, machine-parseable stdout. The exit-code contracts are the part worth reading twice.

Deep dive split out of [README.md](../README.md), which keeps the short version.
Related: [serving.md](serving.md) for the HTTP surface, [configuration.md](configuration.md) for shared flags

---

### identify — stateless, script- and LLM-friendly

```sh
./main identify --image photo.jpg
./main identify --image photo.jpg --top-k 3 --include landmarks,crops
./main identify --image photo.jpg --detector-input 640x480 --set recognize.thresh=0.35
```

stdout is **only** JSON; logs go to stderr. Exit status is 0 whenever the request was
served — including "no face found" (`"faces": []`) and "face found but unknown"
(`"matched": null`) — because those are answers, not failures. Non-zero means the request
itself was wrong (bad path, unreadable image, invalid config).

```json
{"status":"ok","source":"photo.jpg","image_size":[640,607],"detector_input":[160,120],
 "timing_ms":{"detect":1.03,"embed":5.22,"total":6.25},
 "model":{"detector_model":"models/yunet_120x160.onnx","embedder_model":"models/mobilefacenet.onnx","gallery_size":2},
 "faces":[{"index":0,"bbox":[299.1453,71.7509,111.3053,126.4719],"det_score":1.0,"aligned":true,
           "matched":{"person_id":"BruceLi","score":1.0},
           "top_k":[{"person_id":"BruceLi","score":1.0},{"person_id":"IvanYang","score":0.0226}]}]}
```

`bbox` and `landmarks` are in **original-image pixels**, not in the detector's 160×120
space — the letterbox mapping is inverted for you.

### compare / verify — the same check without a server

```sh
./main compare data/BruceLi/photo.jpg data/IvanYang/photo.jpg   # {"cosine":0.022648,...}
./main verify  data/BruceLi/photo.jpg --name BruceLi            # {"verified":true,...}
```

Both use the same `Pipeline::embed_image` + cosine as HTTP `/v1/compare` and MCP
`compare_faces`, so the three interfaces return bit-identical numbers (verified: 0.022648
on all three for the same pair).

The face each image contributes is its **most confident** detection — `identify`'s
`faces[0]`, which NMS orders by score, **not** the largest box. Keeping that one ordering
everywhere is what makes "the face compare compares" and "identify's first face" the same
face; describing it as the largest face (as an earlier tool description did) is a lie the
model would repeat to the user.

`verify` **exits non-zero when the answer is no**:

| exit | meaning |
|---|---|
| 0 | verified — best match is the named person at or above `recognize.thresh` |
| 1 | **not** verified (a valid answer: below threshold, or matches someone else) |
| 2 | cannot answer — unreadable image, or no face found |
| 3 | pipeline error |

The JSON verdict (`verified`, `cosine`, `matched`, `threshold`) is printed for every exit
code. `cosine` is the **best match's** score and `matched` names who — not a score against the
named person. So `"verified":false,"cosine":1.0,"matched":"BruceLi"` with `--name IvanYang` means
"this is confidently somebody else", which is a different answer from the other no: `cosine` below
`threshold` means nobody in the gallery is close enough. Scripts should branch on the exit code or
on `matched`, never on `cosine` alone. The non-zero-on-"no" is deliberate and differs from `identify`, which exits 0 even
when no face is found: `identify` reports a measurement, while `verify` is an
authentication-style decision, and a check that always exits 0 turns every script that
forgets to parse the JSON into a silent pass.

```sh
if ./main verify suspect.jpg --name BruceLi > /dev/null; then echo present; else echo absent; fi
```

### demo — the interactive window

```sh
./main demo <source> [cpu|cuda|openvino] [batch|nobatch]
             [--config <path>] [--set key=value] [--dump-config]
```

`<source>` is either a **camera index** (any integer) or anything `cv::VideoCapture`
accepts — an image file, a video file, a camera device path, a URL/stream. A still image
is processed as its single frame; for non-camera sources the window stays open on the last
frame until you press `Esc`, instead of exiting immediately.

```sh
./main 0                          # camera 0, CPU backend (legacy form of `demo`)
./main demo 1 cuda                # CUDA backend (needs OpenCV built with CUDA)
./main demo 0 openvino            # Inference Engine backend
./main demo 0 nobatch             # force per-face forward passes
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
elsewhere (see [Configuration](configuration.md#configuration)).

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
