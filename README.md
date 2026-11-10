# FaceRecognition — real-time face detection + recognition in C++

Real-time face **detection and recognition** built on the **OpenCV DNN module only** —
no dlib, no ncnn, no TensorFlow/PyTorch runtime. One dependency, one binary.

```
camera frame ──▶ YuNet detector ──▶ 5-point similarity alignment ──▶ MobileFaceNet
                   (160×120)              (112×112 crop)                (512-d embedding)
                                                  │
              tracker (two-gate association, stable IDs, gap filling)
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

## Run

The binary is a multi-command tool. `demo` is the original interactive window; the other
commands are for scripting, services and LLM tool use.

```sh
./main identify --image <path> [options]     # one image, JSON on stdout
./main enroll   --name <person> [--dry-run|--yes] <photo> ...
./main serve    [--host H] [--port P] [--workers N]    # HTTP service
./main demo     <source> [cpu|cuda|openvino] [batch|nobatch]
./main mcp    [--transport stdio|http] [--port P]    # MCP tools for LLM clients
./main compare  <img-a> <img-b>                      # cosine between two images' faces
./main verify   <img> --name <person>                # is this image this person?
./main evaluate <dataset> [--gallery-photos N] [--probe-photos N] [--csv out.csv]
                                         # FAR/FRR + EER at 1:1 and 1:N — picks the threshold
```

`./main <source>` without a command is still treated as `./main demo <source>`, so older
invocations keep working.

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

### serve — HTTP service

```sh
./main serve --port 8080 --workers 8          # workers = concurrent requests
curl -X POST --data-binary @photo.jpg -H 'Content-Type: image/jpeg' \
     localhost:8080/v1/identify
curl -X POST -H 'Content-Type: application/json' \
     -d '{"image_b64":"...","top_k":3,"include":["landmarks"]}' \
     localhost:8080/v1/identify
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/identify` | raw image body **or** JSON `{"image_b64", "top_k", "min_score", "include"}` |
| `POST /v1/embed` | 512-d embedding of one face (`face_index`, default 0) |
| `POST /v1/compare` | `{"a":[...],"b":[...]}` → cosine |
| `GET /v1/people` | enrolled person ids (never embeddings or photos) |
| `GET /healthz` / `GET /readyz` | liveness / readiness (`/readyz` also reports cache hits/misses) |

`/v1/identify` accepts a request-level detector size — more pixels through the detector for
group photos and distant faces:

```sh
curl -X POST -H 'Content-Type: application/json' \
     -d '{"image_b64":"...","detector_input":{"w":640,"h":480}}' \
     localhost:8080/v1/identify
```

It costs roughly quadratically (measured on one image: 160×120 = 1.6 ms, 320×240 = 11.4 ms,
640×480 = 35 ms, 960×720 = 87 ms). The letterbox canvas stays fixed, so the alignment crop
still comes from the same resolution the gallery was built at. If you routinely use a large
detector size, raise `detector.input_w/h` in the config as well so gallery and queries are
built at the same geometry — the request-level knob buys *recall*, not accuracy.

Identical requests are cached by `sha256(image bytes) + params + model/gallery
fingerprint`, and the key is computed **before** decoding, so a hit skips `imdecode`,
detection and embedding. Measured with one image repeated by 16 clients: 205 → 1347 req/s,
p50 72.8 ms → 4.4 ms. Disable with `--cache-entries 0`; `X-Cache: HIT|MISS` is returned.
Changing the gallery requires a restart for the cache to be invalidated (its fingerprint is
taken at startup).

Concurrency model: one `Pipeline` (its own DNN nets) per worker, requests beyond that
queue and eventually return `503` (the lease times out at 5 s). `--workers` defaults to the
CPU count; each pipeline costs ~7 MB RSS and one core, so 16 workers ≈ 220 MB. `--cv-threads 1`
is the default — the model is "many pipelines in parallel", not "one Net spread over threads";
raising it re-introduces thread-pool contention inside every convolution.

Serving throughput (this machine, 16 cores, **Release**, `--cache-entries 0`, one face per image,
4×W simultaneous clients, fresh connection per request):

| `--workers` | p50 | p90 | max | req/s | RSS |
|---|---|---|---|---|---|
| 4 | 74 ms | 108 ms | 117 ms | ≈135 | 112 MB |
| 8 | 100 ms | 136 ms | 150 ms | ≈193 | 147 MB |
| 16 (64 clients, oversubscribed) | 185 ms | 271 ms | 307 ms | ≈187 | 217 MB |

Beyond ~8 workers the CPU is saturated; scale out with several processes behind a
reverse proxy rather than by adding workers.

**Authentication and audit.** The server binds `127.0.0.1` by default and does **not** ship a
TLS terminator: put a reverse proxy in front for both. Inside the process there is one rule
that cannot be forgotten, because it is a bind constraint rather than a flag — **a
non-loopback `--host` without a token table refuses to start**:

```sh
./main token --name ci --tokens-file ./tokens.txt    # prints the token once; stores only its sha256
./main serve --host 0.0.0.0 --port 8080 --set auth.tokens_file=./tokens.txt
curl -H "Authorization: Bearer <token>" -X POST --data-binary @photo.jpg \
     -H 'Content-Type: image/jpeg' localhost:8080/v1/identify
```

`tokens.txt` is line-based (`name:sha256:<64 hex>`, `#` comments), compared in constant time, and
written `0600` — the file holds no usable credential, so a leaked copy is not a leaked key.
`/healthz` and `/readyz` stay open (probes); everything else, `/metrics` included, needs the
token. Every request is one JSON line on `log.sink` (`stderr` by default, or a file path, or
`off`), and `--quiet` silences only successes — a refused request is always logged, because a
silent brute-force attempt is exactly what you cannot afford:

```json
{"audit":"request","ts":"2026-11-07T03:39:09.024Z","svc":"http","method":"POST","path":"/v1/identify",
 "status":401,"ms":0,"caller":"","bytes":0,"auth":"denied","peer":"127.0.0.1","presented":false}
```

Query lines deliberately carry **no person id, no matched name, no embedding, no image bytes**:
an audit log that maps requests to identities is a second biometric database with none of the
protections. Enrollment *is* attributable (`./main enroll` logs the person and the files it
wrote), because writing a template is the irreversible act.

**Operational notes**

- `GET /metrics` — Prometheus text: `facerec_http_requests_total{path,status}` (every route,
  every status: 401/400/413/503/500 all visible; `path` is a **fixed** whitelist, anything
  unrecognised collapses into `path="other"` so a scanner cannot inflate label cardinality),
  `facerec_faces_detected_total`, `facerec_matches_known_total` / `..._unknown_total`,
  `facerec_cache_hits_total` / `..._misses_total` / `..._evictions_total`, and histograms
  `facerec_decode_ms`, `facerec_detect_ms`, `facerec_embed_ms`, `facerec_request_ms`,
  `facerec_match_score`.
  Request counting happens in **one** place — the access-log hook in `src/serve/auth.cpp` — so a new
  route can never be under-counted and an early-return branch can never be missed.
  Note: counters are **per process**; with `--reuseport` a scrape lands on one arbitrary
  instance, so aggregate across instances (see `deploy/`) instead of trusting a single scrape.
- **Logging: one JSON-lines stream, two guarantees.** `log.sink` picks the destination
  (`stderr` default, `off`, or a file path); `log.level` (`off|error|warn|info|debug`, default
  `warn`) filters **operational** events only. Audit lines ignore the level — the only way to
  silence them is `log.sink: off`, which is a deliberate configuration act, not a level slip.
  Both kinds share one destination and one lock, so they interleave in time order: "was the 503
  before or after the index rebuild" does not require correlating two files. Measured: with
  `log.level: off` a 401 still produces its audit line (with peer address and whether a token was
  presented) while every event line disappears.
  Events: `pool_busy` (warn — a 503 the client actually received, with `waited_ms`, `queued_ms`,
  `budget_ms`, `deadline_ms`, `workers`), `queue_shed_start` (warn — **emitted once** when admission
  shedding begins; subsequent sheds are counters, not log lines, because a line per shed is the P1
  outage re-imported), `lease_wait` (info — saturation precursor; 48 concurrent requests produce ~87 of
  these, which is why it is not warn), `pipeline_error` (error — the exception text otherwise
  only ever reaches the response body), `unknown_tool` (MCP: the trace a prompt-injection probe
  for `enroll_person` leaves server-side).
- **The unit of concurrency is the connection, not the request.** httplib queues one task per
  accepted connection, and that task serves up to `keep_alive_max_count` requests — while holding
  a worker thread, blocked in `read` for up to `read_timeout` (10 s) between them. Measured: 12
  requests over **one** keep-alive connection raise `facerec_queue_wait_ms_count` by **1**; the
  same 12 over 12 connections raise it by **12**. So `--http-threads` should be sized against
  expected concurrent *connections*: `N` persistent clients and fewer threads than that means
  head-of-line blocking by connection, not by request.
- **One capacity-wait budget, enforced at both admission points.** Waiting for capacity happens
  twice: in httplib's task queue (accept thread → worker), then in `PipelinePool` (handler →
  pipeline). Giving each stage its own ceiling was wrong — the worst case was *queue ∞ + pool 5 s*.
  `--queue-timeout-ms` (default 5000) is now the **end-to-end** budget: a connection that has
  already waited past it is shed in the pre-routing hook (503, `code:"queue_expired"`,
  `Retry-After: 1`) without touching a pipeline, and whatever is left is the pool's budget.
  Measured continuity (`--workers 2 --http-threads 12 --queue-timeout-ms 100`): every refused
  request's `queued_ms + waited_ms` equals the deadline — 48+52, 30+70, 66+34.
  Values: `N>0` = the budget; **`0` = zero tolerance** (fail if capacity isn't free right now, the
  historical meaning); **`-1` = never shed**. `0` deliberately does not mean "off" — redefining an
  existing value silently would be the worst way to make this change.
- **What shedding actually buys, measured.** Open-loop flood at 140 req/s against 2 workers
  (capacity ≈ 98 req/s, cache off, 1.3 MB JPEG), 6 s:

  | setting | p50 over the run | p99 | outcome |
  |---|---|---|---|
  | `--queue-timeout-ms -1` (no shed) | 232 → 644 → 1061 → 1489 → 1909 → **2338 ms** | 2526 ms | 840/840 "served" |
  | `--queue-timeout-ms 150` | **~161 ms, flat** | 171 ms | 592 served + 248 fast 503 |

  Without shedding, latency grows **linearly forever** and nothing is refused: a 20 ms job makes
  the client wait 2.3 s, and the CPU spent on callers who already left is exactly what the queued
  requests need — that is the death spiral. With it, every client gets an answer in ≤ 173 ms.
  **Throughput does not drop**: 592/6 s = 98.7/s is the capacity, which the no-shed run also
  could not exceed. The "100% served" column is not a win — those callers waited seconds, and any
  real client with a 1 s timeout would have failed anyway, after the work was already burned.
- **Probes and `/metrics` are never shed** (verified 200 while the flood was running). Shedding a
  probe turns an overload into a pod restart, and shedding `/metrics` removes exactly the data
  needed to diagnose it. This exemption is **path-based, so it only exists at the HTTP layer**.
- `--queue-max-pending` (default 0 = unbounded) is the resource backstop: a hard cap on
  connections waiting in front of the workers. Measured with `--workers 2 --queue-max-pending 4`
  and 30 simultaneous arrivals: **6 served, 24 connections closed, `facerec_queue_dropped_total 24`**.
  The client sees a broken pipe, **not** a 503 — at that point the request has not been parsed, so
  no HTTP answer is possible. **Measured hazard:** in that same run, 5 concurrent `/healthz`
  requests were **all dropped** (0×200). A connection-level drop cannot honour the path exemption,
  so turning this on puts liveness probes at risk; that is why the default is off and why the
  deadline is the primary control. Protecting probes at the connection level would need a separate
  listener, not a bigger cap.
- `facerec_queue_wait_ms` is a **histogram over connections**; `facerec_queue_expired_total` and
  `facerec_queue_dropped_total` are the two shed paths at the two admission points. `pool_busy`
  events now carry `queued_ms`, `budget_ms` and `deadline_ms`, because "used the whole budget in
  the queue" and "queue was instant, pool was empty" are different faults with different fixes.
- The MCP HTTP transport follows the same rules (same hook, same budget); stdio never enters a
  queue, so its budget is the full deadline.
- SIGINT/SIGTERM stops accepting new connections and lets in-flight handlers finish.
- The listen backlog is set to 1024 (`CPPHTTPLIB_LISTEN_BACKLOG`, see `CMakeLists.txt`);
  cpp-httplib's default of 5 makes the kernel drop SYNs past 5 simultaneous connects, which
  shows up as a mysterious ~1 s latency tail that server-side timings do not explain.
- **Auth is a binding constraint, not a switch.** Tokens live in `auth.tokens_file`
  (`name:sha256:<64 hex>` per line, created `0600` by `./main token --name <caller>`; the token
  itself is printed once and never stored). Every route except `/healthz` and `/readyz` requires
  `Authorization: Bearer <token>`. Binding a **non-loopback** `--host` with no token table
  **refuses to start** — so `--host 0.0.0.0` can never be a silent open door. A 401 always writes
  an audit line (with peer address and whether a token was presented), even under `--quiet`.
- TLS is still the reverse proxy's job (`deploy/nginx/facerec.conf` terminates it and proxies to
  a loopback-bound instance). `--max-request-mb` and the pixel cap reject oversized /
  decompression-bomb bodies before decoding.
- `paths.gallery_index` holds **plaintext 512-d templates plus names**, so it is created `0600`
  and a wider mode triggers a one-line warning with the fix (`chmod 600`). Treat backups and
  copies of it as biometric data; `paths.gallery_index: ""` keeps templates in memory only.
- `curl --data-binary @photo.jpg` sends `Content-Type: application/x-www-form-urlencoded`, and
  cpp-httplib caps form bodies at 8 KB (413) unless told otherwise — the build pins
  `CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH`, so the payload limit is the only size gate
  and the endpoint's meaning does not depend on the client's `Content-Type`. Send
  `image/jpeg`/`octet-stream` anyway: it is what the API documents.
- The gallery is rebuilt from `data/` at startup and cached to `paths.gallery_index`
  (invalidated by a model/gallery fingerprint change).

### Deployment

Three files in `deploy/`, each doing one job:

| File | Job |
|---|---|
| `deploy/Dockerfile` | multi-stage image: build stage carries `opencv-devel`, runtime carries only `libopencv413` + the binary. Non-root, `serve` as ENTRYPOINT, assets and index on volumes. |
| `deploy/systemd/facerec@.service` | template unit. `systemctl start facerec@{1..4}` starts four processes on one port; `--reuseport` lets the kernel distribute connections. |
| `deploy/nginx/facerec.conf` | TLS termination, `/metrics` restricted to private ranges, everything else proxied to a loopback-bound instance. |

**Scale out, not up.** One process already uses every core (one pipeline per worker), so the
axis that matters is *processes*, and `--reuseport` is what makes that a one-flag change: each
instance keeps its own pipeline pool and its own result cache, no coordinator, no shared memory.
Verified here: two processes on one port, `ss -lnt` shows two LISTEN lines, and 60 requests split
21 / 39 in the access log.

**What to expect when you deploy:**

- Probes are **not** logged when they succeed (k8s hits `/healthz` every 10 s). Verifying traffic
  distribution with `/healthz` therefore shows nothing — use a non-probe route or the counters.
- `/metrics` is **per process**. With `--reuseport` a scrape reaches one arbitrary instance, so
  configure one scrape target per instance (or a sidecar) and sum in PromQL; a single scrape is a
  sample, not the total.
- `log.sink: stderr` is the default and is the right choice in a container: the orchestrator
  collects stderr alongside the stdout banner, so no log-volume mount is needed. (`log.sink` is
  the renamed `log.audit`; operational lines from `log.level` go to the same stream.)
- The runtime dependency on openSUSE Leap 16 is the **single** `libopencv413` package — there is no
  per-module `libopencv_core413` (§ note: `rpm -qf /usr/lib64/libopencv_core.so.413`).

The CMake target is named `main`; the Dockerfile and the systemd unit install it as `facerec`.

### mcp — Model Context Protocol tools

```sh
./main mcp                        # stdio (Claude Desktop / IDE)
./main mcp --transport http --port 8081 --workers 8    # POST /mcp
```

Four read-only tools, each with a JSON Schema:

| Tool | Arguments | Returns |
|---|---|---|
| `identify_faces` | `image_b64`, `top_k`, `min_score`, `include_landmarks` | per-face bbox (original pixels), `matched` or `null`, ranked candidates |
| `compare_faces` | `image_a_b64`, `image_b_b64` | cosine of the most confident face in each (identify's `faces[0]`) |
| `list_people` | — | enrolled person ids (never embeddings or photos) |
| `get_face_crop` | `image_b64`, `face_index` | aligned 112×112 PNG as `image` content |

**Enrollment is deliberately not a tool.** Writing biometric templates is a
non-retractable action and must not be reachable from a prompt-injectable caller; use
`facerec enroll` or an admin API instead. Calling `enroll_person` returns
`-32602 unknown tool`.

The `identify_faces` description (which the model actually reads) states that
`"matched": null` means *unknown*, and that candidates must not be turned into an
identity — a low score is not a weak guess, it is "not this person".

Claude Desktop / MCP client config:

```json
{"mcpServers":{"facerec":{"command":"/abs/path/build/main",
                          "args":["mcp"],
                          "cwd":"/abs/path/build"}}}
```

`mcp` links the library directly rather than shelling out, so a tool call costs one
forward pass instead of a cold `readNet` of both ONNX files.

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

Every tunable is one entry in a single table (`knobs()` in `src/app/config.cpp`) with a dotted key.
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

## Registration

`build/data/<name>/` containing that person's photos. At startup every photo is
detected, aligned and embedded; the per-person feature is the mean of the
**L2-normalized** embeddings, re-normalized. Photos with no detectable face are skipped.

Add people from the command line — validation happens before anything is written:

```sh
./main enroll --name BruceLi --dry-run a.jpg b.jpg   # validate, write nothing (default)
./main enroll --name BruceLi --yes a.jpg b.jpg       # copy into data/BruceLi/
```

`--yes` copies only the photos that contain a usable face, never clobbers an existing file
(`_1` suffix) and restricts `--name` to one safe path component (`../evil`, `a/b` are
rejected). A **running** server keeps its in-memory gallery: restart it to pick up new photos.

More photos per person ⇒ more stable fused feature. Accuracy is dominated by how close
the recognition environment (distance, lighting, camera) is to the registration photos.

The built features are cached in `paths.gallery_index` (default `./gallery.idx`; an empty
value disables the cache). The index carries a fingerprint of the model files and of every
registration photo (path, size, mtime) — any change rebuilds it from the photos, so a change
in the preprocessing convention needs no migration step, and a corrupt index just rebuilds.

---

## Tuning

Tunables are config keys (defaults in `src/app/config.hpp`, full list via `--dump-config`); the
ones worth knowing:

| key | default | effect |
|---|---|---|
| `recognize.thresh` | `0.45` | cosine above which a match is reported. **Not a per-pair number** — see [Choosing the recognition threshold](#choosing-the-recognition-threshold-measured-not-chosen) below: `identify` takes a max over the whole gallery, so the same value means something different at every gallery size. The old default `0.2` admitted essentially every stranger to a 610-person gallery. |
| `display.rfactor` | `6` | display frame = `detector input × rfactor` = 960×720; the letterbox canvas and the coord scale for `draw()`/`putText()`. |
| `detector.conf_thresh` / `detector.nms_thresh` / `detector.keep_top_k` | `0.6` / `0.3` / `750` | detection sensitivity and box suppression. |
| `tracker.iou` / `tracker.dist_ratio` | `0.30` / `0.60` | association gates: a detection is the same track if box IoU ≥ `iou` **or** its centre moved less than `dist_ratio` × box size. Two gates, not one: a face at the frame edge alternates between the whole-face box and the visible-part-only box (measured adjacent-frame IoU **0.37**), so IoU alone splits one face into two tracks that are both drawn forever. |
| `tracker.detect_every` / `tracker.max_lost` / `tracker.min_hits` | `1` / `5` / `2` | `detect_every` below; a confirmed track is carried — box *and* its last real label — for at most `max_lost` frames of pure prediction; a track needs `min_hits` detections before it is shown, so a detection that exists for one frame never reaches the screen. |
| `align.border` | `constant` | how to fill the part of the alignment window that falls outside the frame for edge-clipped faces: `constant` (black) / `reflect` / `replicate`. Does not move the warp; see the measurement above. |


### Choosing the recognition threshold (measured, not chosen)

`identify` answers "who is this?" by scoring the query against **every** enrolled person
and keeping the maximum. So the false-accept rate is a property of the gallery size, not
just of the threshold: a per-pair false-accept rate of 1% means ~99.7% against a 610-person
gallery (1 − 0.99^609). That is arithmetic, not a prediction.

Measured on LFW (deepfunneled) through the **production** path — YuNet detection, landmark
alignment, MobileFaceNet, the same letterbox canvas, the same enrollment fusion — with
photos held out of enrollment (`./main evaluate <dataset>`):

| thresh | 1:1 FAR | 1:1 FRR | 1:N FAR, N=610 | 1:N FAR, N=1680 | 1:N FRR, N=610 |
|---|---|---|---|---|---|
| 0.20 (old default) | 1.26e-2 | 0.022 | **0.997** | **1.000** | 0.017 |
| 0.30 | 1.36e-3 | 0.027 | 0.548 | 0.817 | 0.019 |
| 0.40 | 8.6e-5 | 0.041 | 0.068 | 0.122 | 0.023 |
| 0.45 (default) | 1.9e-5 | 0.057 | 0.013 | 0.030 | 0.026 |
| 0.50 | 3.4e-6 | 0.088 | 0.0016 | 0.011 | 0.040 |
| 0.55 | 3.4e-6 | 0.140 | 0.0008 | 0.0054 | 0.065 |

The 1:1 columns and the two N=610 columns come from **one** run; the N=1680 column comes from
the second run below it (fewer enrolled photos per person, so its 1:1 tail is slightly thinner).
1:1 EER sits at threshold ≈ 0.18, rate ≈ 4.1% in the 610-identity run. Two runs over disjoint identity sets
(610 identities × 2 enrolled photos; 1680 × 1, a disjoint set) put the 1:1 FAR within ~2% of
each other at every threshold (1.88e-5 vs 1.84e-5 at 0.45) and the 1:1 FRR within ~13%. So the
*shape* belongs to the model; the decimals belong to the sample — which is why the last column is
rounded to two significant figures.

`0.45` is where the trade steepens: 0.20 → 0.45 cuts the 610-person false-accept rate 76×
for +0.9 pt of rejections, while 0.45 → 0.55 cuts it another 16× and *doubles* the
rejections. It is a defensible starting point for a small gallery, not a certified operating
point. Pushing to FAR=1e-3 at N=610 wants threshold ≈ 0.71 — that number comes from the
independence model, because this run cannot *measure* a 1:N FAR below 8.2e-4 (one probe in 1220;
the table prints `unreachable` below it), and at 0.70 the measured false-reject rate is already
29%. So a large unconstrained gallery is limited by the embedder, not by the threshold: no
threshold choice buys you past the model.

Two columns and one word are there deliberately:

- **1:1 and 1:N are separate columns.** `/v1/compare` and `compare_faces` are 1:1;
  `/v1/identify`, `/v1/verify` and `./main verify` are 1:N. A threshold tuned on one is
  wrong for the other by orders of magnitude.
- **`unreachable` and measurement resolution.** N people give N(N−1) impostor pairs, so a
  FAR below 1/(pairs) cannot be certified — only extrapolated. The report says `unreachable`
  and prints the certifiable floor instead of inventing a confident number.

Every shell that holds a gallery estimates this at startup **from the gallery itself** — no
dataset needed, because every cross-person pair inside it *is* an impostor pair:

```
recognize.thresh=0.20 against a 200-person gallery admits strangers: an unenrolled face is
matched to somebody with probability 95.3% (per-pair rate 1.518e-02 over 39800 gallery
cross-matches; a gallery-side estimate is good to the order of magnitude, not the third digit -
on LFW it ran 11 points high vs held-out probes). Raise recognize.thresh, or calibrate on your
own population with './main evaluate <dataset>'.
```

And the opposite failure — a threshold that looks safe but only because the gallery is too small
to see the tail:

```
recognize.thresh=0.45 against a 200-person gallery: no cross-person pair among 39800 exceeded
it, so the expected false-accept rate is below 5.0e-03 (this gallery's one-pair resolution).
Enroll more people, or run './main evaluate <dataset>' for a tighter number.
```

`serve` logs it as a `threshold_check` event (WARN once the estimate reaches
`recognize.self_far_warn`, default 0.01), prints it in its banner, and answers it at
`/readyz` (`recognize.far_1n_estimate`, `far_1n_resolution`, `gallery_persons`,
`gallery_fingerprint`); `mcp` logs the same event and prints the same sentence to **stderr** in
both transports (stdout is the protocol channel). `identify` deliberately stays silent: it is one
shot per process, so it would pay the estimate's matmul on every call and bury a caller's JSON in
prose — read `/readyz` or a server banner instead.

The estimate is good to the **order of magnitude, not the third digit**: on 200 LFW identities it said 95% where held-out probes measured 84% (gallery-side
pairs are not probe-side pairs, so the sign of that bias is empirical, not derivable), and
applying the same conversion to the measured 1:1 tail gives 1.14% at N=610 against 1.31%
measured directly. That is enough for what the check is for — telling 0.2 (99.7%) from 0.45
(1.3%) — and it recomputes itself whenever the gallery changes. The mechanism exists because a
false accept emits no error, no 503 and no latency spike: it looks exactly like success.

To get a number that applies to *your* population, measure on your own photos:

```sh
./main evaluate /path/to/dataset --gallery-photos 2 --probe-photos 2 --workers 8 --csv t.csv
# same layout as data/: one directory per person, >=4 photos each. The first 2 photos are
# the enrollment side; the next 2 are held out and never enrolled. --limit/--seed sample
# identities deterministically.
```

Provenance of the table above, so it is checkable rather than folklore — the numbers come from
these two runs (the original `vis-www.cs.umass.edu/lfw` host does not answer from this machine,
so the corpus is the ModelScope re-host of the Kaggle LFW package: dataset
`Composable/labeled_faces_in_the_wild`, file `LFW人脸数据集.zip`, which contains
`lfw-deepfunneled/`):

```sh
# 610 identities (those with >= 4 photos): 2 enrolled, 2 held out
./main evaluate <path>/lfw-deepfunneled --workers 8 --csv lfw_eval.csv
# 1680 identities (those with >= 2 photos): 1 enrolled, 1 held out
./main evaluate <path>/lfw-deepfunneled --workers 8 --gallery-photos 1 --probe-photos 1 \
    --csv lfw_eval_1680.csv
```

LFW is not your population: it is webcam-quality, near-frontal, and enrolled from a single
photo per person in the second run. Treat the *shape* (how steeply FAR falls, how far 1:1 and
1:N separate) as transferable and the absolute numbers as a floor to re-measure on your own
photos.

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

Frames carried by prediction are **not** embedded: `FaceTracker` reports each track as
`measured` or not, the app recognizes only measured ones, and the label persists on the
track's id. Predicted landmarks are last-known values offset by velocity — fine for drawing,
garbage for embedding, and the resulting embedding looks perfectly normal, which is how a
phantom box ends up wearing somebody's name. At `detect_every: 3` this cuts embeddings from
1.06 to 0.39 per frame at the same box accuracy.

Measured before/after the tracker rewrite (both trackers compiled against one synthetic
harness; one face unless stated):

| sequence | old (single IoU gate, `max_lost` 15, `min_hits` 1) | new |
|---|---|---|
| face at frame edge, box geometry alternating, 40 frames | 1.98 boxes/frame, doubled in 39/40 frames | 1.00 boxes/frame, 0 doubled |
| two faces 1.5 face-widths apart | 2.63 boxes/frame, peak 3 | 2.00, peak 2 |
| target vanishes (26 frames) | ghost box drawn 14 frames, embedded 14 times | carried 5 frames, embedded 0 |
| false positive present for one frame | drawn 11 more frames | never drawn |
| motion + 3-frame detection miss | 5 boxes far from the truth (the clamp pinned them at the frame edge) | 0 |
| box-centre error, 7 px/frame motion | 1.70 px mean / 2.92 peak | 1.70 / 2.92 (unchanged) |

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

| Path | Role |
|---|---|
| `tests/` | `ctest` targets — `decode_gates`, `tracker_scenarios`, `admission`, `evaluation` — plus the `augmentation_ab` harness (needs `models/` and `data/`, so it is not in `ctest`) |
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

## Limitations

- Frames from any source are letterboxed onto a fixed 960×720 canvas, so **aspect ratio is
  preserved**; non-4:3 sources get black bars in the window. The price of that simplicity:
  the detector still runs on the whole padded frame at 160×120, so on an extreme aspect
  (portrait phone frame) the face occupies fewer detector pixels than a crop-aware
  pipeline would give it.
- `identify` / `serve` / `mcp` are headless, but there is no batch mode that writes
  annotated images: the only visual output is the `demo` window, which needs a display.
- YuNet's landmarks are coarse: fit residuals are 3–5 px and detection boxes jitter by
  ~5 px frame to frame. Identity assignment is robust to this; box edges are not perfectly
  still.
- Detection runs at a fixed 160×120, so **very small or very distant faces** are missed;
  there is no image pyramid / tiling pass.
- Recognition scales linearly with the enrolled population (`O(N)` scan per face — a few
  microseconds per person at 512-d, so 10k people is still only tens of ms). The real ceiling
  at large N is **accuracy, not compute**: at the default threshold the 1:N false-accept rate
  is 1.3% at 610 people and 3.0% at 1680, and the threshold that FAR=1e-3 would require (≈0.71)
  already rejects 29% of genuine faces at that gallery size
  ([Choosing the recognition threshold](#choosing-the-recognition-threshold-measured-not-chosen)).
  A gallery of thousands needs a stronger embedder and/or a two-stage search (ANN candidate set,
  then re-verify at a per-candidate threshold), not just an index.
- The shipped sample data contains 2 identities with 1 photo each — useful as a regression
  signal, **not** an accuracy benchmark.
- The pinned **fused** YuNet export computes its last level unreliably under OpenCV 4.13:
  6 of its 1076 anchors (stride 64) come out buffer-state dependent — per Net instance, and
  stable for that instance — which is how `nan` scores and ghost boxes once reached clients.
  `PriorBox::decode` filters those candidates out (finite score + a box that could be a face
  in this frame), so they never surface in an API. The per-level export
  (`face_detection_yunet_2023mar.onnx`) showed 0/44 forwards affected, but has never been run
  through the augmentation harness, so it is not the default.
- Five `ctest` targets cover the decode gates, the tracker harness, admission accounting, the
  evaluation report, and the source layout itself (`layers`: include direction plus "every
  `.cpp` is listed in `CMakeLists.txt`") — all without model weights, in ~0.3 s. There is still no
  CI runner and no lint config, and the accuracy path is exercised only by `./main evaluate`,
  which needs a real dataset and is therefore run by hand.

---

## Changes

See [CHANGELOG.md](CHANGELOG.md) for what this revision fixed, added and measured
(older revisions are in `git log`; releases are tagged, latest `v1.2.0`).

## License

Apache License 2.0 — see [LICENSE](LICENSE). Model weights are third-party works with
their own terms; check the upstream repositories before redistributing them.
