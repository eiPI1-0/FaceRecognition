# Changelog

Notable changes to this repository. The format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses release tags rather
than a strict versioning scheme.

## Unreleased

### Changed

- **README leads with the four claims, then proves them.** A claim table —
  **lightweight** (1.3 MiB of our own code, 8.1 MiB of weights, 5.7 ms per face in a frame,
  112 MB RSS at 4 workers, and the honest note that the 298-library linked closure belongs to
  OpenCV's I/O backends, not to this code), **high concurrency** (p50 74 ms / ≈135 req/s at 4
  workers, ≈193 at 8, 1347 req/s through the result cache, `503 + Retry-After` inside one budget),
  **accurate at a stated operating point** (1.3% false accepts against 610 people at 0.45, 99.7%
  at the old 0.2, 3.0% at 1680, 1:1 EER ≈ 0.18 / 4.1%, held-out photos on the production path),
  **MCP-native** (four read-only JSON-Schema tools over stdio or HTTP, one forward pass per call,
  enrollment withheld from prompt-injectable callers) — followed by a "Properties, and what
  enforces each one" table, a Numbers section, a "What this project deliberately does not do"
  list, and a Contents table. All the existing detail is unchanged underneath.
- **The accuracy claim is scoped, not inflated.** The models supply the accuracy; this repository
  supplies the operating point: ArcFace-template similarity alignment, letterbox preprocessing
  shared by registration and inference, and a threshold chosen from held-out photos rather than
  from a 1:1 separability harness.
- Two claims were corrected against the code while writing it: `/readyz` answers
  `far_1n_estimate` / `far_1n_resolution` (not `self_far_1n`), and an unknown config key is a
  *warning + nearest-name suggestion* in a file but a *hard stop* from `--set`.
- **README leads with the contract, not the file tree.** A "Properties, and what enforces each
  one" table (ten properties, each naming the test, startup check or endpoint behind it), a
  Numbers section split into pipeline cost / HTTP / recognition quality / build-and-checks with
  the dataset-and-gallery caveat on every accuracy figure, a "What this project deliberately does
  not do" list (no TLS terminator, no CUDA promise, no encryption at rest for templates, no
  enrollment over MCP, no accuracy claim beyond the measured corpora, no CI), and a Contents
  table. All the existing detail is unchanged underneath. Two claims were corrected against the
  code while writing it: `/readyz` answers `far_1n_estimate` / `far_1n_resolution` (not
  `self_far_1n`), and an unknown config key is a *warning + nearest-name suggestion* in a file
  but a *hard stop* from `--set`.

---

## v1.3.0 — 2026-11-10 — measured thresholds, one source root, one namespace

### Added

- **`./main evaluate <dataset>` — held-out-set evaluation, so the recognition threshold can be
  measured instead of guessed.** A dataset in the same layout as `data/` (one directory per
  person, ≥ 4 photos): the first N photos become the enrollment side, the next M are held out
  and never enrolled, so 1:N numbers are not read off the training set. Reports genuine /
  impostor / max-other distributions at both 1:1 and 1:N, FAR/FRR/EER at every threshold, and
  writes a CSV. The impostor tail is reported as `unreachable` rather than `0` when it is below
  the sample's resolution — the smallest rate that gallery can certify — because a printed
  `0.000000` reads as proof rather than as a missing measurement (at 610 identities the measured
  1:N FAR bottoms out at 8.2e-4 = 1 probe in 1220; below that nothing is observable).
  1:1 and 1:N get separate columns because the same gallery at the same threshold sits **three
  orders apart**: at t=0.80 the per-pair FAR is 6.7e-7 while the 1:N FAR is 8.2e-4.
- **Startup threshold self-check (`check_recognize_threshold` in `src/api.{hpp,cpp}`).** Every
  shell that holds a gallery now prints what that threshold does *to that gallery*, with no
  external dataset: every cross-person pair inside the gallery is already an impostor pair, and
  the per-pair rate converts to the 1:N rate by the binomial. One estimate, one arithmetic path
  (the pure function `one_n_far_from_pair_rate` in `src/evaluation.{hpp,cpp}` — the relation was
  previously asserted in a comment with no code anywhere computing it). `serve` logs a
  `threshold_check` event (WARN at `recognize.self_far_warn`, default 0.01), prints it in the
  banner, and answers it at `/readyz`; `mcp` logs it in both transports; `identify` stays silent
  because it is one-shot. Motivation: a false accept emits no error, no 503 and no latency
  spike — it looks like success, so a too-permissive threshold can run for months unnoticed.
  Its precision is stated rather than assumed: on 200 LFW identities it said 95% where held-out
  probes measured 84% (gallery-side pairs are not probe-side pairs — the sign of the bias is
  empirical), and the same conversion applied to the measured 1:1 tail gives 1.14% at N=610 vs
  1.31% measured directly. Order-of-magnitude agreement is what the check needs: it must
  distinguish 0.2 (99.7%) from 0.45 (1.3%), not resolve a third digit.
- **`recognize.self_far_warn`** — the warning line for that check (a reporting threshold, not a
  safety one: it decides when to shout, not who gets in).
- **`PipelinePool::fingerprint()` — the live gallery's fingerprint, the same string the
  result-cache key already carried.** `/readyz` now reports it next to the person count, so the
  answer "which gallery was this result computed against?" is checkable by eye instead of by
  reading source.

### Changed

- **`recognize.thresh` default: 0.20 → 0.45.** The old value was never calibrated: it came from
  an augmentation harness that only ever measured 1:1 margins, while `identify`/`verify` are 1:N.
  Measured on LFW through the production path (610 identities × 2 enrolled photos, and a
  disjoint 1680 × 1 — their 1:1 tails agree to ~2% on FAR): the per-pair false-accept rate at 0.20 is
  1.26%, but `identify` takes a max over the gallery, so the probability that a stranger matches
  *somebody* is **99.7% at 610 people and 100% at 1680**. 0.45 brings that to 1.3% / 3.0% for
  +0.9 pt of false rejects; going higher buys FAR linearly and pays rejections superlinearly
  (0.45 → 0.55 halves FAR again and *doubles* FRR). README's "Choosing the recognition threshold"
  has the full table. If 0.2 was set deliberately, set it explicitly — the self-check now says
  out loud what it costs.

- **Source layout: one source root, layers instead of a pile.** `src/` is now the only root (CMake
  adds exactly one include directory), every project include names the layer it reaches into
  (`#include "core/engine.hpp"`), and the order `common → core → app → serve → cli`
  (`src/main.cpp` on top) is **asserted** by a new `ctest` target (`scripts/check_layers.py`): it
  fails on an upward include, on an unqualified project include, and on a `src/**/*.cpp` that is not
  listed in `CMakeLists.txt`. That last failure used to be silent — the source list was written out
  four times (the executable plus three tests, each with its own copy), so a file missing from a
  test's list compiled and simply never linked. The repo root now holds no code: `FaceRecognition.h`,
  the forwarding header left over from an earlier split, is deleted, and `main.cpp` moved to
  `src/main.cpp`. The build is four layer libraries (`facerec_common` / `_core` / `_app` / `_serve`)
  plus a `facerec_httplib` INTERFACE target that carries both httplib macros, so a new target can no
  longer forget one and silently run with a different limit; the tests link the libraries instead of
  repeating file lists. No behaviour change: clean build 32.9 s → 30.1 s on the same machine, 26
  objects with no source compiled twice (four files used to be listed three times each), `ctest` 5/5, `identify`
  self-match 1.000 / cross-identity 0.0226, `serve` answers `/healthz` `/readyz` `/metrics` with 200
  and every stderr line still parses as JSON, `mcp --transport stdio` lists the same four tools, and
  `--dump-config` is byte-identical to `config.example.yml`.

- **`cli` is now one file per subcommand.** `src/cli/cli.cpp` (1266 lines, one 11.55 s translation
  unit) is replaced by `src/cli/args.{hpp,cpp}` (the config plumbing every command shares) and one
  file per command — `identify` `enroll` `token` `serve` `mcp` `compare` `verify` `evaluate` — plus
  `src/cli/demo.{hpp,cpp}` for the live window loop, which also absorbed `run_pipeline` and
  `recognize_frame` from `src/main.cpp` (that file is now 97 lines: usage + dispatch). The `cmd_*`
  names and signatures still live in `src/cli/cli.hpp`, so the dispatch table did not change, and no
  command behaviour did: every `--help` is byte-identical apart from include-block changes. Measured:
  recompiling after editing one command 11.55 s → **1.4-1.9 s**, largest cli translation unit 8.67 s,
  clean build 32.9 → **29.8 s** (`-j16`); the cli layer's total compile CPU rises to 37.2 s because
  each file parses its own headers, which parallelism hides. `print_usage` deliberately stays in
  `src/main.cpp`: the command list *is* the dispatch table there, and splitting them guarantees one
  of the two eventually lies.

- **The recognition core is now in `namespace facerec`**, and `src/serve/auth.hpp` no longer includes
  `<httplib.h>` — it forward-declares `httplib::Server` at **global** scope (inside `facerec` it would
  compile and fail only at link time), and `auth.cpp` includes the header, since it defines
  `install_access_control` and needs the complete type. Before, the library was half named / half
  global: everything from `api` up was `facerec::`, while `Detector`, `FeatureExtractor`,
  `Similarity`, `Face`, `PriorBox`, `FaceTracker`, `nms`, `letterbox` and the alignment helpers sat in
  the global namespace. Consumers needed almost no edits — `app/` and `serve/` were already inside
  `namespace facerec`; only the global-scope files (`cli/*.cpp`, `tests/*.cpp`) gained a
  using-directive. Measured effect of the httplib change: `src/cli/token.cpp`, which only uses the
  token table, compiles in **0.92 s instead of 7.98 s**, and `<httplib.h>` is now included by exactly
  four files, all under `src/serve/`, because two `#include <httplib.h>` lines in `cli/serve.cpp` and
  `cli/mcp.cpp` turned out to be vestigial (neither file names an `httplib::` symbol). That alone took
  those two translation units from 8.7 s to **1.4 s** each and the clean build from 29.0 s to
  **22.7 s**. Removing a transitive include bit once: `token.cpp` had been getting `<fstream>`
  through httplib.

### Fixed

- The `recognize.thresh` knob comment asserted the old default "is not a per-pair number… pick
  it with `./main evaluate`" while quoting a default (0.2) that was itself ~100% permissive at
  the very gallery size the comment named. The comment was right about the mechanism and wrong
  about its own value.

### Added

- **Admission control (edge shedding) with one capacity-wait budget across both admission
  points.** Waiting for capacity happens twice — in cpp-httplib's task queue (accept thread →
  worker) and in `PipelinePool` (handler → pipeline) — and only the second one had a bound, so the
  worst case was *queue ∞ + pool 5 s*. `src/admission.hpp` carries the wait as one transmittable
  quantity (per worker thread, no locks): the task-queue wrapper stamps it when a worker picks up a
  connection, the pre-routing hook settles it onto the request, and the pool gets
  `deadline − already_waited`. Past the deadline a request is refused before any pipeline is
  touched: 503 with `code:"queue_expired"` and `Retry-After: 1` (a 503 without `Retry-After` makes
  the client re-push the load it just shed; the pool-stage 503 now carries it too).
  Measured on an open-loop flood — 140 req/s against 2 workers (capacity ≈ 98 req/s), 6 s:
  without shedding p50 grows **linearly** 232 → 2338 ms while nothing is refused; with a 150 ms
  deadline p50 stays **flat at ~161 ms** (max 173), and throughput is unchanged at 98.7/s — the
  excess gets fast refusals instead of unbounded latency. Continuity verified per request:
  `queued_ms + waited_ms` equals the deadline (48+52, 30+70, 66+34).
- **`--queue-max-pending`** (default 0 = unbounded): hard cap on connections waiting in front of
  the workers, using cpp-httplib's own `ThreadPool(n, max_queued_requests)` rather than a new queue
  (measured: `--workers 2 --queue-max-pending 4`, 30 simultaneous arrivals → 6 served, 24
  connections closed, `facerec_queue_dropped_total 24`). It answers no HTTP status because the drop
  happens before the request is parsed, and it also drops probes (measured 5/5 `/healthz` dropped
  in that configuration) — so it is a resource backstop, off by default.
- **Probes and `/metrics` are exempt from shedding** (verified 200 during the flood): killing a
  probe turns an overload into a restart, and killing `/metrics` removes the data needed to see it.
- `facerec_queue_expired_total` / `facerec_queue_dropped_total` (the two shed paths), and the
  `pool_busy` event now reports `queued_ms` / `budget_ms` / `deadline_ms` so "used the budget in the
  queue" and "queue instant, pool empty" are distinguishable. One-time `queue_shed_start` warn
  marks the onset; subsequent sheds are counters, not log lines.
- `tests/admission.cpp` (3rd ctest): the budget arithmetic, the keep-alive accounting rule (second
  request on a connection waits 0 — httplib queues **connections**, measured 12 requests/1
  connection → `queue_wait_count` +1), and that every declared metric actually appears in
  `to_prometheus()` (that list is hand-written and forgetting a line fails silently). Verified to go
  red when the floor or a metric line is removed.
- **`deploy/` — three deployment artifacts, each with one job.**
  - `deploy/Dockerfile`: multi-stage build (openSUSE Leap 16). The build stage carries
    `opencv-devel`; the runtime stage carries only `libopencv413` plus the binary, runs as a
    non-root user, and keeps `models/`, `data/` and the enrollment index on volumes. Because a
    non-loopback bind without a token table refuses to start, the image ships
    `auth.tokens_file: /etc/facerec/tokens` and fails loudly when it is not mounted.
  - `deploy/systemd/facerec@.service`: a template unit, so `systemctl start facerec@{1..4}` is
    horizontal scaling. `--reuseport` lets all instances bind one port and the kernel distributes
    connections — verified here with two processes, two `ss -lnt` lines, and 60 requests split
    21 / 39 across them.
  - `deploy/nginx/facerec.conf`: TLS termination, `/metrics` restricted to private ranges, and a
    proxy to a loopback-bound instance. nginx does no load balancing, because the kernel already
    does; it only does edge work.

### Added

- **Operational log levels: `log.level` (`off|error|warn|info|debug`, default `warn`).**
  `src/log.{hpp,cpp}` is now the single structured-log writer: one JSON-lines stream, one
  destination (`log.sink`, renamed from `log.audit`), one lock, so audit and operational lines
  interleave in time order instead of living in two places. The two kinds keep different
  guarantees — `audit(...)` is never filtered by level (only `log.sink: off` silences it),
  `event(...)` is. Measured: at `log.level: off` a 401 still writes its audit line with peer and
  `presented`, while every event line disappears.
  Events: `pool_busy` (warn), `lease_wait` (info), `pipeline_error` (error), `unknown_tool` (MCP).
  Until now a 503 left no server-side trace at all, and an exception's text reached only the
  response body.
- **`--http-threads`** (serve and MCP HTTP): the transport thread count. Default equals
  `--workers`, which means a pipeline is always free when a handler asks it — so the pool is not
  the admission point and 503/`pool_busy` cannot fire. Raising it makes them fire for real
  (measured: `--workers 2 --http-threads 16 --queue-timeout-ms 1`, 32 concurrent → 30×503 with
  30 `pool_busy` lines), at the cost of concurrent decodes and CPU oversubscription.
- **`--queue-timeout-ms`** (serve and MCP HTTP): replaces three hard-coded lease timeouts
  (2000 / 5000 / 5000 ms) with one backpressure knob.
- Key suggestions now also match on **group**: `log.audit` (renamed) previously matched nothing
  and dumped the whole table; it now answers `did you mean log.sink log.level`. Edit distance
  cannot bridge a rename, group membership can.
- **CLI `compare <img-a> <img-b>` and `verify <img> --name <person>`** — the same comparison
  the HTTP `/v1/compare` and MCP `compare_faces` paths expose, without needing a server.
  Both print JSON on stdout and share `Pipeline::embed_image` + the cosine formula, so the
  three interfaces cannot drift apart.
  `verify` exits **non-zero when the answer is "no"** (0 verified, 1 not verified, 2 cannot
  answer — unreadable image or no face, 3 pipeline error): an authentication-style decision
  that always exits 0 turns every script that forgets to parse the JSON into a silent pass,
  and a silent pass is the worst failure an auth check has. The verdict and the reason are
  always in the JSON (`verified`, `cosine`, `matched`, `threshold`) regardless of exit code.
- **Fixed a description that was a lie to the model**: `compare_faces` told the LLM it compared
  "the largest face in two images". NMS sorts by **score**, so index 0 is the most confident
  face, which is not always the largest. Kept the code and fixed the wording, because score order
  *is* the order `identify` reports faces in — so "the face compare compares" and "identify's
  `faces[0]`" are provably the same face. Switching to area order would have made two interfaces
  silently compare different faces, which is the cross-interface drift this project keeps hitting.
- **`facerec_queue_wait_ms` histogram** — time a request spent in the HTTP task queue before
  a worker picked it up (`enqueue → execute`), complementing `request_ms` (handler-internal
  only). A new `src/timed_queue.hpp` wraps httplib's `ThreadPool` and records the delta on
  every request, so operators can finally tell "the client waited 800 ms" from "the model took
  800 ms" instead of guessing from `request_ms` alone.

### Changed

- **`--queue-timeout-ms` changed meaning: it is now the end-to-end budget for waiting for
  capacity, not the pipeline-wait timeout.** Any value `N > 0` behaves tighter than before (the
  queue stage used to be unbounded); `0` keeps its historical meaning of *zero tolerance*, and
  `-1` is the new explicit "never shed". The 0 → "off" shortcut was rejected on purpose: silently
  flipping the meaning of an existing value is worse than adding a sentinel.
- **`facerec_requests_total` / `facerec_request_errors_total` are gone**; both were wrong in a way
  that mattered. Their HELP text claimed "HTTP/MCP requests served", but they were incremented at
  three points inside `/v1/identify` only: `400 bad_request` and `503 server busy` were invisible,
  and the MCP transport never incremented them at all. They are replaced by one labeled counter,
  `facerec_http_requests_total{path,status}`, incremented in the single access-log hook in
  `src/auth.cpp` — so every route and every status code (401 / 400 / 413 / 503 / 500) is counted,
  and adding a route cannot under-count it. `quiet` and probe log suppression still silence logs,
  never metrics. `path` is a fixed whitelist; unknown paths collapse into `path="other"`, so a
  client looping over `/random-N` cannot inflate label cardinality.
  Per-process counters are unchanged, and now documented: with `--reuseport` one scrape reaches
  one arbitrary instance, so aggregate per instance.

### Fixed

- README claimed "There is **no authentication** and no TLS", contradicting the implemented
  binding constraint (bearer tokens via `auth.tokens_file`, refusal to start on a non-loopback
  host without a token table, audit line on every 401). Corrected, and TLS is documented as the
  reverse proxy's job.

## v1.2.0 — 2026-11-07 — production shells (CLI / HTTP / MCP)

### Added

- **`./main mcp`** — Model Context Protocol server (JSON-RPC 2.0) over stdio or `POST /mcp`,
  linking the library directly so a tool call costs one forward pass instead of a cold
  `readNet`. Four read-only tools with JSON Schemas: `identify_faces`, `compare_faces`,
  `list_people`, `get_face_crop`. **Enrollment is deliberately not exposed**: writing a
  biometric template is non-retractable and must not be reachable from a prompt-injectable
  caller — `tools/call` to `enroll_person` answers `-32602 unknown tool`. Tool results carry
  a human-readable summary *and* machine-parseable JSON, and `identify_faces` states in the
  description the model actually reads that `"matched": null` means unknown, so candidate
  scores are never turned into an invented identity.
- **`./main identify --image <path>`** — one image in, JSON out, no window, no state.
  `bbox`/`landmarks` are in original-image pixels (the letterbox mapping is inverted), so a
  caller never has to know about `rfactor`. "No face" and "below threshold" are answers, not
  errors: `"faces": []` / `"matched": null` with exit status 0. Request-level `--top-k`,
  `--min-score`, `--include landmarks|crops|embedding` and `--detector-input WxH`, so tuning
  needs no restart and no recompile.
- **`./main enroll --name <person> --dry-run <photos...>`** — validates a registration set
  (readable? face found? aligned?) without writing anything. `--yes` prints the plan; the
  persistent index is still a later phase.
- **`./main serve`** — HTTP service on vendored cpp-httplib: `POST /v1/identify` (raw image
  body *or* JSON with `image_b64`), `/v1/embed`, `/v1/compare`, `GET /v1/people`,
  `/healthz`, `/readyz`. Backpressure is the worker pool itself (one `Pipeline` per worker,
  `503` when none is free), body and pixel limits are enforced before decoding, and
  SIGINT/SIGTERM finishes in-flight requests then exits 0.
- **A layered source tree.** The inference library moved to `src/engine.{hpp,cpp}`
  (`FaceRecognition.h` is now a forwarding header, so old includes still build), and the
  request-level API — `Pipeline`, `IdentifyResult`, JSON — lives in `src/api.{hpp,cpp}`.
  CLI, HTTP and MCP share that one implementation, so their output cannot drift apart.
  `src/json_util`, `src/encoding` and `src/pipeline_pool` hold what the shells used to
  duplicate.
- `./main demo <source>` — the original interactive loop, unchanged; `./main 0` still works
  and is reported as a legacy form on stderr.
- **`GET /metrics`** — Prometheus text exposition: request / error / faces-detected /
  known-vs-unknown-match / cache hit-miss-eviction counters, plus histograms for `decode_ms`,
  `detect_ms`, `embed_ms`, `request_ms` and `match_score`. The score histogram is what makes a
  `recognize.thresh` change reviewable: you see the distribution move instead of guessing.
- **Persistent gallery index** (`paths.gallery_index`, default `./gallery.idx`, empty disables):
  fused enrollment features are stored next to the run and reloaded at startup — no photo is
  decoded or embedded when the index is valid. It carries a fingerprint of the model files and
  of every registration photo (path, size, mtime); any change rebuilds it, and a corrupt index
  rebuilds with the reason printed rather than failing.
- **`--cv-threads N`** (default 1): the server parallelizes across pipeline instances, so
  OpenCV's intra-request thread pool is off unless asked for. `--workers` defaults to the CPU count.
- **A missing weight now names its own fix**: `check_model_files()` adds a copy-pasteable hint
  (current working directory plus `--set paths.root=build`) to the existing "which path, what to
  do" message. Path *resolution* semantics are unchanged — relative `paths.*` still resolve
  against `paths.root` (default cwd), anchoring stays opt-in.

### Fixed

- **stdio MCP framing**: JSON Schemas were written as readable multi-line raw strings, and
  a raw newline inside one message breaks the newline-delimited transport — the server sent
  33 lines for 6 requests and clients failed mid-object. Schemas are now emitted on one
  line. (The HTTP transport hides this bug, so stdio is what has to be tested.)
- `httplib` caps `application/x-www-form-urlencoded` bodies at **8192 bytes** independently
  of `set_payload_max_length`, and `curl --data-binary` defaults to exactly that content
  type — so posting a 900 KB image returned 413 regardless of `--max-request-mb`. The cap
  is raised to 64 MB at compile time; the real limit stays the runtime
  `--max-request-mb` (verified: 2 MB body with a 1 MB limit still returns 413).

- **Request-level detector size** (`/v1/identify` `detector_input`, CLI `--detector-input`):
  detect a group photo or distant face with 640×480 instead of 160×120 to trade CPU for
  recall. The letterbox canvas stays fixed, so the alignment crop always comes from the
  resolution the gallery was built at — the knob changes how many pixels reach the
  detector, not the embedding's domain. Cost is roughly quadratic (1.6 / 11.4 / 35 / 87 ms
  at 160×120 / 320×240 / 640×480 / 960×720 on one image). Sizes must be multiples of 8
  within [32,1024], else 400. Up to 8 distinct sizes are kept per worker (LRU), because an
  unbounded set of request-chosen sizes would be an unbounded memory path.
- **Result cache** (`src/result_cache`, `--cache-entries`, default 256): keyed by
  `sha256(image bytes) + request params + model/gallery fingerprint`, computed *before*
  decoding so a hit skips `imdecode`, detection and embedding. Measured 205 → 1347 req/s
  and p50 72.8 → 4.4 ms when 16 clients repeat one image. `X-Cache: HIT|MISS` is returned
  and `/readyz` reports hit/miss counters. `src/sha256` is self-contained (no OpenSSL).
  The key covers every option that changes the output rather than caching a canonical
  result and re-deriving crops/embeddings — half the code and no recompute edge cases, at
  the cost of a miss when the same image is asked for a different `top_k`.

### Changed

- **`Detector::detect(const cv::Mat&) const -> std::vector<Face>`** returns by value instead
  of handing back an internal reference, and no longer keeps `dets`/`output_blobs` as
  cross-call state. `FeatureExtractor::extract*` and
  `FeatureDatabase::select_max_similarity` are `const`. This is what makes "one pipeline
  instance per worker thread" safe; the old interface could not be used concurrently.
- `FeatureDatabase` gained `select_top_k(feature, k)`, so a caller can report ranked
  candidates instead of only the single best match.
- Server default `cv::setNumThreads(1)`: measured p50 latency is unchanged (10.6 ms) while
  throughput rises 26–32% at 4–8 workers — letting each worker's OpenCV also spawn 16
  threads only makes them compete for cores.
- `features_from_img` takes `const cv::Mat&`; it always cloned internally anyway.

### Fixed

- `./main --dump-config`, `./main <source>` and every other pre-existing invocation keep
  working after the subcommand split (non-command first argument falls through to `demo`).
- **`nan` scores and phantom faces under serving** — the same image sometimes returned an extra
  candidate, sometimes two, with `det_score: nan` or a forged 0.999 score. Two independent causes,
  both fixed:
  1. *Detection*: the pinned fused export computes its **last level** (6 of 1076 anchors, stride 64)
     unreliably under OpenCV 4.13 — buffer-state dependent per `Net` instance, stable within one, and
     *not* a race: a process-wide forward mutex, `cv::setNumThreads(1)` and `net.enableFusion(false)`
     all still reproduce it (12–24 of 48 forwards affected). `decode` then *accepted* those anchors,
     because the score gate was written `score < ignore_score`, which is false for NaN.
     `PriorBox::decode` now rejects non-finite scores (`!(score >= ignore_score)`) and any candidate
     box that cannot be a face in this frame (finite, 2 px ≤ w,h ≤ 8× the frame, ≥1% of its area
     inside). Required because `iou_score` is clamped to [0,1], so garbage conf can forge a high
     score while garbage `loc` yields a degenerate box. Edge-clipped faces still pass: only real
     overlap is required. Before: 4 phantom candidates per one-face photo, every run; after: 24
     concurrent identical requests give exactly **1** distinct result, and real detections are
     bit-equal to before. The tracker, `align_face`, OpenCV's thread pool and buffer reuse were each
     ruled out with measurements (PLAN.md §P5) — the earlier "the tracker is thread-unsafe" and
     "buffer reuse explains it" diagnoses in this session's notes were wrong.
  2. *Transport*: a stable ~1.04 s tail on some requests was cpp-httplib's **listen backlog of 5**
     (`httplib.h:117`; `Server::listen()` takes no backlog argument, so the macro is the only lever).
     Above 5 simultaneous connects the kernel drops SYNs and the client retransmits after 1 s —
     invisible in server-side timings (handler 16 ms, `curl`'s own `connect` 1015 ms). CMake now
     defines `CPPHTTPLIB_LISTEN_BACKLOG=1024`. Measured (Debug, cache off): workers=4 max
     1078 → 67 ms, workers=8 1081 → 126 ms, and 5/64 slow requests → 0.
- **`enroll --yes` reported `"status":"ok"` while writing nothing** — the write path was a stub,
  which on a *biometric enrollment* API is the worst possible failure mode (the caller believes a
  person exists; every later query answers `matched: null`). It now copies the photos that contain
  a usable face into `<data_dir>/<person>/`: `--name` must be one safe path component (`../evil`,
  `a/b`, leading `-` rejected; whitespace normalized exactly like the gallery index does), an
  existing filename is never clobbered (`_1` suffix), unreadable/faceless photos are listed in
  `"skipped"`, and a run that writes nothing fails instead of reporting success. `--dry-run` now
  prints `would_write_to`, and both modes emit valid JSON (a misplaced bracket made the dry-run
  output unparseable — caught by feeding it through `json.load`).
- **Bearer authentication with an impossible-to-forget failure mode.** `./main token --name <id>`
  writes a random 64-hex token's `name:sha256:<hex>` line into a `0600` file (the file stores no
  usable credential) and prints the token once; `auth.tokens_file` makes `serve` and
  `mcp --transport http` require `Authorization: Bearer …` on everything except `/healthz` and
  `/readyz`, compared in constant time. A **non-loopback `--host` without a token table refuses to
  start** (with the fix in the message): the dangerous combination is unrepresentable rather than
  merely undocumented, and it is checked before a single pipeline is built.
- **Structured audit log** (`log.audit`: `stderr` by default, a file path, or `off`): one JSON
  object per request with `svc/method/path/status/ms/caller/bytes`, `auth:"denied"` plus
  `peer`/`presented` on 401, and a `startup` line carrying models, gallery fingerprint and auth
  mode. Query lines carry **no person id, matched name, embedding or image bytes** (PLAN.md D1):
  an audit trail that links requests to identities is a second biometric database. `--quiet`
  silences successes only — a refusal is a security event and is always logged. `enroll` logs the
  person it wrote, because a template write is the irreversible act.
- **The gallery index is created `0600`** and a wider mode now warns once with the exact fix:
  `build/gallery.idx` is plaintext 512-d templates plus names.
- **MCP `list_people` reported directories, not enrolled people**: an empty `data/test/` was
  announced as `"3 person(s) enrolled: … test"` while the JSON beside it listed 2 — and a person
  with no usable photos can never match, so a caller would have inferred an identity that does not
  exist. The summary now comes from the gallery itself and appends
  `[enrolled but no usable photos: <dir>]` when a directory contributed nothing.
- **A person directory whose name contained whitespace had two different ids**: `data/John Doe/`
  was `John Doe` after a cold start and `John_Doe` after the index was reloaded (the line-based
  index normalized on write only). The id and provenance are normalized at build time now, through
  the one shared rule (`normalize_entry_id`, declared in `gallery.hpp`).
- **Raw image uploads were rejected with 413 for anything over 8 KB** whenever the client sent
  `Content-Type: application/x-www-form-urlencoded` — which is exactly what
  `curl --data-binary @photo.jpg` sends by default, so the most natural invocation of the raw
  endpoint failed with a body-less 413 while the JSON form worked. Cause: cpp-httplib caps
  form-encoded bodies at 8 KB, and whether that cap fires at all depends on the number of
  translation units that include `httplib.h` (same request, same header: 200 with two, 413 with
  three — measured with the same binary inputs). Rather than inherit a library quirk, the build
  pins `CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH` far above the payload limit, so
  `--max-request-mb` is the only size gate and `Content-Type` cannot change what the API means.
- A bare first argument that is not a path, URL, camera index or existing file (`./main compaer`)
  is reported as an unknown subcommand instead of being opened as a video source.
- `CMakeLists.txt` sets a default `CMAKE_BUILD_TYPE=Release` when none is given: an unconfigured
  build silently produced `-O0` binaries, and every perf number taken from them was off by ~3×.

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
