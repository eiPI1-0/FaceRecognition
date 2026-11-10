# Serving over HTTP: throughput, capacity, auth, deployment

**The capacity story is the interesting part.** This server's unit of concurrency is the *connection*, and it answers overload with `503 + Retry-After` inside one end-to-end budget instead of a queue that grows until every client times out.

Deep dive split out of [README.md](../README.md), which keeps the short version.
Related: the [architecture](architecture.md) of the serving layer, [configuration](configuration.md) for every key, [evaluation](evaluation.md) for the threshold

---

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
| `GET /healthz` / `GET /readyz` | liveness / readiness. `/readyz` also reports cache hits/misses and the threshold block — `far_1n_estimate` is `null` when this gallery cannot measure the tail, with `far_1n_resolution` and the `above`/`pairs` counts next to it ([evaluation.md](evaluation.md)) |

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

Re-measured on the current build with `scripts/stress.py` (keep-alive clients, one 48 KB JPEG,
`--cache-entries 0`): **158 req/s** at 4 workers (p50 25 ms) and **222 req/s** at 8 (p50 34 ms);
with the result cache on and the same image repeated, 3.8–5.1k req/s. The table above is the
stricter case — a 1.3 MB image and a fresh connection per request — so the headline figures are
the conservative pair. One caveat that bites people running their own numbers: with keep-alive and
more clients than workers, a worker holds its connection for `keep_alive_max_count` requests, so
the extra clients wait at the **connection** level and p99 jumps to seconds while p50 stays flat.

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
