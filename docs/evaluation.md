# Measuring accuracy, and choosing a threshold

Why a per-pair false-accept rate and a gallery false-accept rate differ by two orders of magnitude, and how both are measured through the production path.

Deep dive split out of [README.md](../README.md), which keeps the short version.
Related: [configuration.md](configuration.md) (`recognize.thresh`), [limitations.md](limitations.md)

---

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
