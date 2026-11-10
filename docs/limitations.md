# Limitations, in full

The complete list, including a defect in the pinned detector export and how it is contained.

Deep dive split out of [README.md](../README.md), which keeps the short version.
Related: [evaluation.md](evaluation.md), [models.md](models.md), [architecture.md](architecture.md)

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
  ([Choosing the recognition threshold](evaluation.md#choosing-the-recognition-threshold-measured-not-chosen)).
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
  `.cpp` is listed in `CMakeLists.txt`") — all without model weights, in ~0.7 s. There is still no
  CI runner and no lint config, and the accuracy path is exercised only by `./main evaluate`,
  which needs a real dataset and is therefore run by hand.

---
