# Configuration, registration and the knobs worth knowing

One table declares every tunable; the file parser, `--set`, range checks and `--dump-config` are all generated from it.

Deep dive split out of [README.md](../README.md), which keeps the short version.
Related: [evaluation.md](evaluation.md) for `recognize.thresh`, [models.md](models.md) for the model knobs

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
| `recognize.thresh` | `0.45` | cosine above which a match is reported. **Not a per-pair number** — see [Choosing the recognition threshold](evaluation.md#choosing-the-recognition-threshold-measured-not-chosen) below: `identify` takes a max over the whole gallery, so the same value means something different at every gallery size. The old default `0.2` admitted essentially every stranger to a 610-person gallery. |
| `display.rfactor` | `6` | display frame = `detector input × rfactor` = 960×720; the letterbox canvas and the coord scale for `draw()`/`putText()`. |
| `detector.conf_thresh` / `detector.nms_thresh` / `detector.keep_top_k` | `0.6` / `0.3` / `750` | detection sensitivity and box suppression. |
| `tracker.iou` / `tracker.dist_ratio` | `0.30` / `0.60` | association gates: a detection is the same track if box IoU ≥ `iou` **or** its centre moved less than `dist_ratio` × box size. Two gates, not one: a face at the frame edge alternates between the whole-face box and the visible-part-only box (measured adjacent-frame IoU **0.37**), so IoU alone splits one face into two tracks that are both drawn forever. |
| `tracker.detect_every` / `tracker.max_lost` / `tracker.min_hits` | `1` / `5` / `2` | `detect_every` below; a confirmed track is carried — box *and* its last real label — for at most `max_lost` frames of pure prediction; a track needs `min_hits` detections before it is shown, so a detection that exists for one frame never reaches the screen. |
| `align.border` | `constant` | how to fill the part of the alignment window that falls outside the frame for edge-clipped faces: `constant` (black) / `reflect` / `replicate`. Does not move the warp; see the measurement above. |
