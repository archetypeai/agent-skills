# OSM role files — the contract

The interface between the two OSM skills. **This skill produces it;**
[`atai-operational-state-monitoring-agent`](../../atai-operational-state-monitoring-agent/SKILL.md)
(Path 2) **consumes it.** Everything below was verified on the dev deployment
(2026-09-30); the "why" column says what breaks if a rule is ignored.

## Layout

```
roles/
├── manifest.json                 the index below; the only file the lifecycle scripts read first
├── zscore_stats.json             per-channel mean / std, from the library recordings only
├── library/<state>__library.csv  ONE file per state; the state is the filename prefix
├── validation/<recording>__seg<N>.csv   one continuous file per recording segment, with `label`
├── test/<recording>__seg<N>.csv         same format; scored once
├── delivery/<recording>__seg<N>.csv     same, WITHOUT `label`
└── delivery_labels/<recording>__seg<N>.csv   `timestamp,label`, row-for-row with delivery/
```

## Every CSV

- **Header:** `timestamp` first, then the channel columns (same names and order in
  every file), then `label` in validation and test only.
- **`timestamp`:** fractional epoch seconds, 3 decimals (`1687797871.590`). Epoch
  milliseconds and ISO 8601 also work on the platform; this contract fixes one.
- **An exact grid:** consecutive rows exactly `1000 / hz` ms apart inside a file,
  except at library piece boundaries (below).
- **Values:** finite, z-scored with `zscore_stats.json`.
- **`label`:** one of `manifest.states`, never blank.

## Rules, and why

| rule | why (what breaks otherwise) |
|---|---|
| **One library file per state**, holding its windows as continuous **pieces** in time order, with real timestamps; forward jumps only **between** pieces, at whole-window boundaries | The platform stores an optimization's whole config (every training example, pretty-printed, file lists twice) in one **1 MiB** Kubernetes ConfigMap: ~636 B per training example, so ~1,500 examples at most. Over it, the job **never dispatches and no error surfaces** — the optimization just stays `running` with trials `pending`. |
| In training files, a window that crosses a jump is **skipped silently** | So pieces should be whole multiples of the window; seam windows are lost, not wrong. At step < window, the overlapping windows that cross a seam are always lost. |
| **Scored files (validation, test) have no jumps at all**: one continuous file per recording segment | A single scored window that crosses a jump fails the **whole** trial or eval: `eval-mode test data must not contain windows a validation node rejected`. Loosening `sample_rate_interval_tolerance` doesn't rescue a real jump (one gap in an N-row window deviates ~N×), and the search space can't set the check to null. |
| **Every state appears in every scored set** (validation, the search subset, test, delivery) | The platform's macro-F1 counts a state with no windows and no predictions as **F1 = 0**, so one missing state caps macro-F1 at (n−1)/n. |
| **The z-score is fitted on the library recordings only** | Fitting on validation, test or delivery leaks them into training. |
| **Delivery files carry no label**; their labels are held back row-for-row | Delivery is scored afterwards against `delivery_labels/`, pairing each prediction with the label at its window's last row. |
| **A recording (or its setting group) is in exactly one role** | Near-identical recordings on both sides of a split inflate every score. |

## `manifest.json`

```json
{
  "contract": "osm-role-files/v1",
  "hz": 200,
  "window": 1024,
  "channels": ["back.x", "back.y", "..."],
  "states": ["drain", "fill", "spin", "wash"],
  "timestamp_format": "epoch seconds, 3 decimals",
  "zscore_stats": "zscore_stats.json",
  "search_validation": ["<recording>", "..."],
  "library": {
    "per_state": 100,
    "files": [
      {"file": "library/drain__library.csv", "state": "drain", "rows": 102400, "windows": 100,
       "pieces": [{"recording": "<recording>", "segment": 0, "start_ms": 1687797871590,
                   "row": 0, "rows": 2048, "windows": 2}]}
    ]
  },
  "validation": {"files": [{"file": "validation/<recording>__seg0.csv", "recording": "<recording>",
                            "segment": 0, "rows": 198028, "windows": 193,
                            "seconds": {"fill": 306.0, "wash": 276.0, "spin": 216.0, "drain": 192.0}}]},
  "test":     {"files": ["... same fields ..."]},
  "delivery": {"files": [{"file": "delivery/<recording>__seg0.csv", "labels": "delivery_labels/<recording>__seg0.csv",
                          "recording": "<recording>", "segment": 0, "rows": 208220, "windows": 203,
                          "seconds": {"...": 0}}]}
}
```

- `window` is the window the **library pieces** were cut in (rows). The window the
  platform uses is chosen later, in the Optimize search space, and should divide it
  (e.g. 1024 → 256 / 512 / 1024), so library seams fall on window boundaries.
- `windows` in validation / test / delivery counts non-overlapping `window`-row windows,
  for orientation only; the platform counts its own.
- `search_validation` is the subset scored during a search; omit it (or list every
  validation recording) to search on all of validation.
- Paths are relative to `roles/`.
