#!/usr/bin/env python3
"""Baseline, fine-tune and compare Newton C on a labelled video set, end to end.

Stdlib only. Put ATAI_API_KEY and ATAI_API_ENDPOINT in ./.env, build the dataset
with build_dataset.py, then:

    python3 run_fine_tune.py infer --baseline --mode closed     # the "before" (~5-10 min)
    python3 run_fine_tune.py infer --baseline --mode open
    python3 run_fine_tune.py train                              # LoRA fine-tune (~3 h)
    python3 run_fine_tune.py infer --mode closed                # the "after"
    python3 run_fine_tune.py infer --mode open
    python3 run_fine_tune.py score                              # closed-set, base vs fine-tuned
    python3 run_fine_tune.py status | cancel                    # the fine-tune job

Every id (file, job, checkpoint) is saved in ft-state.json the moment it exists,
so a run can be resumed: `train --job <ftj_id>` re-attaches to a running fine-tune,
`infer --job <job_id>` to a running batch job.

Predictions land in predictions_<baseline|finetuned>_<closed|open>.json, aligned
with test_labels.json. Open-ended answers are free text; grade them with an LLM
judge (see SKILL.md), not with `score`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import atai_ft as api


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def predicted_class(answer: str, classes: list[str]) -> str | None:
    """The class a closed-set answer names: the LONGEST class name found in it, so
    nested labels resolve ("BasketballDunk" is BasketballDunk, not Basketball)."""
    a = _norm(answer)
    return max((c for c in classes if _norm(c) in a), key=len, default=None)


def score(preds: list[str], labels: list[str], classes: list[str]) -> dict:
    per = {c: [0, 0] for c in classes}
    for p, y in zip(preds, labels):
        per[y][1] += 1
        per[y][0] += predicted_class(p, classes) == y
    n = min(len(preds), len(labels))
    return {"correct": sum(v[0] for v in per.values()), "n": n, "per_class": per}


# ---------------------------------------------------------------- commands

def cmd_train(a) -> None:
    st = api.load_state()
    job_id = a.job or st.get("ft_job_id")
    if not job_id or a.new:
        train = api.upload_jsonl(a.data / "train.jsonl")
        val = api.upload_jsonl(a.data / "val.jsonl")
        api.save_state(train_file=train, val_file=val)
        print(f"uploaded {train}, {val}")
        job = api.request("POST", "/fine-tuning/jobs",
                          api.fine_tune_body(train, val, a.name, n_epochs=a.n_epochs, seed=a.seed))
        job_id = job["id"]
        api.save_state(ft_job_id=job_id, checkpoint_id=None)
        print(f"fine-tune {job_id} created ({job.get('status')}); budget ~3 h on production")
    job = api.watch(f"/fine-tuning/jobs/{job_id}", api.FT_TERMINAL, interval=60)
    if job.get("status") not in api.FT_SUCCESS:
        sys.exit(f"fine-tune ended {job.get('status')}: {job.get('error')}")
    cks = api.list_all(f"/fine-tuning/jobs/{job_id}/checkpoints")
    for c in cks:
        print(f"  {c['id']}  step={c.get('step')}  eval_loss={(c.get('metrics') or {}).get('eval_loss')}")
    best = api.best_checkpoint(cks)
    if not best:
        sys.exit("job finished but has no checkpoints")
    api.save_state(checkpoint_id=best["id"])
    print(f"best checkpoint {best['id']} -> ft-state.json")


def cmd_infer(a) -> None:
    st = api.load_state()
    model = "baseline" if a.baseline else "finetuned"
    tag = f"{model}_{a.mode}"
    ckpt = None if a.baseline else (a.checkpoint or st.get("checkpoint_id"))
    if not a.baseline and not ckpt:
        sys.exit("no checkpoint: run `train` first, or pass --checkpoint / --baseline")
    job_id = a.job
    if not job_id:
        fid = api.upload_jsonl(a.data / f"test_{a.mode}.jsonl")
        body = api.inference_job_body(fid, f"newton-c-{tag}", ckpt)
        job_id = api.request("POST", "/batch/jobs", body)["id"]
        api.save_state(**{f"{tag}_job_id": job_id})
        print(f"batch job {job_id} ({tag}{', ' + ckpt if ckpt else ''})")
    job = api.watch(f"/batch/jobs/{job_id}", api.BATCH_TERMINAL, interval=20)
    if job.get("status") != "COMPLETED":
        sys.exit(f"batch job ended {job.get('status')}")
    preds = api.predictions(job_id)
    labels = json.loads((a.data / "test_labels.json").read_text())["labels"]
    if len(preds) != len(labels):
        print(f"WARNING: {len(preds)} predictions for {len(labels)} clips")
    out = Path(f"predictions_{tag}.json")
    out.write_text(json.dumps(preds, indent=1))
    print(f"{len(preds)} predictions -> {out}")


def cmd_score(a) -> None:
    meta = json.loads((a.data / "test_labels.json").read_text())
    classes, labels = meta["classes"], meta["labels"]
    runs = {}
    for m in ("baseline", "finetuned"):
        p = Path(f"predictions_{m}_closed.json")
        if p.exists():
            runs[m] = score(json.loads(p.read_text()), labels, classes)
    if not runs:
        sys.exit("no predictions_*_closed.json yet: run `infer --mode closed` first")
    print("| Class | " + " | ".join(runs) + " |\n|---|" + "---|" * len(runs))
    for c in classes:
        print(f"| {c} | " + " | ".join("{}/{}".format(*r["per_class"][c]) for r in runs.values()) + " |")
    for m, r in runs.items():
        print(f"{m}: {r['correct']}/{r['n']} = {r['correct'] / max(r['n'], 1):.1%}")


def cmd_status(a) -> None:
    job_id = a.job or api.load_state().get("ft_job_id")
    if not job_id:
        sys.exit("no fine-tune job in ft-state.json")
    job = api.request("GET", f"/fine-tuning/jobs/{job_id}")
    print(json.dumps({k: job.get(k) for k in ("id", "status", "created_at", "base_model_tag", "error")}, indent=1))
    for e in api.list_all(f"/fine-tuning/jobs/{job_id}/events")[-5:]:
        print(f"  {e.get('event_type') or e.get('type')}: {e.get('message')}")


def cmd_cancel(a) -> None:
    job_id = a.job or api.load_state().get("ft_job_id")
    if not job_id:
        sys.exit("no fine-tune job in ft-state.json")
    print(api.request("POST", f"/fine-tuning/jobs/{job_id}/cancel").get("status"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--name", default=f"newton-c-{time.strftime('%Y%m%d-%H%M')}")
    t.add_argument("--n-epochs", default="auto")
    t.add_argument("--seed", type=int)
    t.add_argument("--job", help="re-attach to an existing ftj_ id")
    t.add_argument("--new", action="store_true", help="start a new job even if ft-state.json has one")

    i = sub.add_parser("infer")
    i.add_argument("--mode", choices=("closed", "open"), required=True)
    i.add_argument("--baseline", action="store_true", help="base model, no checkpoint")
    i.add_argument("--checkpoint", help="ckp_ id (default: ft-state.json)")
    i.add_argument("--job", help="re-attach to an existing batch job id")

    sub.add_parser("score")
    for name in ("status", "cancel"):
        s = sub.add_parser(name)
        s.add_argument("--job")

    a = ap.parse_args()
    {"train": cmd_train, "infer": cmd_infer, "score": cmd_score,
     "status": cmd_status, "cancel": cmd_cancel}[a.cmd](a)


if __name__ == "__main__":
    main()
