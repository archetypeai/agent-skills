#!/usr/bin/env python3
"""Fine-tune an Omega head on labelled sensor CSVs, run it, and score it.

Stdlib only. Put ATAI_API_KEY and ATAI_API_ENDPOINT in ./.env, lay the data out as

    <data>/train/<class>/*.csv    training files       (fine-tune `training_files`)
    <data>/val/<class>/*.csv      validation files     (fine-tune `validation_files`)
    <data>/test/<class>/*.csv     held-out files       (batch inference, scored)

each CSV a `timestamp` column plus numeric channels, one class per file. Then:

    python3 run_omega_ft.py train --data data        # head fine-tune (~1-5 min)
    python3 run_omega_ft.py infer --data data        # machine-state-classification with the head
    python3 run_omega_ft.py score --data data        # accuracy, confusion, per-class P/R/F1
    python3 run_omega_ft.py status | cancel          # the fine-tune job

Every id (files, jobs, checkpoint) is saved in omega-ft-state.json as soon as it
exists; `train --job <ftj_id>` and `infer --job <job_id>` re-attach to a running job.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import omega_ft as api


def upload_split(data: Path, split: str) -> list[tuple[str, str]]:
    """Upload a split's CSVs (content-hashed names); return (file_id, class) pairs."""
    pairs = [(api.upload_csv(p), c) for p, c in api.class_files(data, split)]
    st = api.load_state()
    api.save_state(classes_by_file={**st.get("classes_by_file", {}), **{f: c for f, c in pairs}})
    print(f"{split}: {len(pairs)} files uploaded ({', '.join(sorted({c for _, c in pairs}))})")
    return pairs


def cmd_train(a) -> None:
    st = api.load_state()
    job_id = a.job or (None if a.new else st.get("ft_job_id"))
    if not job_id:
        train, val = upload_split(a.data, a.train_split), upload_split(a.data, a.val_split)
        columns = api.data_columns(api.class_files(a.data, a.train_split)[0][0])
        body = api.head_body(train, val, a.name, columns, window=a.window, step=a.step,
                             n_epochs=a.epochs, seed=a.seed)
        job = api.request("POST", "/fine-tuning/jobs", body)
        job_id = job["id"]
        api.save_state(ft_job_id=job_id, checkpoint_id=None, window=a.window, columns=columns,
                       val_files=val, n_classes=len({c for _, c in train}))
        print(f"fine-tune {job_id} ({job.get('status')}), base model "
              f"{job.get('base_model_tag')} ({job.get('base_model_id')})")
    job = api.watch(f"/fine-tuning/jobs/{job_id}", api.FT_TERMINAL, interval=20)
    if job.get("status") not in api.FT_SUCCESS:
        sys.exit(f"fine-tune ended {job.get('status')}; the reason is in the events above")
    cks = api.list_all(f"/fine-tuning/jobs/{job_id}/checkpoints")
    for c in sorted(cks, key=lambda c: c.get("step", 0)):
        print(f"  {c['id']}  step={c.get('step')}  metrics={json.dumps(c.get('metrics'))}")
    best = api.best_checkpoint(cks)
    if not best:
        sys.exit("job finished but has no checkpoints")
    api.save_state(checkpoint_id=best["id"])
    print(f"best checkpoint {best['id']} (step {best.get('step')}) -> {api.STATE}")
    if api.collapsed(cks, api.load_state().get("n_classes", 2)):
        print("WARNING: the only checkpoint scores like a one-class predictor: the head did not "
              "learn. Usually too few training windows; add data or lower --step.")


def cmd_infer(a) -> None:
    st = api.load_state()
    ckpt = a.checkpoint or st.get("checkpoint_id")
    if not ckpt:
        sys.exit("no checkpoint: run `train` first, or pass --checkpoint")
    job_id = a.job
    if not job_id:
        test = upload_split(a.data, a.test_split)
        val = [tuple(x) for x in st.get("val_files") or upload_split(a.data, a.val_split)]
        columns = st.get("columns") or api.data_columns(api.class_files(a.data, a.test_split)[0][0])
        window = st.get("window") or a.window  # must equal the fine-tune's window
        body = api.inference_body([f for f, _ in test], val, ckpt, a.name, columns, window=window,
                                  step=a.step)
        job_id = api.request("POST", "/batch/jobs", body)["id"]
        api.save_state(infer_job_id=job_id)
        print(f"batch job {job_id}: {len(test)} files, head {ckpt}, window {window}")
    job = api.watch(f"/batch/jobs/{job_id}", api.BATCH_TERMINAL, interval=20)
    if job.get("status") != "COMPLETED":
        sys.exit(f"batch job ended {job.get('status')}")
    events = api.list_all(f"/batch/jobs/{job_id}/events")
    if not any("fine-tuned" in (e.get("message") or "") for e in events):
        print("WARNING: no 'Loading fine-tuned configuration' event: the head may not have been used")
    print(f"done. Next: python3 run_omega_ft.py score --data {a.data}")


def cmd_score(a) -> None:
    st = api.load_state()
    job_id = a.job or st.get("infer_job_id")
    if not job_id:
        sys.exit("no inference job: run `infer` first, or pass --job")
    classes = st.get("classes_by_file", {})
    pairs = []
    for input_id, text in api.outputs(job_id):
        actual = classes.get(input_id)
        if actual is None:
            print(f"  [skip] {input_id}: class unknown")
            continue
        pairs += [(actual, p) for p in api.predictions(text)]
    r = api.score(pairs)
    print(f"job {job_id}: accuracy {r['correct']}/{r['n']} = {r['accuracy']:.1%}, macro-F1 {r['macro_f1']:.3f}")
    names = sorted(r["confusion"])
    print("confusion (rows = actual):  " + "  ".join(f"{n[:10]:>10}" for n in names))
    for n in names:
        print(f"  {n[:24]:>24}  " + "  ".join(f"{r['confusion'][n][m]:>10}" for m in names))
    for n, v in r["per_class"].items():
        print(f"  {n}: precision {v['precision']:.1%}  recall {v['recall']:.1%}  F1 {v['f1']:.3f}")


def cmd_status(a) -> None:
    job_id = a.job or api.load_state().get("ft_job_id")
    if not job_id:
        sys.exit("no fine-tune job in the state file")
    job = api.request("GET", f"/fine-tuning/jobs/{job_id}")
    print(json.dumps({k: job.get(k) for k in ("id", "status", "base_model_tag", "base_model_id")}, indent=1))


def cmd_cancel(a) -> None:
    job_id = a.job or api.load_state().get("ft_job_id")
    if not job_id:
        sys.exit("no fine-tune job in the state file")
    print(api.request("POST", f"/fine-tuning/jobs/{job_id}/cancel").get("status"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    stamp = time.strftime("%Y%m%d-%H%M")

    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--train-split", default="train")
    t.add_argument("--val-split", default="val")
    t.add_argument("--window", type=int, default=128, help="rows per window (16-1024)")
    t.add_argument("--step", type=int, default=4, help="rows between training windows")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--name", default=f"omega-head-{stamp}")
    t.add_argument("--job", help="re-attach to an existing ftj_ id")
    t.add_argument("--new", action="store_true", help="start a new job even if the state has one")

    i = sub.add_parser("infer")
    i.add_argument("--data", type=Path, required=True)
    i.add_argument("--test-split", default="test")
    i.add_argument("--val-split", default="val")
    i.add_argument("--checkpoint", help="ckp_ id (default: from the state file)")
    i.add_argument("--window", type=int, default=128, help="only if the state file has none")
    i.add_argument("--step", type=int, default=1, help="rows between scored windows")
    i.add_argument("--name", default=f"omega-head-inference-{stamp}")
    i.add_argument("--job", help="re-attach to an existing batch job id")

    s = sub.add_parser("score")
    s.add_argument("--data", type=Path, default=Path("data"))
    s.add_argument("--job")
    for n in ("status", "cancel"):
        sub.add_parser(n).add_argument("--job")

    a = ap.parse_args()
    {"train": cmd_train, "infer": cmd_infer, "score": cmd_score,
     "status": cmd_status, "cancel": cmd_cancel}[a.cmd](a)


if __name__ == "__main__":
    main()
