"""
Compare Newton-C 2.6 and Newton-Fusion 1.0 on your own inputs.

On simple descriptions the two models give similar answers, so pick a model
by running both on 10-20 inputs that look like your real workload and
reading the answers side by side. This script runs every input through:

  * c26             — Newton-C 2.6 (8B dense), the default
  * fusion          — Newton-Fusion 1.0 (35B MoE, ~3B active), no reasoning
  * fusion+reason   — Newton-Fusion 1.0 with reasoning (slower)

Images and videos are uploaded once and sent by `fil_` uid on
/inference/query. A `.txt` file's contents are inlined in the prompt (text
attachments are refused). Prints each answer with its latency, then a
latency summary. Use --runs N to see how much answers vary between calls.

Usage:
    python compare_models.py --prompt "Is the cap tightened?" clip1.mp4 clip2.mp4
    python compare_models.py --system "Answer PASS or FAIL." --prompt "..." frames/*.jpg
    python compare_models.py --variants c26 fusion --runs 3 photo.png
"""

from __future__ import annotations

import argparse
import mimetypes
import statistics
import sys
import textwrap
from pathlib import Path

from _common import C26, FUSION, banner, make_client, query, upload_file

VARIANTS = {
    "c26": {"model": C26, "reasoning": False},
    "fusion": {"model": FUSION, "reasoning": False},
    "fusion+reason": {"model": FUSION, "reasoning": True},
}


def kind_of(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or ""
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("video/"):
        return "video"
    if mime.startswith("text/"):
        return "text"
    sys.exit(f"Unsupported input type for {path.name} ({mime or 'unknown'}): use an image, a video, or a .txt file.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="Images, videos, or .txt files.")
    parser.add_argument("--prompt", required=True, help="The question asked about every input.")
    parser.add_argument("--system", default="", help="Optional system prompt.")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--runs", type=int, default=1, help="Calls per input and variant.")
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()

    paths = [Path(p).resolve() for p in args.inputs]
    for path in paths:
        if not path.exists():
            sys.exit(f"Input not found: {path}")

    client = make_client()
    latencies: dict[str, list[int]] = {name: [] for name in args.variants}
    failures: dict[str, int] = {name: 0 for name in args.variants}

    for path in paths:
        kind = kind_of(path)
        if kind == "text":
            prompt, file_ids = f"{args.prompt}\n\n{path.read_text()}", []
        else:
            prompt, file_ids = args.prompt, [upload_file(client, path)]
        banner(f"{path.name} ({kind})")
        for name in args.variants:
            for run in range(1, args.runs + 1):
                label = name if args.runs == 1 else f"{name} #{run}"
                try:
                    text, _, elapsed_ms = query(
                        client,
                        user_query=prompt,
                        system_prompt=args.system,
                        file_ids=file_ids,
                        max_new_tokens=args.max_tokens,
                        **VARIANTS[name],
                    )
                except Exception as error:  # noqa: BLE001 — report and keep comparing
                    failures[name] += 1
                    print(f"--- {label}: FAILED — {error}\n")
                    continue
                latencies[name].append(elapsed_ms)
                print(f"--- {label} [{elapsed_ms} ms]")
                print(textwrap.indent(text.strip(), "    ") + "\n")

    banner("Latency summary")
    for name in args.variants:
        values = latencies[name]
        if values:
            print(
                f"{name:<14} median {statistics.median(values):>6.0f} ms   "
                f"min {min(values):>6} ms   max {max(values):>6} ms   "
                f"failures {failures[name]}"
            )
        else:
            print(f"{name:<14} no successful calls   failures {failures[name]}")


if __name__ == "__main__":
    main()
