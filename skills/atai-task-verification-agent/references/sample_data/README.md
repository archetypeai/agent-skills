# Sample data

Real artifacts from real runs against `tva` on dev, 2026-08-11. Committed so the
output schema, the scorer and the failure signature can all be inspected with **no
API key, no network and no GPU**:

```sh
python3 ../run_tva_agent.py --score tva-output-1_pass_2_pass_3_pass_A.jsonl
python3 ../run_tva_agent.py --score tva-output-1_pass_2_pass_3_fail_A-mnt2048-EMPTY.jsonl
```

## Files

| File | What it is |
|---|---|
| `oring_sop.txt` | A real 3-step SOP, one step per line — the format `PrepareSOPNode` expects. |
| `tva-output-1_pass_2_pass_3_pass_A.jsonl` | A clip where all three steps were performed correctly. Three `PASSED` verdicts with timestamps and reasons. 678 bytes. |
| `tva-output-1_pass_2_pass_3_fail_A-mnt2048-EMPTY.jsonl` | **The failure worth recognising.** Same SOP, a clip where step 3 was skipped, `max_new_tokens: 2048`. 44 bytes, `results: []`, reported as `job.completed` with no ERROR row. |

## Why the empty one is here

It is not a broken file — it is what the platform returns when the output budget is
consumed inside the model's `<think>` block before any verdict is emitted:

```
WARN  parser.running  TaskVerificationResultsParserNode: generation ended inside the
                      model's reasoning block (no `</think>`), so it never produced an
                      answer; dropping 0 rehearsed row(s). Raise `max_new_tokens` if
                      the reasoning was truncated.
```

The pair matters more than either file alone. **Same SOP, same settings, same
budget — the clean clip returned three verdicts and the clip with a skipped step
returned nothing.** Harder judgements reason longer, so the failure correlates with
the inputs a verification agent exists to catch. Any accuracy number gathered
without checking for empty results is measuring the easy half of the set.

## Ground truth in the filename

`1_pass_2_pass_3_fail_A` means step 1 correct, step 2 correct, step 3 not; the
trailing letter is a take id. A repo convention, not a platform feature — it keeps
labels visible in `ls` and impossible to desynchronise from the media. `--score`
reads it from the filename or the record's `id`, or takes `--labels`.

Two cautions about scoring against labels like these:

- **A binary label cannot express `FAILED` vs `MISSING`.** Both mean "not done
  correctly", but they are different claims — attempted-and-wrong versus
  never-happened. In these clips the failures are **omissions** (the wrench never
  enters frame), so `MISSING` is the status that matches the video.
- **An all-pass clip cannot validate a detector.** Three `PASSED` verdicts on a
  clip where everything was done correctly is exactly what a model that always
  answers `PASSED` would produce. `--score` says so when every label is `pass`.

## No video ships here

The clips are bench recordings of an o-ring cap assembly, ~8 MB each. They live in
the worked example, where the twelve labelled takes are committed in full:
[task-verification-agent-example](https://github.com/archetypeai/task-verification-agent-example).
The outputs here are enough to exercise the schema and the scorer offline.
