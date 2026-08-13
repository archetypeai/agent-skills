# Sample data

Real artifacts from real runs against `tva` on dev, 2026-08-11 to 2026-08-12. Two
clips ship, so the skill can be exercised **live** as well as offline.

Offline — no API key, no network, no GPU:

```sh
python3 ../run_tva_agent.py --score tva-output-1_pass_2_pass_3_pass_A.json
python3 ../run_tva_agent.py --score tva-output-1_fail_2_pass_3_pass_B-FALSE-PASS.json
python3 ../run_tva_agent.py --score tva-output-1_pass_2_pass_3_fail_A-mnt2048-EMPTY.json
```

Live — the same clip and SOP that produced the first output, ~20 min:

```sh
python3 ../run_tva_agent.py --video 1_pass_2_pass_3_pass_A.mp4 --sop oring-numbered.txt

# --sop is optional: it defaults to sop/oring-numbered.txt if your project has one,
# else to the copy in this directory. The script prints which is in force.
python3 ../run_tva_agent.py --video 1_pass_2_pass_3_pass_A.mp4
```

Expect three `PASSED` verdicts. The digest may differ from the committed one if your
`max_new_tokens` differs — generation is deterministic for a given budget, and the
committed output predates the current default of 5760.

## Files

| File | What it is |
|---|---|
| `1_pass_2_pass_3_pass_A.mp4` | All three steps performed. 7.5 MB. |
| `1_pass_2_pass_3_fail_A.mp4` | Step 3 skipped — **the wrench never enters frame.** 5.2 MB. |
| `oring-numbered.txt` | The 3-step SOP both clips were run against, one step per line. |
| `tva-output-1_pass_2_pass_3_pass_A.json` | Three `PASSED` verdicts with timestamps and reasons. 678 bytes. |
| `tva-output-sealant-CORRECT-MISSING.json` | The all-pass clip, run with step 3 replaced by *"The worker squeezes thread sealant from a tube onto the manifold port"* — a prop that appears in **no** clip. Steps 1–2 `PASSED`, step 3 **correctly** `MISSING`: *"No thread sealant tube is shown or used in the video."* |
| `tva-output-1_fail_2_pass_3_pass_B-FALSE-PASS.json` | **The failure you cannot detect from the output.** Step 1 `PASSED` on a clip where the O-ring was never fitted, with an invented reason. Its clip is not shipped — see below. |
| `tva-output-1_pass_2_pass_3_fail_A-mnt2048-EMPTY.json` | `results: []` — 44 bytes, `job.completed`, no ERROR row. What an exhausted reasoning budget returns. |

## Read these two together — they are the whole finding

| output | what the model was asked | what it answered |
|---|---|---|
| `sealant-CORRECT-MISSING` | did the worker apply sealant? *(no such prop anywhere in the video)* | `MISSING` ✅ |
| `1_fail…B-FALSE-PASS` | did the worker fit the O-ring? *(the rings are on the mat, untouched)* | `PASSED` ❌ |

Both actions are absent from the footage. The only difference is whether the **object**
is on screen.

The model reliably answers **"is this thing in the video at all?"** — and when the
object is present, it assumes the action happened. A workstation has its parts laid
out by definition, so the omissions you deployed the agent to catch are the ones in
the blind spot. `reason` reads like an observation and is not one: *"stretching the
O-ring, and placing it onto the cap groove"* describes something that never occurred.

**Do not surface `reason` to a user as an audit trail.**

## Why the empty one is here

Not a broken file — it is what the platform returns when the output budget is spent
inside the model's `<think>` block before any verdict is emitted:

```
WARN  parser.running  TaskVerificationResultsParserNode: generation ended inside the
                      model's reasoning block (no `</think>`), so it never produced an
                      answer; dropping 0 rehearsed row(s).
```

The pair matters more than either file alone. Same SOP, same settings: the clean clip
returned three verdicts and the clip with a skipped step returned nothing. **Harder
judgements reason longer, so the failure correlates with the inputs a verification
agent exists to catch.**

Note what does *not* fix it: raising the budget. That WARN suggests it, and on these
clips 8192 and 16384 both returned nothing where 5760 returned correct verdicts. Read
the `dropping N` count instead — see SKILL.md.

## What cannot be reproduced from here

**Both shipped clips have step 1 performed correctly**, so the false pass above cannot
be reproduced locally — it needs a clip where the O-ring is skipped. The twelve
labelled takes, including six such clips, are committed in the worked example:
[task-verification-agent-example](https://github.com/archetypeai/task-verification-agent-example).

## Ground truth in the filename

`1_pass_2_pass_3_fail_A` means step 1 correct, step 2 correct, step 3 not; the trailing
letter is a take id. A repo convention, not a platform feature — it keeps labels
visible in `ls` and impossible to desynchronise from the media. `--score` reads it from
the filename or the record's `id`, or takes `--labels`.

Two cautions about scoring against labels like these:

- **A binary label cannot express `FAILED` vs `MISSING`.** Both mean "not done
  correctly", but they are different claims — attempted-and-wrong versus
  never-happened. Every failure in these clips is an omission, so `MISSING` is the
  status that matches the video, and **`FAILED` has never once been emitted** across
  12 clips.
- **An all-pass clip cannot validate a detector.** Three `PASSED` verdicts on a clip
  where everything was done correctly is exactly what a model that always answers
  `PASSED` would produce. `--score` says so when every label is `pass`.
