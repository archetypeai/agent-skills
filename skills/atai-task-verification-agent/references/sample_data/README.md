# Sample data

Real artifacts from real runs against the `tva` blueprint: 2026-08-11 to 2026-08-12, and
2026-10-26 on the blueprint published 2026-10-15. Three clips ship, so the skill can be
exercised **live** as well as offline.

Offline — no API key, no network, no GPU:

```sh
python3 ../run_tva_agent.py --score tva-output-1_pass_2_pass_3_pass_A.json
python3 ../run_tva_agent.py --score tva-output-1_fail_2_pass_3_pass_A-FALSE-PASS.json
python3 ../run_tva_agent.py --score tva-output-1_pass_2_pass_3_fail_A-mnt2048-EMPTY.json
```

Live — the same clip and SOP that produced the first output, ~20 min:

```sh
python3 ../run_tva_agent.py --video 1_pass_2_pass_3_pass_A.mp4 --sop oring-numbered.txt

# --sop is optional: it defaults to sop/oring-numbered.txt if your project has one,
# else to the copy in this directory. The script prints which is in force.
python3 ../run_tva_agent.py --video 1_pass_2_pass_3_pass_A.mp4
```

On the current (2026-10-15) blueprint, expect **step 3 `FAILED`** on this clip, though
every step was performed — `tva-output-1_pass_2_pass_3_pass_A-FALSE-FAIL.json` is that
run. The three-`PASSED` output predates that blueprint. The digest never matches a
committed one, because the record's `id` carries the run's upload suffix; compare the
`results`.

## Files

| File | What it is |
|---|---|
| `1_pass_2_pass_3_pass_A.mp4` | All three steps performed. 7.5 MB. |
| `1_pass_2_pass_3_fail_A.mp4` | Step 3 skipped — **the wrench never enters frame.** 5.2 MB. |
| `1_fail_2_pass_3_pass_A.mp4` | Step 1 skipped — **the O-ring is never fitted**, though the rings sit on the mat throughout. 7.5 MB. |
| `oring-numbered.txt` | The 3-step SOP both clips were run against, one step per line. |
| `tva-output-1_pass_2_pass_3_pass_A.json` | Three `PASSED` verdicts with timestamps and reasons. 678 bytes. |
| `tva-output-1_pass_2_pass_3_pass_A-FALSE-FAIL.json` | **The same clip on the 2026-10-15 blueprint**: step 3 `FAILED` on a wrench that visibly tightens the cap, with an invented reason (a plastic bag that is not in the clip). Reproduced identically by two clients. |
| `tva-output-1_fail_2_pass_3_pass_A-FALSE-PASS.json` | **The failure you cannot detect from the output.** Step 1 `PASSED` on a clip where the O-ring was never fitted, with an invented reason. Reproduced before 2026-10-15; the current blueprint catches this clip. |
| `tva-output-1_fail_2_pass_3_pass_A-CORRECT-MISSING.json` | **The same clip on the 2026-10-15 blueprint**: step 1 correctly `MISSING`. Across all six O-ring takes in the worked example, though, five still pass — this take is the exception. |
| `tva-output-1_pass_2_pass_3_fail_A-CORRECT-MISSING.json` | The **same clip at an adequate budget**: steps 1–2 `PASSED`, step 3 correctly `MISSING` — *"no wrench is introduced"*. This is what the shipped default returns. |
| `tva-output-1_pass_2_pass_3_fail_A-mnt2048-EMPTY.json` | The **same clip at `max_new_tokens: 2048`**: `results: []` — 44 bytes, `job.completed`, no ERROR row. What an exhausted reasoning budget returns. |

## Read these two together — they are the whole finding

| output | the object the step needs | what it answered |
|---|---|---|
| `1_pass_2_pass_3_fail_A-CORRECT-MISSING` | the wrench — **never enters frame** | `MISSING` ✅ |
| `1_fail_2_pass_3_pass_A-FALSE-PASS` | the O-ring — **on the mat throughout, untouched** | `PASSED` ❌ |

Neither action happens in its clip. The only difference is whether the **object** is on
screen. (Both outputs predate 2026-10-15. The current blueprint catches this particular
O-ring take, but still passes five of the six in the worked example.)

The model reliably answers **"is this thing in the video at all?"** — and when the
object is present, it assumes the action happened. A workstation has its parts laid
out by definition, so the omissions you deployed the agent to catch are the ones in
the blind spot. `reason` reads like an observation and is not one: *"stretching the
O-ring, and placing it onto the cap groove"* describes something that never occurred.

**Do not surface `reason` to a user as an audit trail.**

## Why the empty one is here — and note which budget produced it

`…-mnt2048-EMPTY.json` and `…-CORRECT-MISSING.json` are **the same clip and the same
SOP**. Only `max_new_tokens` differs:

| budget | result |
|---|---|
| 2048 | `results: []` |
| adequate (5760, the shipped default) | 3 verdicts, step 3 correctly `MISSING` |

So running that clip today returns **correct verdicts, not the empty result** — you have
to ask for 2048 to see the failure:

```sh
python3 ../run_tva_agent.py --video 1_pass_2_pass_3_fail_A.mp4 --max-new-tokens 2048
```

The empty file is not broken — it is what the platform returns when the budget is spent
inside the model's `<think>` block before any verdict is emitted:

```
WARN  parser.running  TaskVerificationResultsParserNode: generation ended inside the
                      model's reasoning block (no `</think>`), so it never produced an
                      answer; dropping 0 rehearsed row(s).
```

**Harder judgements reason longer, so the failure correlates with the inputs a
verification agent exists to catch** — at 2048 the all-pass clip returned three verdicts
while this one, with a step skipped, returned nothing.

Note what does *not* fix it: raising the budget. That WARN suggests it, and on these
clips 8192 and 16384 both returned nothing where 5760 returned correct verdicts. Read
the `dropping N` count instead — see SKILL.md.

## What each clip reproduces today

On the 2026-10-15 blueprint, at the shipped default of 5760:

| clip | reproduces | before 2026-10-15 |
|---|---|---|
| `1_pass_2_pass_3_pass_A.mp4` | **the false fail** | the 3-`PASSED` output |
| `1_pass_2_pass_3_fail_A.mp4` | a correct `MISSING` | the same, **and** the empty result at `--max-new-tokens 2048` (not re-tested since) |
| `1_fail_2_pass_3_pass_A.mp4` | a correct `MISSING` | **the false pass** |

The remaining nine labelled takes, a batch sweep and an offline scorer are in the worked
example:
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
  status that matches the video. **`FAILED` was never emitted before 2026-10-15**; since
  then it has appeared only on steps that were performed correctly.
- **An all-pass clip cannot validate a detector.** Three `PASSED` verdicts on a clip
  where everything was done correctly is exactly what a model that always answers
  `PASSED` would produce. `--score` says so when every label is `pass`.
