# Next upload decision — 2026-09-27 ~22:14 IST

Confirmed public baseline: **V3 = 0.982564**, per user. Gap to 0.99: **0.007436**.
No later variant's public score is known here. Remaining attempts requested from user.

## Measured output scope

Streaming comparison of every test S1, matching IDs as sets, against the scored V3:

| Alternative | Changed S1s | Maximum absolute full-test macro-F change |
|---|---:|---:|
| V3b (ES protocol) | 3,894 | 0.00224756 |
| V2 (no confident-pair review) | 17,853 | 0.01030450 |

The bound is changed entities / all entities, since one entity contributes between
zero and one to the macro metric. This is a mathematical full-test bound, **not a
predicted gain**, and not a public-subset bound without its membership. V3b cannot
close the full-test-equivalent remaining gap alone, even with perfect changes.
V2's wider scope does not make it a better model; removing the review may undo gains.
Country details: `next_upload_scope.json`.

## Confirmed active experiment

The other session owns job `ber-final-d`. Its actual command runs sequentially:

1. V3d: five stacker seeds, 63 leaves, learning rate .03.
2. V3e: five seeds, 127 leaves, learning rate .02.
3. V3f: five seeds, 31 leaves, learning rate .03.

All use M2 + old M-v5 + completed CEs/review and ES gate selection. No new retrieval
or CE band expansion is in this job. Do not duplicate its bagging/capacity search.
These may reduce stacker variance but do not inherently address missing candidates
or unsupported language/domain errors. Measure their changed-entity scope and
paired validation result before spending an upload.

### V3d completed result, ~22:14 IST

Read directly from `/data/ber/logs/ber-final-d/v3d/stack_eval.json`:
validation **0.9896501423**, India 0.9883727907, US 0.9905017229, ES gate .70.
This is essentially unchanged from scored V3's validation 0.9896525533, not an
established improvement. Test links: 5,750,524; candidate export remains top 15.

**Ensemble reproduction:** V3d uses five models, `stacker.txt` and
`stacker_s1.txt` through `stacker_s4.txt`, averaging their probabilities. Its JSON
`best_iteration=468` and importance describe only the first member. Preserve all
five models, their feature order, parameters and individual best iterations.
Do not describe it as one 468-tree model or reproduce it using only `stacker.txt`.

The CE audit reports 76,077 repeated CE2 pair keys, 23,185 with score spread above
.001. Code averages them. The code comment attributes differences to bf16 padding,
but that cause is not independently proven here. Preserve input/model manifests
and inspect large disagreements before treating all duplicates as harmless noise.

## Recommendation

- Preserve scored V3 as the reference; V3b/V3c are not automatically replacements.
- Compare the best completed bagged model with V3 on paired validation and test scope.
- If a later variant is only a small gate/seed perturbation, do not present it as a
  credible mechanism for another +.0074 without evidence.
- A larger change would need wider retrieval, below-band recovery, or an independently
  validated French review improvement. Coordinate GPU/CPU availability and the
  22:45 stop-heavy-work checkpoint before attempting these.
- Do not re-upload V2 simply to seek a higher number. A country splice can isolate a
  hypothesis if portal attempts permit, but has no guaranteed gain.

## Reproduction utility added

`src/neural/select_band.py` implements explicit India and high-confidence-review band
selection, writes an ordered-key SHA256 manifest, refuses overwrites and rejects duplicate
keys/non-high-score review input. Role/country selection and deterministic sampling were
tested. It is separate from live inference and does not assert historical input equivalence.

From `src/`, for a future full reproduction:

```bash
python neural/select_band.py --input band.parquet --out band_india_new.parquet --mode country --country India
python neural/select_band.py --input band_m2.parquet --out band_m2_india_new.parquet --mode country --country India
python neural/select_band.py --input band_hi.parquet --out band_hi_sub_new.parquet --mode review --rest-rows 1000000 --seed 7
```

Use fresh inference directories for those newly ordered bands; do not reuse historical
row-count-only CE checkpoints. The main README's M-v7 command also still runs stage 2,
despite its stage-1-only comment: a clean stage-1-only execution path remains a final
reproducibility issue for the owning session to address.
