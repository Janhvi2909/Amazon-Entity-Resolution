# Independent audit — 2026-09-27, ~20:55 IST

This session leaves the other session's GPU jobs and `neural/stack.py` unchanged.
CPU permission: nst-n7 available until 21:15 IST; our jobs have 25-minute hard deadlines.
User clarified **3 leaderboard uploads remain**.

## Measured findings

- Existing SUB-N3 reported eval: 0.9880492468. Independent saved-model replay:
  0.9880520006 (minor ordering/tie differences). Restoring full-pool target competition:
  0.9880512658. Eval-only exclusivity is a code issue, but does not explain the LB gap.
  Full-pool diagnostic has missing FIT CE scores and in-sample REST stacker predictions;
  it must not be described as fully unbiased validation.
- M-v5 + existing CE2/BGE/MuRIL, with all pair features and global base-p2 competition
  in the stacker: **eval 0.9882140353**, India 0.9860742814, US 0.9896405594.
  Train REST; early stop ES; gate .45 selected on ES, not eval.
- Numeric features only: 0.9881166936. Numeric + base competition: 0.9881217968.
- Exact expected F0.5 vs approximation: +0.0000091. Not worth prioritizing.
- Target-to-target exact name+address expansion: 197 eval additions, 169 true,
  +0.0000133. Name-only expansion: 2047 additions, 627 true; -0.0004872. Rejected.
- Frozen SUB-N3 final matching file has zero targets assigned to multiple S1s.

## Code and outputs available for reuse

- `code/business_entity_resolution/src/neural/stack_refine.py`: bounded-memory joins,
  ES-selected gate, all-feature and numerical stacker ablations.
- On PVC `/data/ber/codex_audit/`: compact `band_{train,test}.parquet`, all three models,
  eval/test probability updates, `experiments.json`, logs.
- All-feature model uses **M-v5 stage-2**, not M2 stage-1. Consider joining M-v5 p2 as
  an additional M2 stacker feature (with a missing-value indicator for new candidates).
  This retains the previously trained consensus/context knowledge.
- `base_other_t` is the best base p2 among rival S1s, including confident candidates;
  `base_margin_t`, base p2 sum and count per S1 also help. Do not restrict this context
  to the CE uncertain band.
- `/data/ber/codex_refine_submission/`: separate candidate export job; inspect
  `VALIDATOR.txt` before use. France impact remains unmeasured.

## Submission audit issue

Official rules require **every scored candidate**, not top 15 chosen after scoring.
Current `stack.py` exports top 15 after scoring. The official format validator can pass
this but cannot verify inference provenance. `neural/export_refine.py` instead exports
the exact original score-table pairs in bounded S1 chunks.

## Live-status caveat

The confident-pair GPU review had CUDA OOM errors when inspected ~20:41 IST, and a
queued restart script was present. This session did not change it; the other session
continues to own all GPU recovery and M2 integration.
