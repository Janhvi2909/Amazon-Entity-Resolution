# Start here — current progress and next steps

Last updated: **2026-09-27, approximately 22:16 IST**.
Deadline from original handoff: **23:59 IST today**. **Two portal uploads remain**, user confirmed.
This is the main continuation file. Update it after each completed experiment, portal
result, material failure, or change of plan. A status here is a timestamped observation,
not proof that a process is still running: check live jobs before acting.

## Confirmed standing

**Best user-confirmed public leaderboard: 0.982564, SUB-FINAL-V3_m2_ce_old_hi.**
Gain over SUB-N3: **+0.002960**. Remaining gap to 0.99: **0.007436**.
No result currently proves that the remaining gap can be closed before the deadline.

| File/system | Validation macro F0.5 | Public leaderboard | Status |
|---|---:|---:|---|
| SUB-N3 | 0.988049 | 0.979604 | Frozen previous best |
| SUB-R all-feature stack | 0.988214 | Unknown | Complete validated fallback, not uploaded |
| **SUB-FINAL-V3** | **0.989653** | **0.982564** | Scored best; preserve |
| SUB-FINAL-V3b | 0.989644 | Unknown | Frozen, ES-selected gate |
| V3c | Approximately 0.9897 | Unknown | Full candidate export; distinct model from V3b |
| V3d, five seeds, 63 leaves | 0.989650 | Unknown | Completed; essentially tied with V3 validation |
| V3e, five seeds, 127 leaves | Pending | Unknown | Active step at latest job check |
| V3f, five seeds, 31 leaves | Pending | Unknown | Queued after V3e in same job |

Validation is held-out **US/India training data**, not hidden-test accuracy. France
has no labels. Report portal scores only when the user or portal confirms them.

## Architecture now implemented

M2 normalization/address components and deeper sparse blocking → M2 stage-1 scores
+ old M-v5 stage-1/stage-2 probabilities → mDeBERTa-2, BGE and India-only MuRIL scores
+ confident-pair review → LightGBM stacker → target exclusivity and gated expected F0.5.

M2P1's `p2` is a copy of stage 1, not a trained second stage. The old M-v5 p2 supplies
the retained stage-2 evidence. V3 used 444 stacker trees and gate .55 selected on EVAL.
V3b/c/d use ES gate selection and are distinct artifacts; do not substitute their
metrics/model files into the scored V3 package.

## Current ownership and processes

- **Other session owns GPU jobs and final model experiments.** Do not kill/restart
  its pod, delete its score folders or launch competing GPU work without coordinating.
- CPU job `ber-final-d` runs V3d → V3e → V3f sequentially on nst-n7.
- V3d completed at **22:12:06 IST**, exit 0; V3e started at that time.
- This session's CPU reservation was only until 21:15; it has no continuing exclusive
  claim to nst-n7. Its own model experiments and full SUB-R validation are complete.
- Namespace is `amlc-ber-jy`; never modify `amlc-ber-phase3c`.

Read-only status:

```bash
kubectl -n amlc-ber-jy get pods
kubectl -n amlc-ber-jy logs job/ber-final-d --tail=20
kubectl -n amlc-ber-jy exec ber-util -- tail -10 /data/ber/logs/ber-final-d/v3e.log
```

The launcher returns from failed steps too; inspect each `EXIT=` and metrics/output
files rather than trusting `Job Completed` alone.

## Next actions, in priority order

1. **Finish reading V3e/V3f results.** Record ES selection score, held-out score, country
   metrics, model seeds and changes versus the scored V3. Do not rerun bagging elsewhere.
2. **Choose the next upload from evidence.** V3d currently has no meaningful validation
   advantage. Prefer a distinct supported hypothesis over an arbitrary seed change.
   Keep the last attempt for a response to actual portal feedback.
3. **Preserve scored V3 and finish package provenance in parallel.** It has a top-15
   candidate file; the final package needs the full exact scored set. The newer V3c
   has a full export, but its matching file differs. Verify the same candidate universe
   before combining a full candidate export with V3; do not falsely claim its model
   reproduces V3 matches.
4. **If enough time and coordinated resources remain**, diagnose low-score/absent
   candidates after M2 and test a selective CE-band expansion. A new rule must select
   candidates without using eval truth and include realistic negatives/competition.
5. **Stop substantial new work at 22:45**, freeze a selected model by approximately
   23:10, and leave 23:35 onward for uploads and fixes. Do not sacrifice a valid
   submission waiting for an incomplete experiment.

## What the remaining techniques can and cannot do

| Technique | Evidence/status | Next decision |
|---|---|---|
| Deeper M2 retrieval + old stage 2 + review | Implemented; V3 public +0.002960 | Keep as measured reference |
| Five-seed averaging | V3d validation essentially tied | Await V3e/f; no promised gain |
| Larger/smaller stacker trees | Running/queued | Compare held-out results and output scope |
| Broader low-score CE band | Not implemented in this session | First separate low-score misses from blocking misses |
| Dense e5 candidates | Prior diagnostic recovers 38% of known residual misses | Not a production gain; needs label-free retrieval and negative scoring |
| More French confident-pair review | Existing review already integrated | Verify coverage/disagreement before spending more GPU |
| Country-spliced candidate | Available strategy, not a measured gain | Use only to test a specific hypothesis with remaining uploads |
| Train-derived source caps | Used in newer protocol | Observed train maxima, not guaranteed test constraints; validate consistently |

The full design and implementation requirements are in
[EXECUTION_PLAN_TO_DEADLINE.md](EXECUTION_PLAN_TO_DEADLINE.md), sections 5–9.
Its original schedule is historical; the live status in this file supersedes it.

## Tested techniques not worth repeating tonight

- All pair features/global base-p2 context improved the original stack to 0.988214
  validation (SUB-R), but only 13,950 test entities changed versus SUB-N3.
- Inverse-sqrt/inverse entity-size loss weighting scored 0.988196/0.988178: below SUB-R.
- ES-only stacker training: 0.988051, no benefit.
- Exact expected-F versus approximation: approximately +0.000009.
- Exact target name+address propagation: +0.000013, too small.
- Name-only propagation: **harmful**, 627/2,047 additions correct; reject.
- Full-pool target competition audit changed original stack validation by <.00001.
- Prior handoff rejects blanket nearby-house filters, all-country MuRIL and generic
  gate tuning as a route to a large gain.

## Output-scope evidence for upload decisions

Compared with scored V3:

| Variant | Changed full-test S1s | Maximum possible absolute full-test macro-F change |
|---|---:|---:|
| V3b | 3,894 | 0.002248 |
| V2, no review | 17,853 | 0.010305 |

These are mathematical bounds, **not expected gains**. The public subset is unknown,
so they are not exact public-board bounds. V3b cannot by itself close a .007436 gap
on the full-test-equivalent metric. V2 has more scope but could undo review gains.
Detailed country counts: `analysis/score_audit/next_upload_scope.json`.

## Integrity and reproducibility findings

- Scored V3 matching MD5 verified intact: **fffea65157b36546bbf566889ce23ca4**.
- Supplementary streaming V3 audit **PASS**: 1,732,544 S1 rows, 5,751,431 matches,
  25,988,160 candidates, 102,084 empty predictions. Checks order/coverage, syntax,
  within-list uniqueness, and match subset. It does **not** check target existence,
  exclusive ownership, or inference provenance. Report: `analysis/score_audit/v3_stream_validation.json`.
- V3c full export job completed with approximately 115.7093 candidates per S1;
  remote folder `/data/ber/logs/ber-final-c/v3c`. Its model has 468 trees and gate .70,
  not V3b's 485 trees/.65 or V3's 444 trees/.55.
- V3d averages **five** models. JSON tree count/importance describes only the first.
  Keep `stacker.txt` and `stacker_s1.txt`–`stacker_s4.txt`, plus individual parameters.
- CE2 has repeated pair keys and score disagreements. Averaging is implemented, but
  the explanation that all differences are bf16 padding noise is not independently
  proven. Preserve model/input manifests; inspect material disagreements.
- README's M-v7 command still enters stage 2 despite a stage-1-only comment. A clean
  stage-1-only reproduction path needs attention by the package owner.
- Official full-candidate validation can exceed 8 GiB; don't run it in `ber-util`.
  SUB-R's successful full ID check used a dedicated 40-GiB-limit worker. A helper
  check previously restarted `ber-util`; avoid repeating that disruption.

## Tools added by this session

In `code/business_entity_resolution/src/neural/`:

- `stack_refine.py`, `stack_audit.py`: measured original-stack experiments/replay.
- `paired_audit.py`, `profile_changes.py`: per-entity validation and label-free test diffs.
- `closure_audit.py`, `exact_f_audit.py`, `macro_weight_audit.py`: completed ablations.
- `export_refine.py`: bounded-memory full scored-candidate export for SUB-R.
- `select_band.py`: deterministic country/review subset selection with ordered-key
  SHA256 manifest and overwrite guards. Tested role/country selection, deterministic
  sampling, duplicate rejection and high-confidence input guard. Does not certify
  historical band identity; use new inference folders.
- `validate_streaming.py`: supplementary bounded-memory aligned-file audit; explicitly
  reports omitted checks. Useful for huge candidate files, not a replacement for
  official ID/provenance checks.

## Files to open when resuming

1. This file — current decisions and status.
2. `analysis/score_audit/PORTAL_RESULTS.json` — user-confirmed portal scores/attempt count.
3. `EXECUTION_PLAN_TO_DEADLINE.md` — detailed architecture, commands and experiment designs.
4. `method_result.md` — experiment history and measured outcomes.
5. `analysis/score_audit/NEXT_UPLOAD_DECISION.md` — latest upload-scope/ensemble audit.
6. `submissions/SUB-FINAL-V3_m2_ce_old_hi/` — scored best; never overwrite.
7. `Documentation_template.md` and `code/business_entity_resolution/README.md` — the
   other session is updating these; coordinate edits to avoid clobbering its work.

## Update discipline for the next session

For every new run, append: hypothesis, exact code/model/data versions, train/ES/EVAL
roles, selected gate, per-country validation, changed test entities, runtime/memory,
artifact path/hash, validator coverage, and **portal score only if actually observed**.
Mark proposed, running, completed, rejected and uploaded separately. Preserve failed
experiments and their reason. Keep the goal of 0.99 visible without claiming success
from a validation number or substituting a smaller goal.

## Update 22:42 IST (Claude): CLAUDE_NEXT_STEPS_099 executed

- **Leaderboard:** SUB-FINAL-V3 = **0.982564**.
- **Recommended upload:** `submissions/SUB-FINAL-V3s_support/matching_results.tsv` (md5 `078eed8c2b10e38d2845392f5ff23bcf`, validator PASS). Its score comes from the portal once uploaded.
  - Measured on validation: ES 0.989820 (V3c 0.989768); EVAL 0.989706 (V3c 0.989651).
  - Paired +0.000055, bootstrap 95% CI [+0.000022, +0.000096], 706 better / 510 worse.
  - Mechanism: `stack.py --support` (agreement with the S1's other confident candidates, name frequencies, target address flag).
- **Optional France test:** `SUB-FINAL-V3s_fr08` (md5 `4a1a7240c6d485f46906bffdd7d59173`): France probabilities raised to the power 0.8, +3.5k France links. Unmeasurable; upload only if the portal keeps the best score.
- **Frozen, not uploaded:**
  - V3b `1b7a1b30ed335d80ea581cdbc4966a4c`
  - V3c `ad535ca592bafe9c5f112942caf99738`
- **Rejected:**
  - Bagging (V3d/e/f): no gain.
  - V3s2 (+ core-name frequency, rival count): ES tie, EVAL 0.989680.
- **Step 2 error table (V3s):** missing candidate +0.0035; reviewed-but-rejected +0.0051 (85% no-address targets); below band +0.0009; lost to another S1 +0.00001; false match +0.0007; singleton false match +0.0002. `neural/errtab.py` with `stack.py --dump-eval`.
- **Step 3 France audit:**
  - All 765,958 France confident pairs were scored (input `band_hi_ord`, 5 of 5 parts).
  - Duplicates come from the same checkpoint, max length 160 and identical text; only the batch size differs (128 vs 64, bf16 padding). They are averaged.
- **Steps 4 and 5 skipped:** below-band oracle ≤ +0.0009, and retrieval can't finish before the cutoff.
- **Package:** built on n7 by job `ber-pkg` in `/data/ber/pkg_final`.
  - Contents: the V3s matching file and the full scored candidate set (`ber-final-c/v3c/candidate_pairs.tsv`, every M2P1 test pair, identical for V3s).
  - Checks: the validator runs with `--check-ids`, then `make_submission.py`.

## Update 23:24 IST (Claude): CLAUDE_FRANCE_FINAL_40_MIN_PLAN executed

- **Portal:** V3 0.982564 (best); V3s_fr08 0.982529. V3 is preserved (md5 `fffea65157b36546bbf566889ce23ca4`).
- **Minute 0–5:**
  - Error table (V3s): reviewed-but-rejected +0.0051 oracle, 85% no-address targets; missing candidates +0.0035.
  - France CE coverage complete; duplicates come from the same checkpoint and identical text (batch-size bf16 noise).
  - Benchmark: mDeBERTa-2 at 2,800–3,400 pairs/s (max length 160, batch 128). Budget 820k pairs.
- **Preferred branch (anchor CE):**
  - Anchors from M-v5 p2 only (p2 ≥ 0.99, address-bearing, exclusive target with a single claim ≥ 0.5, best per S1, never self): 961,603 anchors.
  - Candidates: label-free existing CE-band pairs of those S1s with p2 < 0.995. Scored 820,143 pairs: all 363,962 France test pairs, plus whole-S1 samples of REST 200k, ES 69k and EVAL 187k.
  - Scored with the fine-tuned mDeBERTa-2 in `/n6/ce2_anchor`: 2 of 2 parts, finite, no duplicates.
  - Fresh stacker `stack.py --support --anchor` (REST fit, ES gate 0.70):
    - ES 0.989799 (V3s 0.989820); EVAL 0.989701 (V3s 0.989706).
    - Paired −0.000005, CI [−0.000027, +0.000016]; US −0.000012, India +0.000005.
    - False positives equal (1,491); France S1 changed 2,377 (< 12,883).
  - **NO-GO. Branch stopped.**
- **Fallback (canonical-text view):** France-only transformation with no labeled analogue, so it can't be measured offline; not run through the stacker. Its GPU scores exist in `/n6/ce2_frtta` (738,978 France pairs) but are unused.
- **fr125 (France α 1.25):** built and validated, md5 `4e733c238c35c53cf3bf3602f8198b8e`. Not recommended, since the plan forbids blind France scaling.
- **Final candidate:** V3s (`078eed8c2b10e38d2845392f5ff23bcf`, measured US/India gain +0.000055, CI > 0). Otherwise retain V3 at 0.982564.
- **Package:** rebuilding with V3s by job `ber-pkg3` (full scored candidate set, `--check-ids`).

## Update 23:36 IST (Claude): CLAUDE_FINAL_15_20_MIN_DECISION executed

- **Final upload candidate:** `SUB-FINAL-V3s_support/matching_results.tsv`.
  - md5 `078eed8c2b10e38d2845392f5ff23bcf`; validator `--check-ids` PASS.
  - 1,732,544 unique S1 rows (102,815 empty).
  - Paired gain +0.000055 measured against V3c (the V3 pipeline with duplicate averaging and the ES gate), not against V3 directly.
- **V3 preserved:** md5 `fffea65157b36546bbf566889ce23ca4`, LB 0.982564.
- **Package:** `submissions/FINAL_PACKAGE_V3s/team_submission.zip`, md5 `f5828504cd66f77cc057f7be326ce772`, `unzip -t` OK.
  - Inside: matching = V3s (`078eed8c…`); candidates = the full scored set (`ac4c44e9…`, every M2P1 test pair, 1,732,544 rows).
  - README documents `stack.py --support`.
- **Anchor CE:** rejected (see the 23:24 entry). Do not repeat it.
- **0.99 is not achieved.** Record the actual portal score for V3s here once returned.

## Update 2026-09-29 (Claude): final package = V3 (best leaderboard 0.982564)

- **Package:** `submissions/FINAL_PACKAGE_V3/team_submission.zip` (md5 `afe509b60f99190312b9bf7854d1b5b8`, 55 files, `unzip -t` OK).
  - `output/matching_results.tsv`: V3 (`fffea65157b36546bbf566889ce23ca4`), identical to the leaderboard file.
  - `output/candidate_pairs.tsv`: the full scored BLK-v4b@40n20 set (`ac4c44e927509b6809bb8c37e9c23800`, 200,471,427 pairs).
  - `src/neural/stack_v3.py`: the exact stacker V3 ran with (snapshot `ber_code_0927211612`, md5 `264b09e1…`).
  - README and Documentation describe V3: gate 0.55 chosen on eval, test-only caps, review band `band_hi_ord` recipe, first-score duplicates.
- **Checks:**
  - Streaming check: V3 matching vs full candidates PASS (order, coverage, syntax, subset).
  - Target exclusivity: 5,751,431 links, 0 targets claimed twice.
  - Official validator: PASS on V3 matching plus top-15 candidates. `--check-ids` PASS on this exact candidate file (n7, 09-27).
  - The full official validator was not rerun locally (needs ~35 GB); the cluster is no longer accessible (Forbidden).
- **Before upload:** the team name and members placeholders are still open.
