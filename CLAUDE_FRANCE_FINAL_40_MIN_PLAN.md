# Claude: final 30–40 minute France improvement plan

## Read this first

The user requests a plan only from this session. No new training, inference, submission,
or remote job was launched to prepare this document. Local code/results were inspected
to avoid suggesting experiments that have already been completed.

**Latest actual public scores, reported by the user:**

| Submission | Public macro F0.5 |
|---|---:|
| SUB-N3 | 0.979604 |
| **SUB-FINAL-V3_m2_ce_old_hi** | **0.982564** |
| SUB-FINAL-V3s_fr08 | **0.982529** |

The newest upload is **0.000035 below the best**, effectively a very small difference,
not evidence of an improvement. Preserve scored V3. The last confirmed upload count
was two before this new upload; **one probably remains**, but verify the portal count.
Also verify whether the portal retains the best result or requires explicit final selection.

**Target: 0.99. Remaining gap from best: 0.007436.** No honest plan can guarantee this
in 40 minutes without French labels. Do not manufacture confidence or treat local
validation as the hidden-test score. Deliver the strongest justified candidate within
the time budget while keeping the known best available.

France is approximately 14.98% of full-test S1s. If US/India were unchanged, achieving
the entire +0.007436 through France requires roughly **+0.0497 France macro F0.5**,
assuming the public subset has the full-test country mixture. Its mixture and actual
France score are unknown. A tiny calibration adjustment is not evidence for a gain
of that size.

## What is already done — do not repeat it

The current architecture already has multilingual mDeBERTa-2, BGE reranker, India-only
MuRIL, M2 deeper candidates, old M-v5 stage-2 fusion, and confident French pair review.

The local V3s support result is **0.989706185 validation**, gate .70 selected on ES,
583 stacker trees. It already includes `sup_name_eq`, `sup_core_eq`, `conf_n_q`,
`t_has_addr`, `qname_freq`, and `tname_freq`. Adding these same equality/support features
again is not a new experiment. The fr08 variant records 809,058 unseen-country links
versus 805,483 for its base. Its actual portal score is lower than V3.

Prior tests found only small gains from more stacker seeds, numerical features,
exact expected-F, or target name/address equality propagation. Name-only propagation
was harmful. Do not spend the last attempt blindly trying fr125 or another French
probability multiplier. A nearly flat aggregate result cannot tell us the correct
direction of another adjustment, especially when multiple model changes are combined.

## Decision: choose ONE new evidence source, not five simultaneous experiments

**Preferred branch:** neural comparison with a confidently matched target record as
an alternative reference, using the already fine-tuned multilingual model.

This is different from the existing support-name equality features: the CE reads the
full candidate and anchor names/addresses. It can recognize a noisy business alias or
address formulation shared by two target records even when direct S1 comparison is weak.
It can also be wrong, particularly when an anchor is a false match. It requires a
held-out pilot and careful anchor construction; there is no assumed gain.

**Fallback branch:** a single additional text view of existing pairs, scored with the
same trained CE—raw text versus the project's existing canonicalized text. Use this
only if anchor scoring is not implementable quickly or the pilot rejects it.

**Do not combine both branches under this deadline.** If neither passes its pilot
quickly, retain the best measured submission and finish packaging.

## Minute 0–5: freeze reference, inspect resources, select feasible scope

1. Preserve `submissions/SUB-FINAL-V3_m2_ce_old_hi/matching_results.tsv` unchanged.
   Its verified matching MD5 is **fffea65157b36546bbf566889ce23ca4**.
2. Record the new fr08 portal score 0.982529 and verify remaining uploads.
3. Check which jobs are actually live and whether the GPU is available. Do not restart
   another process or delete score folders to reclaim memory without coordinating.
4. Locate the final compact train/test pair scores, raw text, split roles, models and
   labels. Reuse them; do not rerun normalization/blocking or the full base models.
5. Read `neural/errtab.py` output, if completed, to identify the highest-loss recoverable
   group in the current fusion. If absent, compute a compact decomposition on cached
   held-out scores: missing candidates, candidates not CE-reviewed, reviewed rejects,
   rival losses and false matches. Do not spend most of the window on a full new EDA.
6. Benchmark **2,000 representative inference pairs** at the actual token length and
   safe batch size. Use measured throughput to determine workload:

   `maximum pairs = sustained pairs/second × available inference seconds × 0.7`

   Include REST/ES/EVAL pilot rows, French test rows, extra competitor rows and all
   text views in that budget. The 0.7 factor reserves time for stalls and export.
7. If the required workload cannot finish with at least 10 minutes left for selection,
   validation and upload, shrink scope using label-free signals or abandon this branch.

## Minute 5–12: implement and test neural anchor evidence

### A. Define anchors using predictions only

- Use one anchor per S1 initially to keep inference affordable.
- Anchor must be an already predicted match with strong agreement between the tree
  and available CEs, and a clear margin over rival S1 claims.
- Choose confidence/margin thresholds on the development/ES data, not by guessing
  from French test examples. Start the pilot with conservative thresholds.
- Prefer an address-bearing anchor; do not use an ambiguous no-address record as the
  sole source of identity evidence.
- Exclude the candidate itself. **Never let a candidate compare to itself as its anchor.**
- Form training anchors from appropriate out-of-fold/held-out base predictions.
  Do not substitute ground-truth matches as training anchors while using predicted
  anchors at test time. Do not use in-sample final-stack predictions without explicitly
  controlling their optimism; base-model OOF predictions are the safer available source.
- Preserve target exclusivity. An anchor claimed by two businesses is not reliable.

### B. Choose candidate pairs without labels

Use existing M2 candidates so the first experiment does not require another retrieval
system. Prioritize pairs with disagreement between direct CE and tree/stack evidence,
or weak direct-name evidence but strong address evidence. Include predicted matches
that may need rejection as well as rejected pairs that may need recovery.

Apply the same selection rule to training roles and France test. Do not select only
known missed positives. Include the realistic negatives, no-address cases, and rival
S1 claims needed to assess precision and exclusivity.

Keep all unaffected pairs at their baseline scores. Missing support is missing evidence,
not a negative label or probability zero. Retain a flag indicating whether support exists.

### C. Score an additional relationship

For candidate target T and reliable anchor A, score:

`[anchor business_name ; anchor business_address]`
paired with
`[candidate business_name ; candidate business_address]`

Use the **already fine-tuned mDeBERTa-2 checkpoint**, or the existing trained BGE model
if it is available and benchmarks faster for this workload. Do not load a base model
with a random classification head. Do not score both models initially.

Important limitation: this CE was trained on S1-to-target pairs, so target-to-target
inputs are a distribution shift. The pilot must establish whether the score adds useful
information. Do not treat it as a calibrated match probability by itself.

### D. Learn how to combine the new score

Add a small residual classifier or additional stacker features:

- original direct CE and old tree scores;
- anchor-to-candidate CE score;
- anchor confidence and rival margin;
- anchor/candidate address availability;
- direct versus support CE disagreement;
- house-number and legal-form evidence already available;
- explicit missing-support flag.

Fit on REST with label-free candidate selection, early-stop/select decisions on ES,
and use EVAL only for the final comparison. If using a baseline stack score trained on
REST, obtain out-of-fold baseline scores for REST or train the residual on a separate
held-out development subset. **Do not train a residual on in-sample stack scores and
call the resulting test calibration reliable.** A fresh stacker using the existing
base features and new support score may be simpler than nested stacking.

Do not impose `new_probability = max(direct, support)` or unconditional transitive
closure. Those promote false anchors into additional false matches.

## Minute 12–15: hard go/no-go gate

Require all of the following before scaling the branch:

1. End-to-end held-out macro F0.5 improves using the actual set decision and rival
   competition, not just pair log loss or selected-positive recall.
2. Report improved/worsened S1 counts, false positives, singleton mistakes and
   newly recovered links. Check results separately for US and India.
3. The improvement is not explained by candidate self-comparison, label-selected
   anchors, in-sample leakage, or excluding hard negative/rival pairs.
4. Scoring coverage is feasible within the remaining time measured from the benchmark.
5. The affected French S1 count is large enough to matter. For reference, any method
   changing fewer than approximately **12,883 of all 1,732,544 test S1s** cannot add
   .007436 to full-test macro F0.5, even with perfect corrections. This is an upper
   bound, not a predicted gain; public-subset membership is unknown.

Prefer a paired improvement that is stable across the two labeled countries. This
still does not validate France; it merely avoids deploying a method that already
fails in the domains where its correctness can be measured.

If the pilot fails, stop it. Do not lower the evidence standard because of the deadline.

## Fallback: canonical-text CE view, if it fits the same time budget

Use this only if the preferred branch is rejected early enough to leave adequate time.

1. Reuse the project's existing French canonicalization of streets, city/region ordering,
   transliteration and numbering. Do not fetch external address/identity data.
2. Preserve business names, numbers, `bis/ter`, legal forms and distinguishing tokens.
   Canonicalization can erase identity information; do not strip it aggressively.
3. Score a selected disagreement band twice only where needed: existing raw-text score
   plus a new canonical-text score from the already fine-tuned multilingual CE.
4. Teach a small stacker how to use the additional view on REST/ES and evaluate on EVAL.
   Do not assume an arithmetic mean is calibrated or helpful.
5. Test the same transformations on labeled data; a France-only transformation with
   no labeled analogue cannot yield a measured France gain offline.
6. Keep the baseline for unselected pairs. Record precisely which view and model revision
   produced every score. Use fresh checkpoint directories.

**Do not claim this is a proven fix.** French canonicalization alone previously produced
suspicious changes. The experiment here is an extra neural view, not blanket replacement
of raw input or a hand-written address rule.

## Should we add a small pretrained French model?

**Not simply because it understands French.** A generic pretrained language model or
embedding model is not automatically a business-identity classifier. Adding a random
binary head gives unusable predictions; a retrieval reranker may favor related but
distinct businesses. A new model also needs calibration and testing.

Under 30–40 minutes, the strongest feasible default is **reuse the task-fine-tuned
multilingual model already resident on the GPU and give it new evidence**. mDeBERTa-2
and BGE are already multilingual; the missing ingredient may be coverage or context,
not another language encoder.

Only consider a different model if **all** conditions hold:

- weights are already local or download time is negligible;
- exact checkpoint license satisfies MIT/Apache-2.0 and total challenge constraints;
- it already supports meaningful pair scoring, or task adaptation fits the deadline;
- a quick labeled pilot demonstrates complementary errors, not merely different scores;
- inference, fusion, validation and export fit the measured remaining time.

Do not initiate a model shopping spree, multi-billion-parameter LLM adjudication, or
translation model download with one final upload left. Do not pseudo-label French
records from the current model and then claim improvement by matching those pseudo-labels.
No external business databases, geocoding or identity APIs are permitted.

## Minute 15–28: execute only the winning branch

- Score the bounded French workload and required training/validation rows.
- Keep the already measured US/India output fixed unless the new method has established
  a labeled-domain improvement and the combined change is explicitly intended.
- Recompute target ownership wherever changed scores compete. Include relevant rival
  pairs, not just the preferred S1 candidate.
- Preserve baseline predictions for untouched pairs/entities. Record all changes.
- Do not retrain every base model or rebuild the full candidate pipeline.
- Watch actual progress and GPU memory. Use one writer per score directory, fresh
  manifest-bound checkpoints, verified part counts, finite probabilities and pair-key
  coverage. A stale `finished` file is not proof of complete inference.
- Stop at the inference cutoff. Never export partially scored countries silently.

## Minute 28–33: select and validate

Produce a compact comparison against **scored V3**, not merely against fr08:

| Required comparison | Why |
|---|---|
| Held-out US/India macro-F change and country breakdown | Establish measurable behavior |
| French changed S1s, added and removed links | Establish scope, not correctness |
| Empty predictions, no-address share, target ownership | Catch unintended structural changes |
| CE coverage and missing-score counts | Catch incomplete inference |
| Hashes and exact model/input manifests | Preserve output identity |
| All S1 rows, valid IDs, no duplicates, match subset | Avoid invalid submission |

Inspect a small random sample of additions/removals for obvious serialization or
normalization mistakes. Do not hand-label from intuition, look up businesses online,
or tune the whole submission using a handful of examples.

A full-candidate official validator can exceed the helper pod's memory. Use the
already provisioned suitable worker or the supplementary streaming structural check
plus required ID validation. State which checks were actually run. Final packaging
still requires the exact scored candidate set, not a post-score top-15 list.

## Minute 33–40: last upload and fallback

1. Confirm one attempt really remains and the correct final-selection behavior.
2. Upload only a complete, validated file with a coherent source of new evidence.
3. Record its exact MD5, architecture and portal score.
4. If it does not improve, retain/select V3 at **0.982564**, subject to portal rules.
5. Do not replace a known better scored file merely because the latest run finished.
6. Complete the final package and documentation even if 0.99 is not reached.

If the hard deadline arrives sooner than these relative checkpoints, shorten the
experiment and preserve the upload buffer. **Submission correctness outranks finishing
an unvalidated new model.**

## Exact message to give Claude Code

> We have public V3 = 0.982564 and V3s_fr08 = 0.982529. Preserve V3. Verify whether
> one upload remains. We have at most 30–40 minutes. Stop blind France probability
> scaling, further seed searches, and repeats of support equality features already
> present in V3s. First check the current error decomposition and French CE coverage/
> duplicate provenance. If no coverage bug offers an immediate fix, run one tightly
> bounded experiment: use the existing task-fine-tuned multilingual CE to compare an
> uncertain candidate against one reliable, different matched target anchor for its S1.
> Select anchors/candidates without labels and with proper out-of-fold provenance;
> never use self-support or unconditional propagation. Pilot on labeled development
> data with realistic negatives, fit fusion without leakage, and evaluate the actual
> macro-F decision. Existing support equality features are not this neural evidence.
> Benchmark throughput first; reserve at least ten minutes for validation and upload.
> If the pilot shows no credible gain, stop. Only if it fails early, consider one
> canonical-text CE-view pilot using existing normalization and the same trained model.
> Do not download a new generic French model unless licensing, adaptation, complementary
> pilot gain and complete runtime all fit. Keep untouched predictions fixed and redo
> rival competition for changed scores. Use the last upload only for a complete,
> defensible candidate, report actual portal gain separately from validation, and
> preserve V3 as fallback. Target 0.99, but do not claim it is guaranteed. Update
> CURRENT_PROGRESS_AND_NEXT_STEPS.md with every result, rejection, hash and next action.

## Existing references and locations

- `CURRENT_PROGRESS_AND_NEXT_STEPS.md`: living handoff (historical status needs updating).
- `EXECUTION_PLAN_TO_DEADLINE.md`: broader experiment designs and package requirements.
- `analysis/score_audit/PORTAL_RESULTS.json`: earlier confirmed scores; add this fr08 result.
- `submissions/SUB-FINAL-V3_m2_ce_old_hi/`: measured best output.
- `submissions/SUB-FINAL-V3s_fr08/`: latest uploaded output, scored 0.982529.
- `submissions/SUB-FINAL-V3s_support.stack_eval.json`: existing support model results/features.
- `code/business_entity_resolution/src/neural/errtab.py`: current error decomposition code.
- `code/business_entity_resolution/src/neural/stack.py`: current stacker; preserve version identity.
- `code/business_entity_resolution/src/neural/ce.py`: existing fine-tuned CE inference.
- `analysis/score_audit/`: earlier experiments and negative results; avoid repeating them.

**This document is a proposed execution plan, not a report of experiments executed or
a promise that a particular technique will produce the required gain.**
