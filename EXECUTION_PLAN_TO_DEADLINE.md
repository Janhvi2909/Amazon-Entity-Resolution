# Execution plan: from 0.979604 toward 0.99

> **Live update, 2026-09-27 ~22:12 IST:** the user confirmed an actual portal score
> of **0.982564 for SUB-FINAL-V3_m2_ce_old_hi**, a **+0.002960** gain over SUB-N3.
> The M2/old-stage-2/review fusion below has therefore been implemented and tested
> on the portal by the other session. Remaining gap to 0.99: **0.007436**.
> V3b and V3c are distinct later artifacts with no portal score confirmed here.
> Remaining uploads need reconfirmation. Historical baseline/status statements below
> describe the plan when written; use `analysis/score_audit/PORTAL_RESULTS.json` for
> the latest confirmed score. Do not repeat completed M2 fusion experiments.

Written 2026-09-27, approximately **21:05 IST**. Hard deadline recorded in the handoff:
**23:59 IST today**. The user confirmed **three leaderboard uploads remain** during
this session. Recheck that count after every upload. This document supersedes stale
status/ETA information in `HANDOFF_CONTEXT.md`; it does not supersede the official rules.

## 1. What the target actually requires

The metric is **macro F0.5 per S1 business**, not classification accuracy or pair recall.

| Measurement | Value | What it establishes |
|---|---:|---|
| Best measured public leaderboard, SUB-N3 | **0.979604** | Performance on the hidden public test subset |
| SUB-N3 recorded validation | 0.98804925 | Held-out US/India training entities only |
| New SUB-R validation | **0.98821404** | A small measured US/India validation improvement |
| SUB-R leaderboard | **Unknown — not uploaded** | No test improvement established |
| Desired leaderboard | **0.990000** | Requires another **0.010396** |

Reaching 0.99 would remove approximately **51% of the remaining public-score loss**.
A +0.02 gain would reach 0.999604 and remove approximately 98% of that loss. These are
ambitious targets; no current experiment establishes that either can be reached tonight.
Do not turn validation gains, model probabilities, or an estimated France score into
claims about actual test performance.

France is 259,452 / 1,732,544 = approximately **14.98% of full-test S1s**. If all the
improvement came from France, +0.010396 overall would require approximately **+0.0694
France macro F0.5**, assuming the public subset has the same country mix. That mix is
not known. The handoff's France estimate of 0.94–0.95 is also unverified. Under that
estimate, even perfect France predictions would not supply the entire desired gain:
US/India would need improvement as well.

**Strategy:** spend the remaining time on wider candidate coverage and errors outside
the existing neural uncertainty band, particularly in France. Small stacker changes
can support that work, but cannot substitute for it.

## 2. Current architecture and where to add work

```text
Raw S1/S2/S3 records
  -> normalization + learned aliases + France canonicalization
  -> sparse candidate retrieval, separately by country/source
  -> numerical, text, house-number, legal-form and competition features
  -> M-v5 cross-fitted LightGBM stage 1 + stage 2
  -> mDeBERTa-2 / BGE / India-only MuRIL on uncertain pairs
  -> neural-score LightGBM stacker
  -> exclusive target assignment + gated expected F0.5
  -> matching_results.tsv + exact scored candidate set

Additions to prioritize:
  A. M2 deeper retrieval and new CE scores
  B. retain old M-v5 stage-2 evidence when integrating M2 stage 1
  C. neural review of confidently scored French pairs
  D. evaluate recoverable low-score/below-band misses
  E. only if time permits: additional dense retrieval with real negative competition
```

M2 already exists: `NORM-v4t + BLK-v4b@40n20 + FEAT-v5 + MATCH-v7`.
Its full stage 2 exceeded node memory; its completed stage 1 is available as `M2P1`,
where `p2` is simply a copy of `p1`. **Do not mistake M2P1 p2 for a trained stage-2 score.**

Pair blocking recall rose from approximately 98.31% to 98.91% on validation. This is
**pair recall**, not a 98.91% macro-F ceiling. Compute the perfect-matcher macro score
on the actual candidate set before quoting a macro ceiling.

## 3. What this session completed

### Ready candidate: SUB-R, all-feature neural stack on M-v5

- Adds the available pair features and global base-p2 competition to the existing
  CE2/BGE/MuRIL stack: **77 stacker inputs, 704 LightGBM trees**.
- Trains on REST, early-stops on ES, selects gate **0.45 on ES**, evaluates on EVAL.
- Validation: **0.9882140353**; India **0.9860742814**; US **0.9896405594**.
- Paired replay comparison: +0.0001620; 1,057 eval entities improve, 601 worsen.
  Approximate paired 95% interval: +0.0001084 to +0.0002157. This describes sampling
  uncertainty on the same validation domain, **not France or test uncertainty**.
- Outside the original p2 uncertainty band, M-v5 stage-2 predictions are retained.
- Local outputs: `submissions/SUB-R_allfeatures_audit/`.
- Full official validator including `--check-ids`: **PASS**. Local output hashes equal
  the validated remote outputs.

| File | MD5 |
|---|---|
| `matching_results.tsv` | `1215dda24020dcdd8e7226f4dda4e03d` |
| `candidate_pairs.tsv` | `9ed7d62d9732dc51644a8258f2cf03f7` |

There are 5,734,515 predicted links and **83,311,838 scored candidates**. Every one of
the 1,732,544 test S1s has a row. The full candidate file is approximately 1 GiB.

Only **13,950 test S1s change** versus SUB-N3: France 5,447, US 5,404, India 3,099.
That is approximately 0.805% of full-test entities. Even if every changed entity went
from score zero to score one, this variant alone could not add 0.01 to full-test macro
F0.5. Public-subset proportions may differ. **Keep it as a validated alternative and
integration component; do not spend an upload merely because its validation is higher.**

### Experiments already tested — do not repeat tonight

| Experiment | Finding | Action |
|---|---|---|
| Numeric-only neural stack additions | Validation 0.98811669 | Below SUB-R |
| Numeric + global base-score competition | 0.98812180 | Below SUB-R |
| Inverse-sqrt entity-size training weights | 0.98819616 | Below SUB-R |
| Inverse entity-size weights | 0.98817826 | Below SUB-R |
| Train stacker on half ES, stop on other half | 0.98805055 | No useful gain |
| Exact expected F0.5 instead of ratio approximation | About +0.000009 | Too small to prioritize |
| Exact target name+address propagation | 197 eval additions, 169 true; +0.000013 | Too small to prioritize |
| Name-only target propagation | 2,047 additions, 627 true; -0.000487 | Reject |
| Full-pool vs eval-only target exclusivity | Difference below 0.00001 | Fix protocol, not the source of the LB gap |

Earlier handoff experiments also rejected blanket nearby-house filters, large French
legal dictionaries, all-country MuRIL, and generic gate tuning. See `method_result.md`.

## 4. Immediate priority: make the existing GPU work usable

**Owner:** another session, confirmed by the user. Do not kill/recreate its GPU pod,
delete score directories, or launch competing GPU inference without coordinating.
This session's CPU jobs are complete. Its original CPU reservation ended at 21:15 IST;
that is not permission to occupy nst-n7 beyond then.

Latest read-only inspection, approximately 20:59 IST:

- `ce2_m2`, `bge_m2`, `muril_m2` reported **finished**.
- `ce2_hi` had a stale part-0 progress record, CUDA OOM errors, and a
  `FileNotFoundError` for `/n6/ce2_hi/scores/part-0000.tmp`.
- A previous queued script could delete/rebuild the same score directory. These logs
  are not proof of a complete review. The owning session must inspect live state.

Read-only status commands:

```bash
kubectl -n amlc-ber-jy get pods
kubectl -n amlc-ber-jy exec ber-gpu -- sh -c \
  'cat /n6/ce2_m2/PROGRESS.txt /n6/bge_m2/PROGRESS.txt /n6/muril_m2/PROGRESS.txt /n6/ce2_hi/PROGRESS.txt; tail -30 /n6/ce2_hi.log'
```

Before combining score parts, the GPU owner should:

1. Reconcile all live processes and queued restart scripts. One writer per output
   directory; use a **new uniquely named directory** for a restarted review.
2. Open every Parquet part and its marker. Check row count, expected pair keys,
   finite scores, probability range, and duplicate-key disagreements.
3. Verify coverage against the intended inference manifest. A `finished` log is
   insufficient if it belongs to an earlier input selection.
4. Confirm whether the high-confidence band is the original 3,560,400-row version,
   the 2,024,643-row version seen in a log, or another subset. Record its split,
   country, role, and score-range counts. Never infer this from an old filename.
5. Reuse all completed M2 inference; do not re-score it unnecessarily.
6. If recovery is needed, run sequentially at a fitting batch size. Current `ce.py`
   multiplies inference batch size by six, so `--bs 64` means up to 384 examples per
   inference batch. CUDA autocast does not make FP32 model storage disappear.
7. For the review, prioritize all intended French test pairs plus a representative
   training REST sample, ES and EVAL. Do not deploy a high-score stacker without
   enough corresponding held-out training-domain scores to test it.

**Checkpoint pitfall:** `ce.py` currently accepts a cached part based on row count.
That does not prove the input pairs/order are unchanged. For a resumed/reordered band,
use a fresh directory or add a manifest digest of ordered pair keys and model/tokenizer
configuration. Do not reuse same-length chunks from a different input.

## 5. Highest-priority model experiment: M2 + old stage 2 + CEs

**Status: proposed integration, NOT yet implemented or measured in this session.**
The existing `stack_refine.py` is runnable for M-v5; it is not a turnkey M2 fusion script.

Why: replacing M-v5 with M2P1 discards a trained stage-2 model, including consensus and
competition information. Deeper candidates should be added without needlessly losing
that evidence on shared pairs.

### Build the fusion inputs

For each `(split, q_rid, t_rid)` in the union of relevant old/new candidates, assemble:

- M2 stage-1 probability;
- M-v5 stage-1 and **stage-2** probabilities where available;
- explicit missing-old-pair indicator;
- CE2, BGE and India-only MuRIL, with missing values outside actual coverage;
- FEAT-v5 numerical/address components, plus useful existing string/competition features;
- best rival base probability for the target, margin, and S1 probability sum/count;
- CE target competition and S1 ranking features;
- optionally old SUB-N3 stack score on shared pairs, but only with a defensible
  out-of-fold construction for stacker-training rows. Do not feed in-sample predictions
  from a REST-trained stacker back as if they were held-out features.

Verify row-ID identity between NORM-v3t and NORM-v4t using `entity_id` mappings before
joining. Never assume that positional IDs remain compatible after a preprocessing change.

### Memory-bounded implementation requirements

- Never join 253 million rows to all CE/feature columns at once.
- Build compact pair-key tables and stream feature parts into those keys, as in
  `neural/stack_refine.py`.
- For decision scores, preserve every row that could be changed by a learned model.
  **Do not drop a low-M2-probability pair if it has an old CE score or strong old p2.**
  Filtering only `M2 p1 >= .02` would silently discard some potentially recoverable pairs.
- Compute competitor summaries over the appropriate complete candidate pool before
  selecting the compact scoring subset. Include confident rivals outside the CE band.
- Keep a complete manifest of every pair actually scored for the final candidate export.
- Treat missing CE values explicitly; do not substitute zero for an unscored pair.
- Different-model or conflicting scores for the same key should be audited, not silently
  resolved by whichever file `rglob()` encounters first.

### Required comparisons

| Variant | Purpose |
|---|---|
| Frozen SUB-N3 replay | Reproduce the known reference |
| SUB-R | Measured all-feature M-v5 reference |
| M2P1 + completed new/old CEs | Isolate deeper-retrieval architecture |
| Same + old M-v5 stage-2 feature | Test whether retained old knowledge helps |
| Best fusion + confident-pair review | Isolate the review's contribution |

Fit on REST, early-stop on ES, select the gate on ES, and report EVAL overall and by
country, with per-entity paired deltas. Keep the legacy eval protocol as an explicitly
labeled comparator if needed; separately audit all-entity competition. FIT CE predictions
may be missing/in-sample and REST stacker predictions are in-sample, so an all-pool result
alone is not magically unbiased. Preserve role provenance.

Do not require a validation win to assert a France win, or infer a France win from it.
Use validation to detect regressions in labeled domains and the leaderboard to test the
remaining hidden-domain hypothesis. The public board is also only a subset, not the
private final score.

## 6. France review: the branch with the strongest historical evidence

**Evidence:** the first multilingual CE stack improved the public LB by approximately
0.0108 while US/India validation improved by approximately 0.0027. This suggests a
large France contribution; it does not directly measure France's score.

**Status:** inference attempted by the other session; completeness must be verified.

1. Score confident French pairs with the trained multilingual CE. Add BGE only if
   time and GPU capacity permit; reuse existing BGE scores first.
2. Train the extended stack on a corresponding mixture of uncertain and confident
   REST training pairs, with held-out ES/EVAL scores. Record any subsampling of the
   confident group; account for sampling when calibrating probabilities.
3. Compare three behaviors: original confident scores unchanged; extended stack over
   all CE-scored pairs; a conservative review policy learned/tuned on ES.
4. In particular inspect highly confident tree matches contradicted by the CE and
   explicit house/legal evidence. **CE disagreement is not a label.**
5. Keep source caps experimental: maxima 5 S2/6 S3 were observed in train, not stated
   as hard test constraints. Apply any cap consistently in validation and test and
   report its effect. Do not silently add a test-only rule.
6. Preserve a country-spliced alternative: best labeled-domain improvement for US/India,
   with either frozen SUB-N3 France or reviewed France. This helps isolate a risky
   unseen-country change without refitting everything.

For each candidate, report changed S1 counts, additions/removals, empty predictions,
no-address share, number conflicts, legal conflicts, and raw examples. These checks
can identify obvious bugs; they cannot score French correctness.

## 7. Next recall experiment: below-band and absent candidates

**Status: requires additional implementation and measurement.**

For SUB-R validation there are **52,442 missed links** and 1,611 false-positive links.
Of the missed links, **29,760 are absent from the retained `M-v5 p2 >= .02` table**.
That last number combines blocking misses and low-score candidates; do not call all of
them blocking misses. Split them by joining against the complete original score table.

### Cheap first diagnostic

- Partition true misses into: absent candidate; candidate below .02; candidate in
  CE band; lost under exclusivity; dropped by final set decision.
- Report by country, source, missing address, true match count and name multiplicity.
- Measure the maximum macro-F improvement of perfectly fixing each partition, not
  just its number of links. A first recovered link can be worth much more than a fifth.
- Compare these partitions after M2; do not design the next retrieval experiment from
  misses that M2 already fixes.

### Selective CE band expansion

If low-score candidates contain enough remaining loss, expand the band using signals
available equally in training and test, such as strong address agreement, strong name
agreement with missing address, or disagreement between M2 p1 and old M-v5 p2.

- Pilot on ES with both positive and negative candidates and normal target competition.
- Include train REST/ES/EVAL and all test countries under a documented selection rule.
- Measure precision of recovered links and macro-F change before full inference.
- Do not score only known validation misses and report that as a deployable experiment:
  that uses labels to select candidates and ignores the false-positive workload.

### Dense retrieval, only if the timing works

The earlier multilingual-e5-small diagnostic recovered 38% of residual blocking misses
at top 5, but it queried **known missed entities**. That is a diagnostic recovery rate,
not a measured production gain. The handoff's +0.0005–0.0008 estimate is unverified.

A real experiment must define a label-free query policy, retrieve candidates including
negatives, merge/deduplicate them with existing candidates, compute features, score them,
and redo competition/set decisions. Cache embeddings. A selectively targeted pilot is
reasonable; a fresh all-record embedding run late tonight is not a safe deadline bet.

## 8. Upload policy: use three attempts to learn, not three arbitrary variants

Portal submission is performed by the team unless an authorized portal integration is
available. Do not claim an upload or score without its actual result.

1. **First attempt:** the strongest complete, validated wider-coverage/France-review
   candidate. Record exact filename, MD5, architecture and changed-entity counts before
   upload. Keep the 0.979604 file intact.
2. **Second attempt:** respond to the first result. If a combined change regresses, use
   a country-spliced or review-off variant to separate France from US/India changes.
   If it improves, test the next genuinely distinct validated addition.
3. **Third attempt:** reserve for a final evidence-backed candidate or a correction.
   Confirm how the portal chooses the final submission: best, last, or explicit selection.
   Do not assume that uploading a worse file automatically preserves the best one.

Do not mechanically add historical gains together. Methods often fix the same pairs.
Tiny validation improvements alone are a poor reason to spend a scarce upload when
the principal uncertainty is an unlabeled country.

## 9. Timing and stop conditions

These are decision checkpoints, not promised runtime estimates. Adjust after checking
current throughput and actual portal turnaround.

| IST checkpoint | Required outcome |
|---|---|
| 21:05–21:20 | Resolve ownership/status; validate completed M2 CE parts; decide whether confident review can finish |
| 21:20–21:50 | Evaluate M2 baseline and fusion retaining old stage 2; review branch if ready |
| Around 21:50–22:10 | First meaningful validated upload and public-score feedback |
| 22:10–22:40 | One targeted follow-up based on error decomposition and upload feedback |
| **22:45** | Stop launching substantial new training/retrieval; finish only work with a credible short ETA |
| **23:10** | Choose final model/file; preserve exact selected artifacts |
| 23:10–23:35 | Final validation, reproduction README, methodology, hashes, package |
| **23:35 onward** | Upload/select final package with time for transfer/rejection fixes |

If confident review remains broken after the first checkpoint, finish the M2 fusion
without it and retain the known good submission. Do not wait indefinitely on one branch.
If dense retrieval cannot finish training/test scoring and validation before 22:45,
defer it. A trained model without complete test outputs is not a submission.

## 10. Exact files and infrastructure for continuation

### Repository and local artifacts

Root: the repository root.

| Path | Contents/status |
|---|---|
| `HANDOFF_CONTEXT.md` | Earlier full architecture/infrastructure context; live status is stale |
| `method_result.md` | Experiment history, including this session's measured results |
| `analysis/score_audit/INTER_SESSION_FINDINGS.md` | Short coordination note for the other session |
| `analysis/score_audit/experiments.json` | Three unweighted stacker ablations |
| `analysis/score_audit/paired/paired.json` | Paired entity comparison and error counts |
| `analysis/score_audit/paired/{misses,false_positives}.parquet` | Held-out error pair keys |
| `analysis/score_audit/test_changes/` | Label-free test diffs and raw country samples |
| `analysis/score_audit/weighted/results.json` | Weighting/training-regime ablations |
| `analysis/score_audit/all_features.txt` | SUB-R trained stacker |
| `analysis/score_audit/all_features_test_updates.parquet` | SUB-R test probability updates |
| `submissions/SUB-N3_ce2_bge_murilIN/` | Frozen measured leaderboard best |
| `submissions/SUB-R_allfeatures_audit/` | New complete validated candidate; unsubmitted |

Temporary local analysis environment: `/private/tmp/ber-audit-venv/bin/python`.
It has Python 3.11, Polars 1.44.2, LightGBM 4.7.0, NumPy 2.4.6, SciPy 1.17.1,
RapidFuzz and anyascii. The remote training job used Python 3.12 and Polars 1.41.2.
Record actual versions for reproduction; do not assume the repository's old CPU-only
requirements describe the neural GPU environment.

### Existing remote caches

Namespace **`amlc-ber-jy`** only. Do not touch `amlc-ber-phase3c`.

```text
nst-n7 / PVC / CPU:
  /work/full/prep/NORM-v3t
  /work/full/prep/NORM-v4t
  /work/full/runs/NORM-v3t__BLK-v4b@20__FEAT-v4__MATCH-v5
  /work/full/runs/M2P1
  /work/full/feat/NORM-v3t__BLK-v4b@20__FEAT-v4
  /work/full/feat/NORM-v4t__BLK-v4b@40n20__FEAT-v5
  /data/ber/code_0927203238/src
  /data/ber/final_ce/{ce2,ce_bge,ce_muril_in}/scores
  /data/ber/codex_audit/                 compact bands, models, updates, metrics
  /data/ber/codex_refine_submission/     validated SUB-R output files
  /data/ber/codex_closure/               rejected propagation diagnostics

nst-n6 / ber-gpu:
  /n6/ce2_m2/scores
  /n6/bge_m2/scores
  /n6/muril_m2/scores
  /n6/ce2_hi/scores                     MUST verify completeness/current owner
```

`/work` is the mounted hostPath `/var/lib/amlc-ber/work`; `/data` is the RWO PVC.
Do not attach the PVC to the GPU node. GPU host data lives under `/n6` separately.

AWS profile recorded as working: **`amlc-personal`**, account **<AWS_ACCOUNT_ID_2>**,
bucket **`<S3_BUCKET_2>`**. The older account in `CLAUDE.md` had expired auth.
[Current bucket console](https://us-east-1.console.aws.amazon.com/s3/buckets/<S3_BUCKET_2>?region=us-east-1&tab=objects).
This session's SUB-R artifacts are verified locally and on PVC; **no S3 upload of SUB-R
was performed by this session**.

### Existing executable tools added by this session

All under `code/business_entity_resolution/src/neural/`:

- `stack_audit.py`: saved SUB-N3 replay and competition audit.
- `stack_refine.py`: M-v5 all-feature/numerical stacker experiments and test updates.
- `closure_audit.py`: target-propagation diagnostic; not a shipped rule.
- `exact_f_audit.py`: exact expected-F comparison; tested against exhaustive enumeration.
- `macro_weight_audit.py`: weighting and ES-only-training ablations.
- `paired_audit.py`: paired validation comparison and error keys.
- `profile_changes.py`: streaming raw test-change report; no test correctness claims.
- `export_refine.py`: matching export and complete scored candidate export.

## 11. Commands that are ready now

Run from the repository root. These reproduce/export **SUB-R**, not the proposed M2 fusion.

### Verify the ready local files

```bash
md5 -r submissions/SUB-R_allfeatures_audit/matching_results.tsv \
  submissions/SUB-R_allfeatures_audit/candidate_pairs.tsv
cat submissions/SUB-R_allfeatures_audit/VALIDATOR_IDS.txt
```

### Reproduce the stacker experiments on an appropriately provisioned CPU worker

Deploy `neural/stack_refine.py` with the repository source on `PYTHONPATH`, then run:

```bash
export PYTHONPATH=/data/ber/code_0927203238/src
export POLARS_MAX_THREADS=12 OPENBLAS_NUM_THREADS=1
python -u /data/ber/stack_refine_audit.py \
  --run /work/full/runs/NORM-v3t__BLK-v4b@20__FEAT-v4__MATCH-v5 \
  --prep /work/full/prep/NORM-v3t \
  --feat /work/full/feat/NORM-v3t__BLK-v4b@20__FEAT-v4 \
  --ce-root /data/ber/final_ce \
  --out /data/ber/codex_audit_reproduction --threads 16 --test
```

The standalone deployed script is present at `/data/ber/stack_refine_audit.py`.
This command writes a **new** output directory. Do not reuse a compact-band cache
after changing its upstream model, CE selection, features or normalization.

### Reproduce SUB-R exports locally from the already-downloaded caches

```bash
export PYTHONPATH=code/business_entity_resolution/src
export POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=1
/private/tmp/ber-audit-venv/bin/python -u \
  code/business_entity_resolution/src/neural/export_refine.py \
  --run /private/tmp/claude-503/stack/run \
  --prep /private/tmp/claude-503/stack/prep \
  --updates analysis/score_audit/all_features_test_updates.parquet \
  --out /private/tmp/sub-r-reproduction --gate .45 --candidate
```

These `/private/tmp` caches are convenient session artifacts, not a self-contained
final reproduction package. The final README must explain how the upstream artifacts
are generated from the original challenge files.

### Reliable transfer

```bash
bash infra/k8s/xfer.sh get ber-util \
  /data/ber/codex_refine_submission/matching_results.tsv \
  /private/tmp/sub-r-matching-copy.tsv
```

The script checks each chunk and the complete MD5. `kubectl cp` failed on a 111-MiB file
in this session. For S3-mediated transfer, use unique object names, verify checksums,
and do not print presigned URLs into logs or documentation.

## 12. Final submission and reproducibility checklist

1. Freeze the selected matching file, model identifiers, selection rule, logs and MD5.
   Do not overwrite prior validated submissions.
2. Export **the exact set of pairs the model scored**, not a top-15 list pruned after
   scoring. Current old `neural/stack.py` uses the latter. SUB-R's exporter fixes this.
   For a union/fusion model, specify the complete scoring pipeline and union provenance.
3. Validate full S1 coverage, no repeated IDs per row, existing S2/S3 IDs, matching
   subset of candidates, country alignment, and at-most-one-owner target behavior.
4. Run the official validator on a sufficiently large CPU worker. It retains Python
   mappings of all candidate lists: the full 83M-pair `--check-ids` run OOMed at an
   8-GiB limit and passed with a 40-GiB limit. Do **not** repeat it in `ber-util` or on
   the 16-GB laptop. On a small machine, validate matching with `--check-ids` separately
   and use a streaming candidate audit; retain the already-passed full remote report.
5. Fill `Documentation_template.md` with the final architecture and actual measured
   results. Do not label validation or inferred per-country estimates as test scores.
6. Update the pipeline README: exact data-to-prep-to-block-to-model-to-neural-to-export
   commands; CE checkpoint/model acquisition and licenses; role splits; hardware;
   all inference inputs; dependency versions; deterministic seeds; cache manifests.
7. Pin the neural dependencies as well as the existing CPU dependencies. The current
   requirements file does not contain the full PyTorch/Transformers environment.
   Ensure the source package includes any France resources referenced at runtime.
8. State that only provided challenge records were used. Model documentation/research
   is allowed context; external business identity/address lookup or augmentation is not.
9. Package the exact file selected on the portal. `make_submission.py` accepts
   `--output-dir`, so changing the root `output/` folder is unnecessary:

   ```bash
   python3 make_submission.py --team "ACTUAL_TEAM_NAME" \
     --output-dir submissions/ACTUAL_SELECTED_SUBMISSION
   ```

   Replace both placeholders. **Run this packager on a sufficiently large worker for
   full candidate lists**, because it invokes the memory-heavy official validator.
   Its existence does not mean the documentation and neural reproduction instructions
   are already finished; they are still required work.

10. Check archive layout and available disk space. Approximately 9.8 GiB was free on
    the laptop before the final local candidate export; recheck rather than accumulating
    multiple full-output copies. Confirm the portal/package upload completed before
    the deadline and record which submission is selected for final evaluation.

## 13. Research basis and boundaries

- [Ditto: deep entity matching](https://www.vldb.org/pvldb/vol14/p50-li.pdf) supports
  pretrained pair models, domain-aware inputs and difficult-example augmentation.
  The current architecture already uses a related cross-encoder approach; citing the
  paper is not evidence that another model will add a leaderboard point here.
- [Transitivity constraints in entity matching](https://arxiv.org/abs/2104.12589)
  cautions that naive closure can hurt precision. The measured name-only propagation
  failure here is a direct reason to reject it.
- [mDeBERTa model card](https://huggingface.co/microsoft/mdeberta-v3-base) and
  [BGE reranker documentation](https://bge-model.com/tutorial/5_Reranking/5.2.html)
  describe the multilingual models already being used. Retain exact model revisions
  and license records in the final reproduction materials.

The best remaining path is **complete wider retrieval + preserve old stage-2 evidence +
validate the French confident-pair review**, then use actual leaderboard feedback.
This is a prioritized plan toward 0.99, not a guarantee of 0.99.
