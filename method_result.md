# Method & Results Log — Business Entity Resolution

## Confirmed portal result — SUB-FINAL-V3, 2026-09-27

The user reports **0.982564 public leaderboard macro F0.5** for
`submissions/SUB-FINAL-V3_m2_ce_old_hi`, **+0.002960** over SUB-N3 (0.979604).
Architecture: M2 deeper retrieval/stage-1 scores + old M-v5 stage-1/stage-2 scores +
mDeBERTa-2/BGE/MuRIL-India + confident-pair review + LightGBM stacker (444 trees),
exclusive gated expected-F (gate .55, selected on EVAL for this artifact).
Local validation 0.9896525533 is a separate measurement. Gap to target public score
0.99 is 0.007436. Do not attribute this portal score to later V3b/V3c artifacts.

A record of every experiment we run: what method we used, how we measured it, and what score it got. Add one entry per run, and keep the summary table in sync. The final `Documentation_template.md` is written from this file.

## Independent deadline audit — 2026-09-27 ~21:00 IST

**Leaderboard and validation are different measurements.** Current measured public LB is
SUB-N3 **0.979604**. All numbers below are held-out **US/India training validation macro
F0.5**, not test scores. France has no labels; no new leaderboard gain is established.
User clarified that three portal uploads remain. The other session owns GPU inference/M2.

| Experiment | Validation F0.5 | Decision |
|---|---:|---|
| SUB-N3 reported baseline | 0.98804925 | Public LB 0.979604 |
| SUB-N3 saved-model replay | 0.98805200 | Minor ordering/tie discrepancy vs original; recorded explicitly |
| Replay with all-entity target competition | 0.98805127 | Eval-only competition issue is immaterial to this result |
| Existing CEs + numerical features | 0.98811669 | Small improvement |
| Existing CEs + numerical/global base-p2 context | 0.98812180 | Small improvement |
| **SUB-R: existing CEs + all pair/global base-p2 features** | **0.98821404** | Separate validated candidate, not uploaded |
| SUB-R features, sqrt inverse entity-size training weight | 0.98819616 | Below unweighted SUB-R |
| SUB-R features, inverse entity-size training weight | 0.98817826 | Below unweighted SUB-R |
| Stacker fitted on half ES, stopped on other half | 0.98805055 | No benefit from matching mean-fold base-score training regime |
| Exact expected F0.5 decision on saved SUB-N3 replay | 0.98806115 | Only +0.000009; not shipped |

SUB-R trains a LightGBM stacker on REST, early-stops on ES, and selects gate **0.45 on
ES**, then evaluates once on EVAL (441,521 S1s). India **0.98607428**, US **0.98964056**;
precision 0.99868296, recall 0.96607798. It retains M-v5 stage-2 outside the uncertain band.
Base-score context includes confident rival candidates, not just CE-band rivals.
Code: `neural/stack_refine.py`; model/compact bands and updates on PVC
`/data/ber/codex_audit/`; local reports under `analysis/score_audit/`.

Conservative target-to-target retrieval was also tested, without using labels to select
links: exact normalized full name + address added 197 eval links, 169 true (85.8%),
improving the all-pool SUB-R diagnostic by only 0.0000133. Name-only expansion added
2,047 links with 627 true (30.6%) and **lost 0.0004872**. Neither was shipped. These full-pool
diagnostics have in-sample REST stacker scores/missing FIT CE; they are not fully unbiased.

**Test export:** SUB-R has 5,734,515 links (France 813,701; US 2,234,943; India 2,685,871).
Versus SUB-N3: France +3,584/-2,027 links, US +4,337/-1,129, India +1,850/-1,295.
Affected S1s: France 5,447, US 5,404, India 3,099. These are label-free changes, not
correctness claims. Only 13,950/1,732,544 test entities change (~0.805%), so this variant
alone cannot improve full-test macro F0.5 by 0.01; public-subset changes may differ.

Matching MD5: `1215dda24020dcdd8e7226f4dda4e03d`. Full official validation including
`--check-ids` passed on the CPU node. An initial helper-pod check and an 8-GiB validation
job OOMed; the successful dedicated check used a 40-GiB limit. GPU jobs were untouched.

**Candidate provenance correction:** `stack.py`'s old top-15 post-score export does not
represent the exact model-scored set required by the official statement. SUB-R's new
`neural/export_refine.py` exports all **83,311,838** original scored test pairs in chunks.
This is an audit/package fix, not a leaderboard improvement. Frozen prior files remain intact.

Research reviewed only methodology/model documentation, with no external business data:
[Ditto](https://www.vldb.org/pvldb/vol14/p50-li.pdf),
[transitivity caution](https://arxiv.org/abs/2104.12589),
[mDeBERTa model card](https://huggingface.co/microsoft/mdeberta-v3-base),
[BGE documentation](https://bge-model.com/tutorial/5_Reranking/5.2.html).

---

## 1. Task context

- **Goal:** for every Source 1 (S1) entity, list all Source 2 / Source 3 (S2/S3) records that refer to the same real-world business.
- **Fields:** `entity_id`, `business_name`, `business_address`, `country`. There are no shared IDs across sources.
- **Metric:** F_0.5, averaged per S1 entity (macro). Precision counts twice as much as recall.
  - If an S1 entity truly has no matches, predicting an empty list scores **1.0**, and predicting anything scores **0.0**.
  - If an entity has matches and we predict an empty list, it scores **0.0**.
  - Formula: `F_0.5 = 1.25·P·R / (0.25·P + R)`
- **Deliverables:** `output/matching_results.tsv` (this is scored) and `output/candidate_pairs.tsv` (the exact candidate set the model scores). Every ID in the matches file must also appear in the candidates file.
- **Constraints:**
  - The final model must be MIT or Apache 2.0 licensed, with at most 8B parameters.
  - No external data, APIs, or geocoding.
  - `country` is an open set. The test data includes **France**, which never appears in training.

---

## 2. Dataset facts (measured 2026-09-25)

| Split | Source | Rows | US | India | France | Empty address |
|---|---|---:|---:|---:|---:|---:|
| train | S1 | 2,206,821 | 1,323,633 | 883,188 | — | 0 |
| train | S2 | 5,034,616 | 3,016,817 | 2,017,799 | — | 168,967 |
| train | S3 | 5,285,603 | 3,170,056 | 2,115,547 | — | 175,916 |
| test | S1 | 1,732,544 | 663,106 | 809,986 | 259,452 | 0 |
| test | S2 | 4,887,273 | 1,871,330 | 2,312,565 | 703,378 | 129,408 |
| test | S3 | 5,082,316 | 1,945,701 | 2,405,000 | 731,615 | 136,098 |

No record in any file has an empty `business_name`.

**Training ground truth:**

- 2,206,821 S1 rows, one per S1 entity.
- **123,247 singletons (5.58%)**: S1 entities with no match.
- Matches per S1 entity: mean 3.46, max 11.
  - Distribution: 0→123,247 · 1→119,157 · 2→375,212 · 3→530,841 · 4→484,115 · 5→321,957 · 6→164,868 · 7→63,968 · 8→18,680 · 9→4,205 · 10→534 · 11→37
- Links: 3,693,619 go to S2 and 3,944,746 go to S3, for 7,638,365 in total.
- **Every S2/S3 record is linked to at most one S1 entity.** No ID appears under two S1 rows.
- Unmatched records, which act as distractors: **1,340,997 in S2 (26.6%)** and **1,340,857 in S3 (25.4%)**.

### What this means for method design

1. **One-to-many assignment.** Each S2/S3 record belongs to at most one S1 entity. When several S1 entities claim the same record, we can keep only the highest-scoring one. This is a free precision boost.
2. **Singletons are rare but costly.** They make up only about 5.6% of entities. Each one scores 0 if we predict even one false match, so we should consider a separate "has any match?" gate.
3. **A quarter of S2/S3 records match nothing**, so blocking has to cope with many distractors.
4. **Most entities have 2 to 5 matches.** A top-k cutoff that is too tight will cost recall, while one that is too loose will cost precision. The cutoff should be tuned against F_0.5, not set by hand.
5. **France appears only in test.** Features must not depend on language- or country-specific rules learned only from US and India data. Text normalisation should also handle French (accents, "rue", "avenue", "SARL", "SAS").
6. **About 3% of S2/S3 records have no address.** Blocking and features need a name-only path for these.

---

## 3. Evaluation protocol

_Fix this protocol before the first run, then keep it unchanged so that runs can be compared._

- **Development sample (DEV-10):** 10% of train S1 entities (seed 42), all of their true S2/S3 matches, and 10% of the distractor (unmatched) S2/S3 records, which keeps the full data's matched-to-distractor ratio. Size: 220,682 S1 records, 1,031,969 S2/S3 records and 763,784 true links. Every blocking version below is measured on DEV-10 on the laptop (12 threads).
- **Validation split (full data, `run_pipeline.py`):** train S1 entities are split once with seed 42 into fit 30% / early stopping 5% / **eval 20%** (441,521 entities) / rest 45%. Every entity stays in the candidate search, so targets are contested as densely as at test time. Aliases and region merges are learned without the eval entities' links. Stage 1 and stage 2 are both cross-fitted over the fit entities (3 folds); eval, rest and test rows get the mean of the fold models, so eval rows are scored exactly like test rows. The decision rule and its parameter are chosen on the eval slice.
- **Blocking metrics:**
  - Pair recall: the share of true links found among the candidates.
  - Mean candidates per S1 entity.
  - Reduction ratio.
- **Matching metrics:** macro F_0.5, macro precision, and macro recall on the validation set, reported separately for singletons and non-singletons, and for each country.
- **Threshold:** tuned to maximise validation F_0.5, with the chosen value recorded.
- **Format check:** run `utils/validate_submission.py` on every run that produces a submission.

---

## 4. Results summary

### 4.1 Blocking versions (candidate generation), side by side

All rows are measured on DEV-10. **Pair recall** is the share of true links that appear among the candidates, and is the upper bound on final recall. **All found** is the share of S1 entities whose true matches are all among the candidates. Top-K is per target source (S2 and S3 separately), plus a name-only pass over targets with no address from BLK-v2 onwards.

| Version | Date | What changed | Pair recall | All found | Cand. / S1 | Time | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| BLK-v1 | 2026-09-25 | IDF-weighted sparse token search. Features: name tokens, address tokens, 4-char prefixes, consonant skeletons, first/last 3 digits of house numbers. df cap 1%, top-30. | 97.66% | 92.76% | 60.0 | 26 s | Baseline |
| BLK-v2 | 2026-09-25 | + joined-name token, name character 3-grams, `shree/shri/sri` merged, name-only pass (top-5) for targets with no address | 99.31% | 97.71% | 69.1 | 135 s | Recall up, but cost grows about 100× on full data |
| BLK-v3 | 2026-09-25 | v2 with a fixed df cap (equal to 3,000 / 6,000 on full data) in place of the 1% cap | 97.74% / 98.49% | 93.24% / 95.30% | 69 | 27 / 30 s | **Rejected**: too much recall lost |
| BLK-v4 | 2026-09-25 | v2 + search split by region (tail tokens of S1 addresses, region merges learned from training links) | 99.48% | 98.20% | 66.9 | 48 s | Noisy region keys (`rd`, `nagar`) |
| **BLK-v4b** | 2026-09-25 | v4 + region keys limited to tokens that sit mostly at the end of an address: 16 Indian states, 37 US states; one learned merge (Telangana ↔ Andhra Pradesh) | **99.52%** | **98.35%** | 65.9 | **43 s** | **Current** |
| BLK-v4b @10 | 2026-09-25 | same, top-10 per source | 99.10% | — | 25.9 | — | Cheaper option for the matcher |
| **BLK-v4b@20, FULL train** | 2026-09-25 | BLK-v4b top-20 on the **full** data (2.2M S1 x 10.3M targets); eval slice shown | **98.32%** | 94.53% | 47.5 | 703 s (EC2, 8 cores) | Denser competition than DEV-10: -1.1 points. Targets without address: 88.7%. Recall still rising at k=20 |

### 4.2 End-to-end versions (validation F_0.5)

Unless noted, the scores are on the DEV-10 **evaluation slice**: 20% of the DEV-10 S1 entities (seed 42), never used for fitting or early stopping. The model is fitted on 70% and early-stopped on 10%. The competition features use all DEV-10 S1 records, as they would at test time.

| ID | Date | Method (short) | Blocking recall | Candidates / S1 | Val F_0.5 | Val P | Val R | Singletons / non-singletons | LB F_0.5 | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| M-v1 | 2026-09-25 | NORM-v2 + BLK-v4b (top-20 per source) + 44 features + LightGBM (log loss) + exclusive assignment + threshold 0.725 | 99.45% | 46.9 | 0.9882 | 0.9954 | 0.9745 | 0.9805 / 0.9887 | — | Upper bound with a perfect matcher on these candidates: 0.9984. Laptop, DEV-10 |
| M-v2 | 2026-09-25 | M-v1 + 8 soft word-alignment features (52 in total); exclusive, threshold 0.725 | 99.45% | 46.9 | 0.9894 | 0.9959 | 0.9772 | 0.9826 / 0.9899 | — | Scored on EC2 from the saved M-v2 predictions |
| M-v3 | 2026-09-25 | 2 stages: stage 1 cross-fitted (3 folds) + stage 2 on stage-1 probability context; exclusive, expected-F | 99.45% | 46.9 | **0.9906** | 0.9963 | 0.9794 | 0.9814 / 0.9911 | — | Threshold 0.725 instead: 0.9904 (singletons 0.9935). EC2, DEV-10 |

---

## 5. Experiment log

Copy the template below for each run, newest entry first. Record every run, including failed ones, and say why it failed.

<!--
### EXP-XXX — <short title>

- **Date:**
- **Git commit:**
- **Hypothesis:** what change is expected to help, and why

**Methodology**
- Preprocessing / normalisation:
- Blocking / candidate generation: (keys, index, top-k, filters)
- Features:
- Model: (type, params, license, size)
- Post-processing: (threshold, one-to-one assignment, singleton gate)

**Results**

| Metric | Overall | US | India | Singletons | Non-singletons |
|---|---:|---:|---:|---:|---:|
| Blocking recall | | | | — | |
| Candidates / S1 | | | | | |
| F_0.5 | | | | | |
| Precision | | | | | |
| Recall | | | | | |

- Chosen threshold:
- Leaderboard F_0.5 (if submitted):
- Runtime / memory:

**Analysis**
- Typical false positives:
- Typical false negatives:
- Conclusion: keep / drop / iterate
- Next step:
-->

### NEURAL BRANCH — cross-encoder reranker stacked on M-v5 (SUB-N, SUB-N2)

- **Date:** 2026-09-27. GPU: nst-n6, NVIDIA RTX PRO 4000 Blackwell (HAMi share), PyTorch 2.7.1+cu128.
- **Band:** every pair with M-v5 stage-2 probability in [0.02, 0.98]: 1,526,892 train pairs (fit 454,943 / early
  stop 76,748 / eval 307,916 / rest 687,285; synthetic clones excluded, no raw text) and 1,464,340 test pairs, with
  the raw name and address text (original scripts).
- **Cross-encoder:** microsoft/mdeberta-v3-base (MIT, 278M), pair input "name ; address" x2, max 128 tokens, 1
  epoch on the fit entities' band pairs, bf16, batch 64, lr 2e-5, 401 pairs/s (19 min). Held-out band (early
  stop): log-loss 0.251 vs M-v5's 0.259; accuracy 0.891 vs 0.902.
- **Stacker:** LightGBM (63 leaves, lr 0.03) on the rest entities' band rows with p1, p2, ce, ce rank within the
  S1's band, best rival ce for the target, band size; early-stopped on the early-stop entities (345 trees). Gain:
  p2 51%, ce 41%, ce_other_t 6%.

| Eval (new, test density) | F0.5 | P | R | India | US |
|---|---:|---:|---:|---:|---:|
| M-v5 (p2) | 0.9852 | 0.9972 | 0.9609 | 0.9822 | 0.9872 |
| **M-v5 + CE stack (p3), gated expected-F 0.6** | **0.9879** | **0.9986** | **0.9653** | 0.9857 | 0.9893 |

- Test: 5,738,463 links for SUB-N2 (+78,605 / -18,095 vs SUB-C2t, all US/India; added links 5-6% near-number).
  SUB-N (CE also on France) vs the floor's France: added 27,041 links (legal conflict 0.5%, same house 82.7%),
  removed 36,949 (2.9%, 82.1%): neutral on label-free proxies, unmeasurable offline.
- **M1 (MATCH-v6 = M-v5 + support features):** new eval 0.9853 (+0.0001, noise): not shipped separately.
- **LEADERBOARD: SUB-N (CE stack on all countries, France included) = 0.978848** (2026-09-27), up from 0.968014.
  Above the US/India-only estimate for SUB-N2 (~0.974 +/- 0.004) by ~0.005: the cross-encoder most likely lifted
  France by ~3 points (it reads raw French text; M-v5 alone never saw French). The final candidate keeps the CE on France.
- **Stronger CE (2 epochs, max 160 tokens):** held-out band log-loss 0.240 (1 epoch: 0.251); stacked eval 0.9880
  (India 0.9859, US 0.9895), +0.0001 over the 1-epoch stack.
- **Independent retrieval channels on M2's residual misses (16,587 eval links; India 10,663, US 5,924):**
  address+name top-5 recover 831 India / 797 US (top-10: 1,095 / 948); rare address tokens add ~10. About +0.0002 F.
  Residual misses are mostly crowding (India: 40% have name AND address token-set >= 80; 34% address-only similar,
  i.e. invented trade names) and, in the US, no-address targets (52%).

### FINAL-DAY — test-density eval, twin filter (SUB-A), France canon (SUB-B), M-v5 (SUB-C)

- **Date:** 2026-09-27. **Compute:** k8s node nst-n7 (24 cores, 62 GB), Job `ber-full`, work dir hostPath
  `/var/lib/amlc-ber/work/full`, logs and outputs on PVC `amlc-data:/data/ber/logs/ber-full/`. The EC2 runner and
  its bucket (account <AWS_ACCOUNT_ID>) were unreachable (expired `aws login` session), so M-v3 was re-created from scratch.
- **Leaderboard of the current file (M-v3, FULL):** public LB **0.968014**; local eval 0.981263 (P 0.994966, R 0.954396).
- **Checkpointing:** `ber/ckpt.py` (marker written after the file is closed; artifacts trusted only after re-opening
  them), per-country blocking parts, per-part features (4 spawn workers), per-fold models, per-part predictions,
  `PROGRESS.txt` every 2 minutes. Verified: torn / marker-less / corrupt-footer parts are deleted and recomputed.

**Test-density data scope (NORM-v3t).** Real unmatched twin distractors (same core name and street key, house
number 1-30 away) cloned, with a new house number drawn from the real twins' gaps, until each train country
reaches the twin density measured on test with the same detector:

| Country | Twins / S1 train | Twins / S1 test | Clones |
|---|---:|---:|---:|
| US | 0.166 | 0.301 | 179,898 |
| India | 0.065 | 0.116 | 44,932 |

224,830 synthetic targets (2.09M candidate rows); eval blocking recall 98.31% (FULL-v1: 98.32%). "New eval" scores
the eval slice with the clones, "old eval" the same rows without them. Limitation: clones are anchored to S1 records
that already have a twin (twins per affected S1 about 2 instead of test's 1.1).

**M-v5a = NORM-v3t + BLK-v4b@20 + FEAT-v4 + MATCH-v2x** (the current model re-created: FEAT-v2 columns, clones
excluded from training; rule chosen on new eval: stage 2, gated expected-F, gate 0.5):

| Eval | F0.5 | P | R | India | US | Test-weighted (US+India) |
|---|---:|---:|---:|---:|---:|---:|
| Old (train density) | 0.9812 | 0.9950 | 0.9545 | 0.9776 | 0.9836 | 0.9803 |
| **New (test twin density)** | **0.9809** | 0.9946 | 0.9545 | 0.9773 | 0.9833 | 0.9800 |
| New + twin filter (SUB-A rule) | 0.9799 | | | | | |

- The re-creation reproduces the 0.9813 run (P 0.9950 / R 0.9545 vs 0.9950 / 0.9544).
- **Doubling the twins costs only 0.0003: the twin shift does not explain the 0.981 -> 0.968 gap.** If US and
  India score ~0.980 on test, the public 0.968 implies France ~0.90.
- **SUB-A (twin filter, no retraining) loses 0.0010:** on eval it removes 5,648 links of which 4,693 (83%) are true.
  Train ground truth said the same beforehand: the rule flags 0.0225 true links per US S1 in train, and it removed
  0.0243 per S1 on test.

**France canon (SUB-B = floor for US/India + M-v5a France rows).** Learned rules: `resources/france_canon/MANIFEST.md`.
France has no labels, so it was judged on label-free proxies against the floor:

| France links vs floor | n | Legal-form conflict | Same house number | Name token-set |
|---|---:|---:|---:|---:|
| Added | 34,915 | 17.0% | 61.8% | 77.5 |
| Removed | 24,376 | 3.6% | 81.1% | 86.8 |
| Unchanged (sample) | 80,000 | 0.1% | 98.5% | 92.8 |

The canon adds links that look like different businesses and removes links that look like true ones: judged
harmful (cause not measured; leading hypothesis: canonical city+region addresses inflate address similarity for
same-street different businesses in a model trained without France).

**M-v5 = NORM-v3t + BLK-v4b@20 + FEAT-v4 + MATCH-v5** (FEAT-v3 number gaps + 10 house/legal features, stage-2
house-number consensus, twin negatives x1.9, fit on fit+rest = 75% of S1, easy negatives (M-v2x p1 < 0.01) kept at
10% and weighted 10x: 19.76M training rows, 65 + 17 features; rule chosen on new eval: stage 2, gated expected-F 0.45):

| Eval | F0.5 | P | R | India | US | Test-weighted (US+India) |
|---|---:|---:|---:|---:|---:|---:|
| Old | 0.9852 | 0.9972 | 0.9609 | 0.9822 | 0.9872 | 0.9844 |
| **New** | **0.9852** | **0.9972** | **0.9609** | 0.9822 | 0.9872 | 0.9844 |
| New + twin filter | 0.9836 | | | | | |

- +0.0043 over M-v5a on the new eval (precision +0.0026 and recall +0.0064; about 20x the eval's standard error). Not
  ablated: the gain is the sum of FEAT-v4, consensus, twin weighting and 3x more training entities.
- The twin filter now removes 5,230 eval links of which 5,166 (99%) are true: the model rejects twins by itself.
- **Ablation (leave-one-out on the new eval, eval-only runs, Job `ber-ablate`):**

  | Variant | New eval F0.5 | P | R | India | US | vs M-v5 |
  |---|---:|---:|---:|---:|---:|---:|
  | M-v5 | 0.9852 | 0.9972 | 0.9609 | 0.9822 | 0.9872 | — |
  | MATCH-v5-nocons (no stage-2 house-number consensus; stage 1 reused from M-v5) | 0.9847 | 0.9969 | 0.9601 | 0.9816 | 0.9868 | **-0.0005** |
  | MATCH-v5-fit30 / -nofeat / -notw | cancelled | | | | | |

  The other three were cancelled by decision (2026-09-27): training on 100% of S1 was ruled out because the eval
  slice would become training data, so the remaining ablations could not change what ships. The rest of the
  +0.0043 (0.0038) is therefore split among FEAT-v4, twin weighting and 75% training in unknown proportions.
- Test link profile vs the 0.968 file: 78,383 added, 83,923 removed (net -5,540); near-number links removed 15,217,
  added 6,652. France changes are mixed on label-free proxies (added links: legal-form conflict 0.6%, same house
  number 82.8%, lower name similarity 73.6) and cannot be evaluated.

**Submissions (all validated, md5 in `submissions/<name>/MD5SUMS`, copies in `s3://<S3_BUCKET_2>/ber/final/`):**

| Submission | US/India rows | France rows | New eval | Old eval | Links vs 0.968 file |
|---|---|---|---:|---:|---|
| SUB-0 floor | M-v3 | M-v3 | ~0.9809 (M-v5a) | ~0.9812 | — (LB 0.968014) |
| SUB-A | M-v3 + twin filter | M-v3 + twin filter | 0.9799 | 0.9801 | -22,386 |
| SUB-B | M-v3 | M-v5a (France canon) | ~0.9809 | ~0.9812 | France +34,915 / -24,376 |
| SUB-C | M-v5 | M-v5 (France canon) | **0.9852** | 0.9852 | +78,383 / -83,923 |
| SUB-C2 | M-v5 | M-v3 | **0.9852** | 0.9852 | US/India +52,546 / -56,578 |

### FULL-v1 blocking — BLK-v4b@20 on the full data

- **Date:** 2026-09-25
- **Setup:** `run_pipeline.py`, preset M-v3 (NORM-v2 + BLK-v4b@20), EC2 r7a.2xlarge (8 cores, 64 GB). Aliases and region merges learned from non-eval links: 1,105 name and 3,107 address aliases; region merges telangana↔andhrapradesh and **delhi↔haryana** (the second only shows up on full data).

**Results**

| Split | Candidates | Per S1 | Time |
|---|---:|---:|---:|
| Train (2.2M S1) | 104,748,300 | 47.5 | 703 s (India 222 s, US 481 s) |
| Test (1.73M S1) | 83,319,185 | 48.1 | 488 s (France 60 s, India 238 s, US 190 s) |

Peak memory 44.4 GB. Eval slice (441,521 S1, 1,528,407 links): **pair recall 98.32%**, all links found for 94.53% of S1.

| Eval links | Links | Recall | Missed |
|---|---:|---:|---:|
| Target without address | 66,939 | 88.73% | 7,544 |
| Target with address | 1,461,468 | 98.76% | 18,132 |
| India S2 / S3 | 295,785 / 315,346 | 98.34% / 96.58% | |
| US S2 / S3 | 443,077 / 474,199 | 98.85% / 98.97% | |

Recall by depth of the main pass (the name-only pass always counted): top-5 96.00%, top-10 97.42%, top-15 97.98%, top-20 98.32%.

**Analysis**
- DEV-10 overstated blocking recall by 1.1 points (99.45%): each region holds about 10 times more rival records on full data, so true matches fall out of the top 20 more often.
- The name-only pass for targets without an address keeps only 5 per S1; with many more same-name records it finds just 88.7% of those links, 29% of all misses.
- Recall is still rising at k=20, so a deeper search should pay. **Next (BLK-v5):** measure recall curves with top-40 main pass and top-20 name-only pass, then pick the cutoffs from the curves.

### Error analysis — what M-v3 still gets wrong

- **Date:** 2026-09-25
- **Setup:** M-v3 stage 2, exclusive assignment, threshold 0.725, DEV-10 evaluation slice (`experiments/dev_errors.py`).

**Results:** 152,187 true links, 149,431 predicted, 3,187 missed, 431 false.

| Missed links | Count | Share | Targets without address |
|---|---:|---:|---:|
| Never a candidate (blocking miss) | 839 | 26% | — |
| Lost to another S1 entity under exclusive assignment | 925 | 29% | 95% |
| Scored below the threshold | 1,423 | 45% | 46% |

**Analysis**
- Targets without an address are about 3% of all targets but about half of the recoverable misses. Their only evidence is the name, and many S1 entities share a name, so the wrong one often wins the target.
- 82% of false positives do have an address. The typical one is the same name at a **nearby house number** (831 vs 835 Winsor Pl, 3-4-114/12 vs /13, G/70/15 vs /16): the data plants such distractors. True matches also carry number noise (189 vs 889, 20 vs 19, 730 vs 30), so exact number overlap alone cannot separate them.
- Some true matches carry an invented trade name (Asahi Charitable Trust ↔ Korevo, East Saint Louis Baptist Center ↔ Ectozetalum). Only the address, or the entity's other records, can link these.
- **Next:** number-gap features (digit edit distance and numeric gap of the closest unmatched numbers) and support features (how much a candidate resembles the S1's confident candidates), tested in `experiments/dev_v3.py`.

### M-v3 — two-stage stacking and the decision rule

- **Date:** 2026-09-25
- **Git commit:** not yet committed
- **Hypothesis:** In M-v1 the margin of the *blocking score* over the best other S1 carried 79% of the gain. The same competition measured in stage-1 *probabilities* should resolve contested targets better. The set of links should also be chosen per entity rather than by one global threshold.

**Methodology**
- Stage 1: M-v2 features, cross-fitted over 3 folds of the fit entities. Fit rows get the prediction of the fold model that did not see their entity; all other rows get the mean of the 3 models.
- Stage 2: stage-1 features plus `p1` and 11 context features from `probability_context`: rank of `p1` within its S1 (overall and per source) and for its target; best other `p1` for the target, the S1 and the S1's source; margins over those; sum of `p1` over the S1; count above 0.5; in-degree of the target.
- Decision rules, each after exclusive assignment: a single threshold; **expected F** (per S1, predict the top k that maximise expected F0.5 under independent probabilities, or nothing when "no match" is the better bet); **gated expected F** (nothing unless the best candidate reaches a gate).

**Results (DEV-10 evaluation slice, 44,137 S1)**

| Model / rule | F0.5 | P | R | Singletons | Others |
|---|---:|---:|---:|---:|---:|
| Stage 1 cross-fitted, threshold 0.725 | 0.9894 | 0.9959 | 0.9770 | 0.9822 | 0.9898 |
| Stage 1, expected F | 0.9887 | 0.9948 | 0.9770 | 0.9603 | 0.9904 |
| Stage 2, threshold 0.725 | 0.9904 | 0.9970 | 0.9789 | **0.9935** | 0.9902 |
| **Stage 2, expected F** | **0.9906** | 0.9963 | 0.9794 | 0.9814 | **0.9911** |
| Stage 2, gated expected F (gate 0.55) | 0.9906 | 0.9966 | 0.9792 | 0.9858 | 0.9909 |

- **Top stage-2 features by gain:** `p1` 69%, `p1_margin_t` (margin over the best other S1 for the target) 26%, `margin_vs_other_s1` 3%.

**Analysis**
- Stage 2 adds +0.0010 to +0.0012, mostly on singletons: at the same threshold they go from 0.9822 to 0.9935, because `p1_margin_t` removes targets that a stronger S1 claims.
- A threshold protects singletons; expected F picks better set sizes for entities with matches. Gating did not combine the two on DEV-10: every gate from 0.30 to 0.55 gives 0.9906. The full pipeline picks the rule on its own evaluation slice.
- Caveat: `dev_stack.py` trains stage 2 once on the fit rows, so on DEV-10 those rows compete with in-sample probabilities during exclusive assignment. `run_pipeline.py` cross-fits stage 2 as well.
- **Conclusion:** keep both stages; choose the rule on full data.

### M-v2 — soft word alignment of names

- **Date:** 2026-09-25
- **Hypothesis:** Token-level alignment should tell noise (typos, added generic words) apart from a different business (a rare word replaced by another).

**Methodology**
- `align_features`: each core-name token is aligned to its best Jaro-Winkler match on the other side, in both directions. Per side: mean similarity (Monge-Elkan), worst token, number of unaligned tokens (similarity below 0.85) and the highest S1-side IDF among them. Tokens never seen in S1 get the country's maximum IDF. 8 features, 52 in total.

**Results (DEV-10 evaluation slice):** exclusive, threshold 0.725: **F0.5 0.9894** (P 0.9959, R 0.9772, singletons 0.9826, others 0.9899). Non-exclusive best: 0.9893 at 0.825. Expected F: 0.9890 (singletons 0.9635).

- **Conclusion:** keep (+0.0012 over M-v1).

### M-v1 — first end-to-end matcher (LightGBM + exclusive assignment)

- **Date:** 2026-09-25
- **Git commit:** not yet committed
- **Hypothesis:** A gradient-boosted classifier on string-similarity and competition features, plus the constraint that each S2/S3 record belongs to at most one S1 entity, should reach high precision.

**Methodology**
- Preprocessing and blocking: NORM-v2, learned aliases, BLK-v4b with top-20 per source plus the no-address pass. That gives 10.36M candidate pairs for 220,682 S1 records, of which 7.33% are positive.
- **Features (44):**
  - Blocking: score, name cosine and address cosine (IDF-weighted, from the blocking matrices).
  - Names (rapidfuzz), on the full name, the core name (legal words removed) and the joined-up name: ratio, token-set, token-sort, partial ratio, Jaro-Winkler.
  - Aliases: best match against the `fka` / `dba` parts.
  - Token containment in both directions, for names and addresses.
  - Addresses: ratio, token-set, token-sort, partial ratio.
  - House numbers: Jaccard, share of B's numbers in A, whether A's first number is in B, truncated-number match.
  - Lengths, and flags: address missing, website name, alias present, non-Latin script.
  - Context: rank of the candidate within its S1 record (per source and overall), gap to the best, number of candidates.
  - Competition: how many S1 records retrieved this target, this S1 record's rank among them, and the score margin over the best *other* S1 record.
  - Ambiguity: how many S1 records share this name.
- **Model:** LightGBM, `objective=binary` (**log-loss / binary cross-entropy**). Settings: 255 leaves, learning rate 0.05, min 200 rows per leaf, 0.8 feature and bagging fractions, L2 = 1. Early stopping after 100 rounds on the 10% slice; best iteration 385.
- **Why log loss:** F_0.5 is decided per S1 entity over a *set* of links, so the decision step needs calibrated probabilities. A ranking loss (LambdaRank) orders candidates well, but its scores are not probabilities and it has no natural cut-off for "no match".
- **Post-processing:** exclusive assignment (each target kept only for the S1 entity that scores it highest), then a single probability threshold tuned for macro F_0.5 (grid 0.20–0.95).

**Results (DEV-10 evaluation slice, 44,137 S1 entities)**

| Metric | Overall | Singletons | Non-singletons |
|---|---:|---:|---:|
| F_0.5 (non-exclusive, threshold 0.75) | 0.9880 | 0.9814 | 0.9884 |
| **F_0.5 (exclusive, threshold 0.725)** | **0.9882** | 0.9805 | 0.9887 |
| Precision / Recall (exclusive) | 0.9954 / 0.9745 | | |
| Upper bound (perfect matcher on these candidates) | 0.9984 | | |

- **Top features by gain:** margin over the best other S1 record 79.2%, rank among the target's S1 records 10.2%, blocking score 3.1%, then address containment, number Jaccard and address token-set.
- **Runtime (laptop, 12 threads):** blocking 67 s, features 121 s, training 175 s.

**Analysis**
- Precision is already 0.995. Most of the remaining loss is recall (0.9745 against a 0.9945 blocking ceiling).
- DEV-10 has 10× fewer competing records per region than the full data, so this score is optimistic. The model must be trained and validated on the full data.
- **Next:**
  - A second stage that uses first-stage probabilities as competition features (probability margin over the other S1 records, and over the other candidates of the same S1 record).
  - A decision step that maximises expected F_0.5 per entity.
  - A full-data run on SageMaker.

### BLK-v4b — region-split search with positional region keys (current)

- **Date:** 2026-09-25
- **Hypothesis:** BLK-v4 picked up common words that happen to end some addresses (`rd`, `nagar`) as region keys. Their groups swallowed most records and produced bogus merges. States sit at the end of an address almost every time they appear, so a positional test should keep only real regions.

**Methodology**
- Region keys: the last token of at least 0.2% of S1 addresses in that country, **and** the last token in at least 60% of the addresses that contain it.
- Merges: two keys are merged when at least 10 true training links cross them.
- A country is split only if it has training links. An unseen country (France) gets a full search, because its keys can't be validated.
- Everything else is as in BLK-v4.

**Results (DEV-10):** pair recall **99.52%**, all found 98.35%, 65.9 candidates per S1, 43 s. The keys are now clean (16 Indian states, 37 US states), with a single merge (`telangana` ↔ `andhrapradesh`, because some Telangana records are still filed under the pre-2014 state). With top-10 per source: 99.10% recall at 25.9 candidates per S1.

- **Conclusion:** keep. It is the default blocking for the first matcher.

### BLK-v4 — search split by region

- **Date:** 2026-09-25
- **Hypothesis:** Sparse search cost grows with (S1 size × target size), so BLK-v2 would take about an hour per run on the full data. A true match almost always shares its state with the S1 record. Comparing each S1 record only with targets from its own region should therefore cut the cost roughly by the number of regions, with little recall loss.
- **Measured before building:** splitting by region would lose only 0.12% of true links (both records have a region, but different ones). These were mostly Telangana ↔ Andhra Pradesh and `dl` ↔ `delhi`.

**Methodology**
- Keys: the frequent last tokens of S1 addresses.
- Region pairs that true links cross are merged with union-find.
- An S1 record is compared with targets in its own region group(s) plus all targets that have no region token. An S1 record with no region token is compared with everything.
- IDF stays per (country, source), so scores are comparable across groups.

**Results:** pair recall 99.48%, all found 98.20%, 66.9 candidates per S1, 48 s (v2: 135 s). Recall went **up** versus v2, because lookalikes from other regions no longer take places in the top 30.

**Analysis:** the learned merges included `rd`, `nagar` and `hyderabad`, because those words end some Indian addresses. Fixed in v4b.

### BLK-v3 — fixed df cap (rejected)

- **Date:** 2026-09-25
- **Hypothesis:** Character 3-grams account for 60% of the search work (US/S2 profile). A fixed df cap would bound the work per feature, so full-data cost would grow about 10× instead of 100×.
- **Results:** a cap equal to 3,000 on full data gave 97.74% recall; a cap equal to 6,000 gave 98.49%. Without 3-grams the recall was 97.36% and 98.12%.
- **Why rejected:** the recall loss (0.8–1.6 points) is too large. A fixed cap drops the medium-frequency words (city names, common name words) that rank the true match, so region splitting (v4) was used instead.

### BLK-v2 — joined names, 3-grams, name-only pass for missing addresses

- **Date:** 2026-09-25
- **Hypothesis:** The v1 misses were mainly:
  - website or joined names with no word in common with the S1 name (`eyeassociates`, `unitedit`)
  - heavy typos (`csmpaneis`)
  - targets with no address, which lose on score to same-name records that have one
  - `shree` vs `sri`

**Methodology**
- Add the whole joined-up core name as one feature (it matches about 5 records on average, so it costs almost nothing).
- Add character 3-grams of the joined-up name.
- Merge the honorific variants.
- Add a second, name-only search (top-5) restricted to targets with no address.

**Results:** pair recall 99.31% (+1.65 points), all found 97.71%, 69.1 candidates per S1. The no-address pass alone recovered 3,046 links. Time 135 s: the 3-grams dominate, and cost would grow about 100× on the full data.

### BLK-v1 — IDF-weighted sparse token search

- **Date:** 2026-09-25

**Methodology**
- **Normalisation (NORM-v2):**
  - transliterate to ASCII with anyascii
  - canonicalise legal forms, street types, US states and ordinals
  - join multi-word Indian state names into one token
  - add French street types
  - fix digit-for-letter typos (`5ummit` → `summit`)
  - split names at `fka` / `dba`
  - turn website names back into words
  - strip `(ID: …)` tags
- **Learned aliases:** per country, from training links. A target token that is unaligned in at least 50% of its pairs, is always accompanied by the same unaligned S1 token, and is rare on the S1 side becomes an alias of that S1 token. Ties are broken by spelling similarity. This yields 452 name and 135 address aliases, e.g. `praivet` → `pvt`, `eksports` → `exports`, `mh` / `mharastr` → `maharashtra`, `tn` → `tamilnadu`, `lnc` → `inc`.
- **Features:** name tokens, address tokens, 4-char prefixes, consonant skeletons, and the first/last 3 digits of house numbers (for `7530` ~ `530`). All features are hashed into 2^24 slots.
- **Scoring:** IDF weights on the target side; name and address blocks L2-normalised separately; score = cos(name) + cos(address). Top-30 per (country, source), via sparse_dot_topn (Apache-2.0).

**Results:** pair recall 97.66%, all found 92.76%, 60 candidates per S1, 26 s. Recall at top-5: 95.2%; top-10: 96.5%; top-20: 97.4%.

### Data facts that shaped the design (EDA, 2026-09-25)

- Country never differs between an S1 record and its matches (0 of 7.64M links), so blocking runs within each country, with no country hard-coded.
- 39.6% of S1 records share their exact normalised name with another S1 record ("Primary Care Group" appears 253 times), so the address and competition features must break ties.
- Only 5–6% of distractors share an exact name with an S1 record. The hard negatives are same-name businesses at a different address.
- At most 5 matches per S1 come from S2, and at most 6 from S3.
- Indian S2/S3 names are in Devanagari 7–13% of the time, and addresses about 13% of the time. S1 is always clean Latin text with a full address.

## Final evening (2026-09-27, 22:00–22:30 IST): V3 on the leaderboard, CLAUDE_NEXT_STEPS_099 steps 1–3

**SUB-FINAL-V3 scored 0.982564 on the leaderboard** (SUB-N3: 0.979604).

Step 1: bagged stackers vs V3b (paired per-entity), test S1 changed vs scored V3. Bags average 5 models' predictions.

| Variant | ES | EVAL | P | R | US | India | Singletons | Better / worse vs V3b | Test S1 changed vs V3 (Fr / In / US) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| V3b gate 0.65 | 0.989774 | 0.989644 | 0.9988 | 0.9701 | 0.990508 | 0.988347 | 0.9960 | — | 2,115 / 903 / 876 |
| V3c dup-avg, gate 0.70 | 0.989768 | 0.989651 | 0.9988 | 0.9701 | 0.990497 | 0.988382 | 0.9966 | 171 / 163 | 2,233 / 1,045 / 991 |
| V3d 5 seeds | 0.989753 | 0.989650 | 0.9988 | 0.9701 | 0.990502 | 0.988373 | 0.9965 | 187 / 200 | 2,218 / 929 / 921 |
| V3e 5 seeds, 127 leaves | 0.989763 | 0.989651 | 0.9989 | 0.9700 | 0.990500 | 0.988377 | 0.9972 | 274 / 259 | 2,973 / 1,248 / 1,299 |
| V3f 5 seeds, 31 leaves | 0.989774 | 0.989606 | 0.9988 | 0.9702 | 0.990449 | 0.988342 | 0.9967 | 560 / 373 | 3,483 / 1,844 / 1,835 |

Bagging: no gain.

Step 2: error table of V3c on EVAL (`neural/errtab.py`, base 0.989651 reproduced). The oracle gain is the change in macro F0.5 if that group were fixed perfectly.

| Group | Links | S1 | Oracle gain | India / US | Target without address |
|---|---:|---:|---:|---|---:|
| Missing candidate | 16,587 | 15,008 | +0.00353 | 10,663 / 5,924 | 4,566 |
| Below band, never reviewed | 4,405 | 4,262 | +0.00094 | 1,309 / 3,096 | 3,978 |
| Reviewed but rejected | 24,966 | 23,699 | **+0.00514** | 8,996 / 15,970 | **21,243** |
| Lost to another S1 | 64 | 64 | +0.00001 | 25 / 39 | 53 |
| False match | 1,428 | 1,409 | +0.00075 | 595 / 833 | 690 |
| Singleton false match | 85 | 84 | +0.00019 | 33 / 52 | 28 |

Singletons predicted non-empty: 84. Matched S1 left empty: 943. The largest recoverable loss is no-address targets that were reviewed and rejected, so the below-band pilot (+0.0009 at most) is not worth it tonight.

Step 3: France review audit.
- All 765,958 France confident test pairs were scored: the input was `band_hi_ord`, which keeps every France row and subsamples only the US/India training rows.
- 5 of 5 parts have markers; no part is missing.
- All 76,077 duplicate keys (hi vs m2) come from the same checkpoint (`/n6/ce2/model_final`), the same max length (160) and byte-identical input text.
- 23,185 differ by more than 1e-3 (640 by more than 0.05). The only difference is batch size, 128 vs 64: length-sorted bf16 batches pad differently. Averaging them is justified.
- Confident training rows (US/India, ES/EVAL/REST sample) have mean p2 0.99951; France test mean p2 is 0.99919.
