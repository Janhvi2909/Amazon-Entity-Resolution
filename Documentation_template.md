# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** kishut mariken  
**Team Members:** Ronit Kumar Choudhary, Vaishnav Verma, Janhvi Yadav  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We match records in three stages: an IDF-weighted sparse search generates candidates, a
two-stage LightGBM classifier scores each (Source 1, candidate) pair, and a per-entity decision
turns probabilities into the match list that maximises macro F0.5. Two ideas carry most of the
accuracy: **competition features** — every Source 2/3 record belongs to at most one Source 1
entity, so a candidate is judged against the rival Source 1 records that want the same record —
and a **validation protocol that keeps the full competition density**, which showed that a 10%
development sample overstates blocking recall by more than a point. On top of the LightGBM matcher,
three multilingual **cross-encoders** (mDeBERTa-v3-base, bge-reranker-v2-m3, MuRIL for India only) re-read
the raw text of uncertain pairs, and a LightGBM stacker combines everything. Public leaderboard: 0.968014
(LightGBM only) → 0.978848 (first cross-encoder stack) → 0.979604 (three cross-encoders) → 0.982564 (M2 fusion + France review, V3).

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the training files (2.2M Source 1, 5.0M Source 2, 5.3M Source 3 records):

- **One-to-many, never many-to-many.** No Source 2/3 record is linked to two Source 1 entities.
  A record claimed by several Source 1 entities can therefore go to at most one of them.
- **Matches per entity:** mean 3.46, at most 5 from Source 2 and 6 from Source 3. 5.6% of entities
  have no match; each such singleton scores 1.0 only if nothing is predicted for it.
- **Distractors:** 26% of Source 2/3 records match nothing. Only 5–6% of them share an exact name
  with a Source 1 record; the hard negatives are same-name businesses at a *different address*.
- **Shared names:** 39.6% of Source 1 records share their normalised name with another Source 1
  record ("Primary Care Group" appears 253 times), so the address must break ties.
- **Noise:** abbreviations and legal-form variants, typos (including digit-for-letter, "5ummit"),
  word-order changes, website-style names, "formerly known as / doing business as" names, invented
  trade names ("Korevo" for "Asahi Charitable Trust"), Devanagari names in 7–13% and addresses in
  ~13% of Indian Source 2/3 records, house-number perturbations, and missing addresses (~3%).
- **Country** never differs between linked records. Test adds France, absent from training.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage classifier + cross-encoder reranking + stacker + per-entity set decision  
**Core Innovation:** competition-aware scoring (rank and probability margin against rival Source 1
records, then exclusive assignment), full-density validation, and cross-encoders on raw text for the
uncertain pairs and for France (unseen in training).

Every component is a named version (normalisation NORM, blocking BLK, features FEAT, matcher
MATCH) recorded in `src/ber/versions.py`. The submission combines two runs — **M2** (NORM-v4t +
BLK-v4b@40n20 + FEAT-v5 + MATCH-v7, stage 1) and **M-v5** (NORM-v3t + BLK-v4b@20 + FEAT-v4 + MATCH-v5,
stage 1 + 2) — with the cross-encoder stacker in `src/neural/stack_v3.py` (Section 4). The submitted file is **SUB-FINAL-V3**
(public leaderboard **0.982564**).

---

## 3. Candidate Generation (Blocking)

- **Normalisation (NORM-v2):** transliteration to ASCII (anyascii), canonical legal forms, street
  types, US states and ordinals, Indian state names joined into one token, French street types,
  digit-for-letter repairs, "fka / dba" names split, website names turned into words. On top,
  **per-country token aliases learned from the training links** (1,105 name and 3,107 address
  aliases, e.g. `praivet → pvt`, `mh → maharashtra`); France gets none, by design.
- **Blocking keys used:** each record becomes a sparse vector of hashed features — name and
  address tokens, 4-character prefixes, consonant skeletons, leading/trailing digits of house
  numbers, the whole joined name, and character 3-grams of the joined name. Features are
  IDF-weighted on the target side, name and address blocks are normalised separately, and the
  score is cos(name) + cos(address). The exact top-k per (Source 1 record, target source) is
  found with a multi-threaded sparse top-n product (`sparse_dot_topn`, Apache-2.0).
- **Region split:** within a country, the search runs inside region groups — frequent last
  tokens of Source 1 addresses that sit at the end of an address (states), with merges learned
  from training links that cross them (Telangana ↔ Andhra Pradesh, Delhi ↔ Haryana). Records
  without a region token are searched against everything. France, having no training links, is
  searched whole.
- **Name-only pass** for targets without an address, which would otherwise lose to same-name
  records that have one.
- **Normalisation used by the final runs:** **NORM-v3t** (M-v5) = the above plus test-density twin
  augmentation of the training targets (real near-number distractors cloned up to the density measured,
  label-free, on test) and, for countries without training links (France), city/region canon learned by
  co-occurrence, repaired digit-corrupted street types, bis/ter kept with the house number and legal forms
  kept where stripping them would merge distinct records. **NORM-v4t** (M2) = NORM-v3t plus address
  components (street core, landmark clause, unit and PO box split out).
- **Candidate pairs generated:** 83.3M for the test set (48.1 per Source 1 record) with
  BLK-v4b@20; the final run uses the deeper **BLK-v4b@40n20** (top-40 per source, name-only pass
  top-20): **200,471,427 test pairs (115.7 per Source 1 record)**. The model scores every one of them, and
  `candidate_pairs.tsv` lists exactly this set; every submitted match is in it.
- **How we ensured true matches were not lost:** recall is measured on held-out training entities
  at full density: 98.32% of true links with BLK-v4b@20 (88.7% for targets without an address,
  98.8% for the rest). Recall-versus-depth curves for both passes (still rising at top-20) led
  to BLK-v4b@40n20: **98.91%** of true links (from 98.31%).

---

## 4. Matching Model

**Features used.** The first matcher used 52 (FEAT-v2, below); the final runs add 3 number-gap features (FEAT-v3),
10 house/legal features (**FEAT-v4, 65 in total, used by M-v5**) and 7 address-component features (**FEAT-v5,
72 in total, used by M2**):
- **Name features:** rapidfuzz ratio, token-set, token-sort and partial ratios, and Jaro-Winkler
  on the full, core (legal words removed) and joined names; best match against "fka / dba" parts;
  token containment both ways; **soft word alignment** (each core-name token aligned to its best
  Jaro-Winkler match, both directions: mean, worst, number of unaligned tokens and the highest IDF
  among them — separating typos from a different business).
- **Address features:** ratio, token-set, token-sort and partial ratios; token containment;
  house-number agreement (Jaccard, share of target numbers present, truncated-number match).
- **Number, house and legal features (FEAT-v3/v4):** digit edit distance and numeric gap of the closest
  unmatched numbers; parsed house number: both present, same street key, gap, log gap, digit edit, most
  significant differing digit, near-number twin flag, bis/ter conflict; legal form equal / conflicting.
- **Address components (FEAT-v5):** street-core ratio / token-set / partial ratio; for landmark, unit and
  PO box a state (neither side, Source 1 only, candidate only, both sharing a token, both present and disjoint);
  landmark token-set similarity when both records have one (from NORM-v4t's component split).
- **Competition and context features:** blocking score and name/address cosines; rank of the
  candidate within its Source 1 record (per source and overall); gap to the best candidate;
  number of candidates; **number of Source 1 records that retrieved the target, this record's rank
  among them, and its score margin over the best rival**; how many Source 1 records share the name.
- **Stage 2** adds the stage-1 probability and its context: rank among the record's candidates and
  among the target's Source 1 records, best rival probability and the margins over it.

**Model type:** LightGBM (MIT), binary log loss, 255 leaves, learning rate 0.05, early stopping.
Both final runs (MATCH-v5 for M-v5, MATCH-v7 for M2) are fitted on the fit and rest entities (75% of
Source 1) and **cross-fitted** over 3 folds, so every training-entity probability used downstream is
out-of-fold; eval and test rows get the mean of the fold models. Twin (near-number) distractor negatives are
weighted 1.9x; easy negatives (probability below 0.01 from an earlier or quick screening model) are subsampled to 10% and weighted 10x.
M-v5 uses stage 1 and stage 2 (stage 2 adds a house-number consensus feature); M2 uses **stage 1 only**,
because its stage 2 on 253M training pairs exceeded the 62 GB node.

**Threshold selection method:** each target is kept only for the Source 1 entity that scores it
highest (exclusive assignment); then a rule chosen on held-out training entities by macro F0.5:
a single threshold, expected-F (per entity, the top-k maximising expected F0.5 under the
predicted probabilities, or no match), or gated expected-F.

**Cross-encoders (final system).** Pairs whose LightGBM probability lies in [0.02, 0.98] are re-read
from their **raw** text ("name ; address", original scripts) by cross-encoders fine-tuned as binary pair
classifiers on the fit entities' band pairs (bf16, one RTX PRO 4000 GPU):
- microsoft/mdeberta-v3-base (MIT, 278M): 2 epochs, 160 tokens; held-out log-loss 0.240 vs LightGBM 0.259.
- BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M): 1 epoch, 128 tokens; log-loss 0.246.
- google/muril-base-cased (Apache-2.0), trained and applied to **India pairs only**: log-loss 0.206 vs
  0.227 for LightGBM on India (trained on all countries it was poor, 0.509).
- A **confident-pair review**: mDeBERTa also scores pairs above 0.98 of M-v5 (every France test pair there,
  765,958; in US/India all early-stop rows plus samples of 400,000 rest and 500,000 eval rows), so the stacker
  can overturn confident LightGBM links in France, where LightGBM never saw training data.

**Stacker** (`src/neural/stack_v3.py`). LightGBM (63 leaves, learning rate 0.03, seed 42) on every
cross-encoder-scored row of the rest entities (uncertain band plus the confident-pair review), early-stopped
on the early-stop entities. Inputs: M2 stage-1 probability, the M-v5 stage-1/2 probabilities (missing where M-v5 never
generated the pair), the three cross-encoder scores (MuRIL missing
outside India), cross-encoder context (rank within the Source 1 record's band, best rival score for the
target, band size) and 16 house-number / legal-form / address-component features. A pair scored twice by
the same cross-encoder (M2 band and review band, same checkpoint and text) keeps the review-band score.
Pairs without any cross-encoder score keep the LightGBM probability.

**Decision (final).** Exclusive assignment, then gated expected-F0.5; the rule (threshold or gated
expected-F) and its parameter are chosen on the evaluation entities: gated expected-F, gate 0.55. On test,
per-entity caps of 5 Source 2 and 6 Source 3 matches (the training maxima) are then applied, keeping the most
probable. Result: 5,751,431 links, every target given to at most one Source 1 record.

**Validation:** Source 1 training entities are split once — fit 30%, early stopping 5%,
evaluation 20% (441,521 entities), rest 45%. All entities stay in the candidate search, aliases
and region merges are learned without evaluation links, and evaluation rows are scored exactly
like test rows. The LightGBM matchers train on fit + rest, the cross-encoders on fit, the stacker on
rest (whose LightGBM probabilities are out-of-fold and whose pairs no cross-encoder saw in training); the
decision is chosen on eval.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), full-data evaluation slice (441,521 US/India entities, test-density
  distractors):** **0.98965** (precision 0.99873, recall 0.97019; India 0.98835, US 0.99052) for the final
  system (V3), against 0.9852 for M-v5 alone and 0.9880 for the previous three-cross-encoder stack.
  **Public leaderboard (portal): 0.982564** (previous stack 0.979604); the gap to validation is mostly France,
  which has no labels.
  Development sample (10%,
  44,137 entities): 0.9906 (precision 0.9963, recall 0.9794) with stage 2 and expected-F,
  against 0.9882 for the first single-stage model.
- **Common false positives (wrong merges):** the same business name at a *nearby* house number
  (831 vs 835 Winsor Pl; 3-4-114/12 vs /13) — planted distractors that differ from true matches
  only in the number; names that differ in one distinctive word.
- **Common false negatives (missed matches):** records without an address (about half of the
  recoverable misses on the development sample: with only a name, a same-name Source 1 entity
  often claims the record); invented trade names that match only through the address; blocking
  misses in dense regions.

| Version (development sample) | F0.5 | Precision | Recall |
|---|---:|---:|---:|
| Single model, 44 features | 0.9882 | 0.9954 | 0.9745 |
| + soft word alignment (52) | 0.9894 | 0.9959 | 0.9772 |
| + stage 2, expected-F | 0.9906 | 0.9963 | 0.9794 |

**Error table of the final pipeline** (V3c: the V3 pipeline with the gate chosen on early-stop entities,
eval 0.98965; `neural/errtab.py`; oracle = macro-F0.5 gain if the group were fixed perfectly): missing
candidate +0.0035 (16,587 links); reviewed by a cross-encoder but rejected +0.0051 (24,966 links, 85% of them
targets without an address); below the review band +0.0009; lost to another Source 1 record +0.00001; false
matches +0.0007; singleton false matches +0.0002. Validation numbers are on held-out training entities;
leaderboard numbers are the portal's.

---

## 6. Conclusion

Treating the problem as competition for records — rather than independent pair classification —
and validating at the real competition density were the decisive choices. The main lesson: a
convenient 10% sample hid a 1.1-point blocking-recall loss that only full-density validation
exposed. Deeper blocking recovered 0.6 points of recall ceiling (98.31% → 98.91%), and cross-encoders on raw
text gave the largest single leaderboard step (0.968 → 0.979), most of it on France, which the LightGBM
features had never seen; deeper blocking, fusion with the older model and a cross-encoder review of confident
French links took it to 0.982564. The remaining gap is recall (0.970): targets without an address lost to a same-name
rival, and links never retrieved.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: `src/run_pipeline.py` runs every stage (prep → block →
features → train → predict) and writes `output/matching_results.tsv` and
`output/candidate_pairs.tsv`, then runs the official validator. `src/ber/` holds the components
(`normalize`, `aliases`, `blocking`, `features`, `model`, `metrics`, `versions`); `src/experiments/`
holds the development-sample experiments behind each version. `src/neural/` holds the final branch: band extraction
(`band.py`, `band_hi.py`), cross-encoder training and inference (`ce.py`), the M2 score rebuild
(`m2p1.py`), the submitted stacker (`stack_v3.py`) and a later stacker with post-submission experiments
(`stack.py`), and audits (`errtab.py`, `validate_streaming.py`); `README.md` lists the exact final commands and
`requirements-neural.txt` the GPU packages; `requirements.txt` pins the CPU environment
(Python 3.12).

### B. Additional Results

Full experiment log with every version: `method_result.md`.

| System (full-data eval slice) | F0.5 | Precision | Recall | Public LB |
|---|---:|---:|---:|---:|
| M-v3 (LightGBM, BLK-v4b@20) | 0.9813 | 0.9950 | 0.9544 | 0.968014 |
| M-v5 (test-density twins, FEAT-v4, 75% training entities) | 0.9852 | 0.9972 | 0.9609 | — |
| M-v5 + mDeBERTa stack | 0.9879 | 0.9986 | 0.9653 | 0.978848 |
| + mDeBERTa-2 + bge + MuRIL-India | 0.9880 | 0.9987 | 0.9657 | 0.979604 |
| **Final V3: M2 + M-v5 + three cross-encoders + France review** | **0.98965** | **0.99873** | **0.97019** | **0.982564** |

Post-submission experiments (not the submitted file): gate chosen on early-stop entities with duplicate CE
scores averaged (V3c) 0.98965; + name-agreement / name-frequency support features (V3s) 0.98971, whose
France-calibration variant (fr08) scored 0.982529 on the leaderboard; stacker bagging over 5 seeds: no gain;
target-to-target cross-encoder anchors: paired eval −0.000005, rejected.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
