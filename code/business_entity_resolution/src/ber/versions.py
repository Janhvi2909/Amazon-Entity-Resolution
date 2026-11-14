"""Named versions of every pipeline component — the IDs used in method_result.md.

A run is described by four component versions:
  NORM   text normalisation and learned aliases     -> stage prep
  BLK    candidate generation (blocking)            -> stage block
  FEAT   pairwise features                          -> stage features
  MATCH  models and the decision rule               -> stages train, predict
An end-to-end version (M-v1, M-v2, ...) is one combination of the four (PRESETS).

Every stage caches its output under the versions it depends on, e.g. block/NORM-v2__BLK-v4b@20,
so switching one version recomputes only what it changes and never reuses stale files.

Rule: never edit a registered version. Add a new one, and log it in method_result.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

STAGE_DEPS = {"prep": ("norm",), "block": ("norm", "block"), "features": ("norm", "block", "feat"),
              "train": ("norm", "block", "feat", "match"), "predict": ("norm", "block", "feat", "match")}


@dataclass(frozen=True)
class Norm:
    id: str
    note: str
    alias_min_count: int = 25       # a target token needs this many links with the same unaligned S1 token ...
    alias_min_ratio: float = 0.5    # ... in at least this share of the links that contain it
    alias_max_unaligned: int = 4    # links with more unaligned tokens are too noisy to learn from
    twin_density: bool = False      # clone real train twin distractors up to the twin density measured on test
    unseen_canon: bool = False      # countries without training links: learned geography, street-type and bis/ter canon
    legal_learned: bool = False     # keep legal forms in the core name where stripping them merges distinct S1 records
    addr_components: bool = False   # split addresses into street core, landmark clause, unit and PO box


@dataclass(frozen=True)
class Block:
    id: str
    note: str
    k: int = 20                     # candidates per S1 record and target source
    k_noaddr: int = 5               # extra name-only pass over targets without address (0 = off)
    char_grams: bool = True         # character 3-grams of the joined core name
    joined_name: bool = True        # the whole joined core name as one feature
    max_df_frac: float = 0.01       # ignore features found in more than this share of targets ...
    min_df_cap: int = 1000          # ... unless the cap would fall below this document frequency
    region_split: bool = True       # search within region groups (states) instead of the whole country
    region_min_share: float = 0.002     # region key: last token of at least this share of S1 addresses ...
    region_min_last_ratio: float = 0.6  # ... and the last token in at least this share of its occurrences
    region_min_cross: int = 10          # merge two regions when this many training links cross them


@dataclass(frozen=True)
class Feat:
    id: str
    note: str
    align: bool = True              # soft word alignment of core names (8 features)
    number_gap: bool = False        # closest unmatched numbers: edit distance and numeric gap (3 features)
    twins: bool = False             # house number / street key / legal-form features (9 features)
    addr_parts: bool = False        # street-core similarity + landmark / unit / PO box states (7 features)


@dataclass(frozen=True)
class Match:
    id: str
    note: str
    stages: int = 2                 # 1: one model; 2: + a model on stage-1 probability context
    folds: int = 3                  # cross-fitting folds over the fit entities
    support: bool = False           # stage 2 also sees similarity to the S1's confident candidates
    rules: tuple = ("threshold", "expected_f", "gated_ef")  # decision rules tried on the eval slice
    lgb: tuple = ()                 # (name, value) overrides of model.DEFAULT_PARAMS
    pred_margin: float | None = None  # stop summing trees once 2*|raw score| exceeds this (LightGBM pred_early_stop)
    cols: str = "all"               # "all" features of the FEAT version, or "v2": only the FEAT-v2 columns
    train_synth: bool = True        # train on rows whose target is a synthetic twin (test-density data scopes)
    twin_weight: float = 1.0        # sample weight of twin negatives (h_near = 1, y = 0)
    fit_rest: bool = False          # also fit on the "rest" entities (75% of S1 instead of 30%)
    easy_neg_keep: float = 1.0      # keep this share of easy negatives (p1 < 0.01 in easy_from's run), reweighted
    easy_from: str = ""             # MATCH version whose stage-1 train scores define easy negatives
    consensus: bool = False         # stage 2 sees house-number agreement with the S1's confident candidates
    eval_only: bool = False         # ablations: score train/eval only, no test predictions or submission files
    easy_screen: bool = False       # define easy negatives with a quick screening model when no easy_from run exists

    def lgb_params(self) -> dict:
        return dict(self.lgb)


def _index(*versions) -> dict:
    return {v.id: v for v in versions}


NORM = _index(
    Norm("NORM-v3t", "NORM-v2 + (1) train targets augmented with clones of real twin distractors up to the "
                     "twin density measured on test; (2) for countries without training links (France): city/region/"
                     "departement canon learned by co-occurrence, digit-corrupted street types (cour2) repaired from "
                     "S1's street-type vocabulary, bis/ter kept with the house number; (3) legal forms kept in the core "
                     "name where stripping them merges distinct S1 records (learned: France only)",
         twin_density=True, unseen_canon=True, legal_learned=True),
    Norm("NORM-v4t", "NORM-v3t + address components: street core, landmark clause, unit and PO box split off "
                     "(generic keywords; India landmarks, US suites/PMBs, French appt)",
         twin_density=True, unseen_canon=True, legal_learned=True, addr_components=True),
    Norm("NORM-v2", "anyascii transliteration; legal forms, street types, US states and ordinals canonicalised; "
                    "Indian state names joined; French street types; digit-for-letter fixes; fka/dba split; "
                    "website names turned into words; shree/shri/sri merged; per-country aliases learned "
                    "from training links"),
)

BLOCK = _index(
    Block("BLK-v1", "IDF-weighted sparse search on name and address tokens, 4-char prefixes, consonant skeletons "
                    "and house-number affixes; df cap 1%; top-30",
          k=30, k_noaddr=0, char_grams=False, joined_name=False, region_split=False),
    Block("BLK-v2", "BLK-v1 + joined-name token, name 3-grams, name-only pass (top-5) for targets without address",
          k=30, region_split=False),
    Block("BLK-v3", "BLK-v2 with a fixed df cap of 3,000 instead of 1% (rejected: recall -0.8 to -1.6 points)",
          k=30, region_split=False, max_df_frac=0.0, min_df_cap=3000),
    Block("BLK-v4", "BLK-v2 searched within regions: frequent last address tokens, merges learned from links",
          k=30, region_min_last_ratio=0.0),
    Block("BLK-v4b", "BLK-v4 with positional region keys (last token in at least 60% of its occurrences)", k=30),
    Block("BLK-v4b@20", "BLK-v4b with top-20 per source: the candidate set the matcher scores", k=20),
    Block("BLK-v4b@40", "BLK-v4b with top-40 per source: on full data top-20 finds only 98.3% of links "
                        "(99.45% on DEV-10), because regions hold 10x more rival records", k=40),
    Block("BLK-v4b@40n20", "BLK-v4b@40 with a top-20 name-only pass: measures both recall-vs-depth curves in one "
                           "run (targets without address: 88.7% recall at top-5 on full data)", k=40, k_noaddr=20),
)

FEAT = _index(
    Feat("FEAT-v1", "44 features: blocking score and cosines; rapidfuzz similarities of name, core name, joined "
                    "name and address; alias parts; token containment; house-number agreement; lengths and "
                    "flags; rank and competition context; S1 name frequency", align=False),
    Feat("FEAT-v2", "FEAT-v1 + 8 soft word-alignment features (52)"),
    Feat("FEAT-v3", "FEAT-v2 + 3 number-gap features (55)", number_gap=True),
    Feat("FEAT-v4", "FEAT-v3 + 9 house/legal features: parsed house number gap, digit edit, most significant "
                    "differing digit, same street key, near-number twin flag, bis/ter conflict, legal form equal/conflict",
         number_gap=True, twins=True),
    Feat("FEAT-v5", "FEAT-v4 + 7 address-component features: street-core ratio / token-set / partial, landmark "
                    "state and similarity, unit state, PO box state", number_gap=True, twins=True, addr_parts=True),
)

MATCH = _index(
    Match("MATCH-v1", "one LightGBM (binary log loss), cross-fitted; exclusive assignment; threshold",
          stages=1, rules=("threshold",)),
    Match("MATCH-v2", "MATCH-v1 + stage 2 on stage-1 probability context, cross-fitted; rule chosen on the eval "
                      "slice among threshold, expected F and gated expected F"),
    Match("MATCH-v3", "MATCH-v2 + support features in stage 2", support=True),
    Match("MATCH-v4", "EXPERIMENTAL, not validated: MATCH-v2 with LightGBM prediction early stopping (margin 10) to "
                      "cut scoring time (full-data models grow ~1,400 trees). On tiny data it moved stage-1 "
                      "probabilities by up to 0.31 and stage-2 ones further (their inputs shift): not a free speedup",
          pred_margin=10.0),
    Match("MATCH-v2x", "MATCH-v2 restricted to the FEAT-v2 columns and trained without synthetic twins: the current "
                       "(0.968) model re-created on a test-density data scope; threshold or gated rule only",
          cols="v2", train_synth=False, rules=("threshold", "gated_ef")),
    Match("MATCH-v5", "all FEAT-v4 columns + house-number consensus in stage 2; twin negatives weighted 1.9x; fitted "
                      "on fit+rest (75% of S1) with easy negatives (MATCH-v2x p1 < 0.01) kept at 10% and weighted 10x; "
                      "threshold or gated rule only",
          twin_weight=1.9, fit_rest=True, easy_neg_keep=0.1, easy_from="MATCH-v2x", consensus=True,
          rules=("threshold", "gated_ef")),
    Match("MATCH-v6", "MATCH-v5 + stage-2 support features (similarity to the S1's confident candidates): aimed at "
                      "targets without an address, 53% of M-v5's missed links",
          twin_weight=1.9, fit_rest=True, easy_neg_keep=0.1, easy_from="MATCH-v2x", consensus=True, support=True,
          rules=("threshold", "gated_ef")),
    Match("MATCH-v7", "MATCH-v6 with easy negatives defined by a quick screening model (for data scopes without a "
                      "MATCH-v2x run)",
          twin_weight=1.9, fit_rest=True, easy_neg_keep=0.1, easy_screen=True, consensus=True, support=True,
          rules=("threshold", "gated_ef")),
    # leave-one-out ablations of MATCH-v5 (eval only); each drops one of its four changes
    Match("MATCH-v5-nofeat", "ABLATION: MATCH-v5 without the 13 new FEAT-v4 columns (number gap, house, legal)",
          cols="v2", twin_weight=1.9, fit_rest=True, easy_neg_keep=0.1, easy_from="MATCH-v2x", consensus=True,
          rules=("threshold", "gated_ef"), eval_only=True),
    Match("MATCH-v5-nocons", "ABLATION: MATCH-v5 without the stage-2 house-number consensus",
          twin_weight=1.9, fit_rest=True, easy_neg_keep=0.1, easy_from="MATCH-v2x", consensus=False,
          rules=("threshold", "gated_ef"), eval_only=True),
    Match("MATCH-v5-notw", "ABLATION: MATCH-v5 without the 1.9x twin-negative weight",
          twin_weight=1.0, fit_rest=True, easy_neg_keep=0.1, easy_from="MATCH-v2x", consensus=True,
          rules=("threshold", "gated_ef"), eval_only=True),
    Match("MATCH-v5-fit30", "ABLATION: MATCH-v5 fitted on the fit entities only (30% of S1), easy negatives as in v5",
          twin_weight=1.9, fit_rest=False, easy_neg_keep=0.1, easy_from="MATCH-v2x", consensus=True,
          rules=("threshold", "gated_ef"), eval_only=True),
)

PRESETS = {
    "M-v1": ("NORM-v2", "BLK-v4b@20", "FEAT-v1", "MATCH-v1"),
    "M-v2": ("NORM-v2", "BLK-v4b@20", "FEAT-v2", "MATCH-v1"),
    "M-v3": ("NORM-v2", "BLK-v4b@20", "FEAT-v2", "MATCH-v2"),
    "M-v4": ("NORM-v2", "BLK-v4b@20", "FEAT-v3", "MATCH-v3"),
    "M-v5a": ("NORM-v3t", "BLK-v4b@20", "FEAT-v4", "MATCH-v2x"),
    "M-v5": ("NORM-v3t", "BLK-v4b@20", "FEAT-v4", "MATCH-v5"),
    "M-v7": ("NORM-v4t", "BLK-v4b@40n20", "FEAT-v5", "MATCH-v7"),
}
DEFAULT_PRESET = "M-v3"


@dataclass(frozen=True)
class RunVersions:
    norm: Norm
    block: Block
    feat: Feat
    match: Match
    preset: str | None = None

    def key(self, stage: str) -> str:
        """Cache directory name for ``stage``: the IDs of the versions it depends on."""
        return "__".join(getattr(self, part).id for part in STAGE_DEPS[stage])

    def describe(self) -> dict:
        return {"preset": self.preset, **{p: asdict(getattr(self, p)) for p in ("norm", "block", "feat", "match")}}


def resolve(preset: str | None = None, norm: str | None = None, block: str | None = None,
            feat: str | None = None, match: str | None = None) -> RunVersions:
    """A preset, optionally with some components swapped (the result then has no preset name)."""
    base = PRESETS[preset or DEFAULT_PRESET]
    ids = [c or b for c, b in zip((norm, block, feat, match), base)]
    swapped = any(c and c != b for c, b in zip((norm, block, feat, match), base))
    return RunVersions(NORM[ids[0]], BLOCK[ids[1]], FEAT[ids[2]], MATCH[ids[3]],
                       preset=None if swapped else (preset or DEFAULT_PRESET))
