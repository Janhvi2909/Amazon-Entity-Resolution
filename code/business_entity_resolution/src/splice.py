"""Splice two submissions by country: rows of ``--countries`` from the override, all others from the base.

Used for SUB-B: SUB-A (the 0.968 model, twin-filtered) everywhere, and the rows of the country
the training data cannot evaluate (France) from the pipeline run with its canonicalisation.
Candidate lists are spliced the same way, so every match stays inside its row's candidate list.

    python splice.py --base-matching A/matching_results.tsv --base-candidates A/candidate_pairs.tsv \
        --over-matching B/matching_results.tsv --over-candidates B/candidate_pairs.tsv \
        --data <dataset_dir> --countries France --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from ber.data import read_tsv


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("base-matching", "base-candidates", "over-matching", "over-candidates", "data", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--countries", default="France")
    a = ap.parse_args()
    s1 = read_tsv(Path(a.data) / "test" / "test_source1.tsv").select(pl.col("entity_id").alias("source1_entity_id"), "country")
    cs = a.countries.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"countries": cs}
    for kind, b, o, col in (("matching", a.base_matching, a.over_matching, "matched_entity_ids"),
                            ("candidates", a.base_candidates, a.over_candidates, "candidate_entity_ids")):
        B, O = read_tsv(b), read_tsv(o)
        if not (B.height == O.height == s1.height):
            raise RuntimeError(f"{kind}: row counts {B.height} / {O.height} / S1 {s1.height}")
        B = B.rename({B.columns[1]: col}).join(s1, on="source1_entity_id", how="left", maintain_order="left")
        O = O.rename({O.columns[1]: "over"}).select("source1_entity_id", "over")
        if B["country"].null_count():
            raise RuntimeError(f"{kind}: base has S1 ids missing from the test file")
        M = B.join(O, on="source1_entity_id", how="left", maintain_order="left")
        use = pl.col("country").is_in(cs)
        M = M.with_columns(pl.when(use).then(pl.col("over")).otherwise(pl.col(col)).fill_null("").alias(col))
        n = lambda c: M.filter(use).select(pl.col(c).fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().sum()).item()
        rep[kind] = {"rows_replaced": int(M.filter(use).height), "ids_before": int(n(col if False else col)), }
        rep[kind]["ids_base_in_replaced_rows"] = int(B.filter(pl.col("country").is_in(cs)).select(
            pl.col(col).fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().sum()).item())
        rep[kind]["ids_override_in_replaced_rows"] = int(n(col))
        M.select("source1_entity_id", col).write_csv(out / ("matching_results.tsv" if kind == "matching" else "candidate_pairs.tsv"),
                                                     separator="\t", quote_style="never")
    (out / "splice_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
