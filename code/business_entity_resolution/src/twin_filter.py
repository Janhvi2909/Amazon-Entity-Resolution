"""Post-filter a matching_results.tsv: drop links to near-number twins (SUB-A).

A link is dropped when the candidate's street key equals the S1 record's and its house
number is ``--lo``..``--hi`` away, for S1 records of the ``--countries`` given. No model is
involved; see ber/twins.py for the evidence.

    python twin_filter.py --matching in.tsv --data <dataset_dir> --out out.tsv --countries US,France
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from ber.data import load_split
from ber.normalize import ADDR_CANON, clean_addresses
from ber.twins import house_frame, twin_links


def read_lists(path: str | Path, col: str) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return df.rename({df.columns[0]: "s1", df.columns[1]: col})


def houses(df: pl.DataFrame) -> pl.DataFrame:
    a = clean_addresses(df)
    a = a.with_columns(pl.col("a").str.split(" ").list.eval(pl.element().replace(ADDR_CANON)).list.join(" "))
    return house_frame(a["rid"], a["a"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--countries", default="US,France")
    ap.add_argument("--lo", type=int, default=2)
    ap.add_argument("--hi", type=int, default=30)
    a = ap.parse_args()
    m = read_lists(a.matching, "ids")
    s1, T = load_split(a.data, "test")
    Q = s1.with_row_index("rid")
    links = (m.with_columns(pl.col("ids").fill_null("").str.split(",")).explode("ids").filter(pl.col("ids") != "")
             .rename({"ids": "t_id"}))
    n_in = links.height
    assert n_in > 0, "no links in the input file"
    Tl = T.join(links.select(pl.col("t_id").alias("entity_id")).unique(), on="entity_id", how="semi").with_row_index("rid")
    L = (links.join(Q.select(pl.col("entity_id").alias("s1"), pl.col("rid").alias("q_rid"), "country"), on="s1")
         .join(Tl.select(pl.col("entity_id").alias("t_id"), pl.col("rid").alias("t_rid")), on="t_id"))
    assert L.height == n_in, f"join lost links: {n_in} -> {L.height}"
    Qs = Q.join(L.select(pl.col("q_rid").alias("rid")).unique(), on="rid", how="semi")
    flag = twin_links(L, houses(Qs), houses(Tl), a.lo, a.hi)
    countries = a.countries.split(",")
    drop = flag & L["country"].is_in(countries)
    kept = L.filter(~drop)
    report = {"links_in": n_in, "links_removed": int(drop.sum()), "s1_affected": int(L.filter(drop)["s1"].n_unique()),
              "by_country": {c: {"removed": int((drop & (L["country"] == c)).sum()),
                                 "flagged_all": int((flag & (L["country"] == c)).sum())}
                             for c in sorted(L["country"].unique().to_list())},
              "rule": {"lo": a.lo, "hi": a.hi, "countries": countries}}
    ids = (kept.sort("s1", "t_id").group_by("s1", maintain_order=True).agg(pl.col("t_id").str.join(",").alias("matched_entity_ids")))
    out = (m.select("s1").join(ids, on="s1", how="left", maintain_order="left")
           .with_columns(pl.col("matched_entity_ids").fill_null(""))
           .rename({"s1": "source1_entity_id"}))
    assert out.height == m.height, "row count changed"
    out.write_csv(a.out, separator="\t", quote_style="never")
    Path(a.out + ".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
