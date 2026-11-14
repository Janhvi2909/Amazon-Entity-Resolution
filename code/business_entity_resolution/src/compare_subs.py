"""Link profile of a submission against a base: links added / removed per country, and how many are twins.

A change that adds links on test is suspect until this says what it added (another team saw
validation rise and the leaderboard fall exactly that way).

    python compare_subs.py --base floor/matching_results.tsv --new SUB-C/matching_results.tsv --data <dataset_dir>
"""
from __future__ import annotations

import argparse
import json

import polars as pl

from ber.data import load_split
from ber.twins import twin_links
from twin_filter import houses, read_lists


def long(path: str) -> pl.DataFrame:
    return (read_lists(path, "ids").with_columns(pl.col("ids").fill_null("").str.split(",")).explode("ids")
            .filter(pl.col("ids") != "").rename({"ids": "t_id"}).select("s1", "t_id"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    s1, T = load_split(a.data, "test")
    Q = s1.with_row_index("rid")
    B, N = long(a.base), long(a.new)
    added = N.join(B, on=["s1", "t_id"], how="anti").with_columns(pl.lit("added").alias("kind"))
    removed = B.join(N, on=["s1", "t_id"], how="anti").with_columns(pl.lit("removed").alias("kind"))
    D = pl.concat([added, removed])
    Tl = T.join(D.select(pl.col("t_id").alias("entity_id")).unique(), on="entity_id", how="semi").with_row_index("rid")
    L = (D.join(Q.select(pl.col("entity_id").alias("s1"), pl.col("rid").alias("q_rid"), "country"), on="s1")
         .join(Tl.select(pl.col("entity_id").alias("t_id"), pl.col("rid").alias("t_rid")), on="t_id"))
    Qs = Q.join(L.select(pl.col("q_rid").alias("rid")).unique(), on="rid", how="semi")
    L = L.with_columns(twin_links(L, houses(Qs), houses(Tl), 1, 30).alias("near"))
    rep = {"links_base": B.height, "links_new": N.height, "added": added.height, "removed": removed.height,
           "s1_changed": int(D["s1"].n_unique()), "by_country": {}}
    for r in L.group_by("country", "kind").agg(pl.len().alias("n"), pl.col("near").sum().alias("near_number")).sort("country", "kind").iter_rows(named=True):
        rep["by_country"].setdefault(r["country"], {})[r["kind"]] = {"links": r["n"], "near_number_1_30": r["near_number"]}
    txt = json.dumps(rep, indent=2)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
