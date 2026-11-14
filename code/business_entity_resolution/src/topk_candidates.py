"""Candidate file = a run's top-k test candidates per S1 by matcher probability, plus every link of a matching file.

Links outside the top-k are kept (forced inclusion) and counted per country, so a matching file whose
links came from another model shows how far it sits from this run's ranking.

    python topk_candidates.py --run <work>/runs/<key> --prep <work>/prep/<norm> --matching m.tsv --out cand.tsv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from ber.data import read_tsv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--matching", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--score", default="p2")
    a = ap.parse_args()
    S = pl.read_parquet(Path(a.run) / "scores_test.parquet", columns=["q_rid", "t_rid", a.score])
    Q = pl.read_parquet(Path(a.prep) / "Q_test.parquet", columns=["rid", "entity_id", "country"])
    T = pl.read_parquet(Path(a.prep) / "T_test.parquet", columns=["rid", "entity_id"])
    top = (S.with_columns(pl.col(a.score).rank("ordinal", descending=True).over("q_rid").alias("r"))
           .filter(pl.col("r") <= a.k)
           .join(Q.select(pl.col("rid").alias("q_rid"), pl.col("entity_id").alias("s1")), on="q_rid")
           .join(T.select(pl.col("rid").alias("t_rid"), pl.col("entity_id").alias("t_id")), on="t_rid")
           .select("s1", "t_id"))
    m = read_tsv(a.matching)
    m = m.rename({m.columns[0]: "s1", m.columns[1]: "ids"})
    links = (m.with_columns(pl.col("ids").fill_null("").str.split(",")).explode("ids")
             .filter(pl.col("ids") != "").select("s1", pl.col("ids").alias("t_id")))
    forced = links.join(top, on=["s1", "t_id"], how="anti")
    cand = pl.concat([top, forced]).unique()
    per = forced.join(Q.select(pl.col("entity_id").alias("s1"), "country"), on="s1").group_by("country").len()
    out = (m.select("s1").join(cand.sort("s1", "t_id").group_by("s1", maintain_order=True)
                               .agg(pl.col("t_id").str.join(",").alias("candidate_entity_ids")), on="s1", how="left",
                               maintain_order="left")
           .with_columns(pl.col("candidate_entity_ids").fill_null("")).rename({"s1": "source1_entity_id"}))
    if out.height != m.height:
        raise RuntimeError("row count changed")
    out.write_csv(a.out, separator="\t", quote_style="never")
    rep = {"k": a.k, "score": a.score, "links": links.height, "candidates": cand.height,
           "cand_per_s1": cand.height / m.height, "forced": forced.height,
           "forced_by_country": dict(per.iter_rows())}
    Path(a.out + ".report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
