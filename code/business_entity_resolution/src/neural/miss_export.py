"""Export eval blocking misses (true links absent from the candidate set) and the records retrieval needs.

Streams the candidate files (only eval S1 rows are read), so memory stays small. Writes to <out>:
  misses.parquet   q_rid, t_rid, country, missed_by (M2 = BLK-v4b@40n20, M5 = BLK-v4b@20)
  Q.parquet / T.parquet   normalised train records (rid, country, name_n, addr_n, name_nosp, has_addr, entity_id)
"""
import sys
from pathlib import Path
import numpy as np
import polars as pl

W = "/work/full"; OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
P2 = f"{W}/prep/NORM-v4t"
role = pl.read_parquet(f"{P2}/split.parquet")
ev = role.filter(pl.col("role") == 2).select("q_rid")
pairs = pl.read_parquet(f"{P2}/pairs.parquet").join(ev, on="q_rid", how="semi")
cols = ["rid", "country", "name_n", "addr_n", "name_nosp", "has_addr", "entity_id"]
Q = pl.read_parquet(f"{P2}/Q_train.parquet", columns=cols)
out = []
for tag, blk in (("M2", f"{W}/block/NORM-v4t__BLK-v4b@40n20/cand_train.parquet"), ("M5", f"{W}/block/NORM-v3t__BLK-v4b@20/cand_train.parquet")):
    C = pl.scan_parquet(blk).select("q_rid", "t_rid").join(ev.lazy(), on="q_rid", how="semi").collect(engine="streaming")
    miss = pairs.join(C, on=["q_rid", "t_rid"], how="anti").with_columns(pl.lit(tag).alias("missed_by"))
    print(tag, "eval true links", pairs.height, "missed", miss.height, f"recall {1 - miss.height / pairs.height:.4f}", flush=True)
    out.append(miss)
    del C
M = pl.concat(out).join(Q.select(pl.col("rid").alias("q_rid"), "country"), on="q_rid")
M.write_parquet(OUT / "misses.parquet")
print(M.group_by("missed_by", "country").len().sort("missed_by", "country"), flush=True)
Q.write_parquet(OUT / "Q.parquet")
pl.read_parquet(f"{P2}/T_train.parquet", columns=cols + ["src"]).write_parquet(OUT / "T.parquet")
print("written", flush=True)
