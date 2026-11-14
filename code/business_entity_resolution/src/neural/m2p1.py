"""M2 without stage 2: score tables from M2's checkpointed stage-1 predictions (p1), then its uncertain band.

M2's stage 2 (context features over 253M train pairs) took nst-n7 down (memory), so the final stacks the
cross-encoders on M2's stage-1 probabilities instead. The p1 prediction parts were written in candidate
order (one .npy per feature part, each with a marker); this rebuilds scores_{train,test}.parquet with
p2 := p1 so the band extractor and the stacker run unchanged.
"""
import sys
from pathlib import Path
import numpy as np
import polars as pl

W = "/work/full"; R = f"{W}/runs/NORM-v4t__BLK-v4b@40n20__FEAT-v5__MATCH-v7"; BLK = f"{W}/block/NORM-v4t__BLK-v4b@40n20"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
for split in ("train", "test"):
    C = pl.read_parquet(f"{BLK}/cand_{split}.parquet", columns=["q_rid", "t_rid", "src"])
    parts = sorted(Path(R, "pred").glob(f"p1_{split}-*.npy"))
    for p in parts:
        if not Path(str(p) + ".ok").exists():
            raise SystemExit(f"FATAL: {p} has no marker")
    p1 = np.concatenate([np.load(p) for p in parts])
    if len(p1) != C.height:
        raise SystemExit(f"FATAL: {split}: {len(p1):,} predictions for {C.height:,} candidates")
    S = C.with_columns(pl.Series("p1", p1.astype(np.float32)), pl.Series("p2", p1.astype(np.float32)))
    S.write_parquet(OUT / f"scores_{split}.parquet")
    print(split, S.height, "band share", float(((p1 >= 0.02) & (p1 <= 0.98)).mean()), flush=True)
print("done", flush=True)
