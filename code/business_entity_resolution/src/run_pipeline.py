"""End-to-end pipeline: data -> normalise -> block -> features -> LightGBM -> decision -> output.

Usage (from src/):
    python run_pipeline.py --data <dataset_dir> --work <work_dir> --out <output_dir>   # preset M-v3
    python run_pipeline.py ... --preset M-v4                    # another end-to-end version
    python run_pipeline.py ... --feat FEAT-v3 --stages features,train,predict   # swap one component
    python run_pipeline.py ... --sample 0.01                    # quick end-to-end smoke run
    python run_pipeline.py --list                               # every registered version

Every component comes from a named version in ber/versions.py (NORM, BLK, FEAT, MATCH; the
IDs of method_result.md), and each stage caches its output under the versions it depends on:
  <work>/prep/NORM-v2/                               normalised records, links, split, aliases
  <work>/block/NORM-v2__BLK-v4b@20/                  candidates, region spec
  <work>/feat/NORM-v2__BLK-v4b@20__FEAT-v2/          features, in parts
  <work>/runs/NORM-v2__BLK-v4b@20__FEAT-v2__MATCH-v2/  models, scores, metrics.json
so changing one version recomputes only what depends on it. One work dir holds one data
scope (--sample/--seed/--fit/--es/--eval); prep refuses to be reused with another.

Checkpointing (ber/ckpt.py). Every artifact gets a marker only after its file is closed, and
is trusted only after it is re-opened and agrees with the marker. A stage whose <stage>.json
and artifacts are all good is skipped; blocking resumes per (split, country), features per
part, training per fold model. A status file (<work>/PROGRESS.txt) is refreshed every 2 minutes.

Validation. Train S1 entities are split once (seed 42) into fit / early-stopping / eval /
rest. Every entity stays in the candidate search, so targets are contested as densely as
at test time. Aliases and region merges are learned without the eval entities' links, no
model sees eval or rest entities, and those rows are scored exactly like test rows (the
mean of the fold models), so the eval score estimates the test score. With NORM-v3t the
train targets also hold clones of real twin distractors up to the twin density measured on
test: "new eval" scores that test-density slice, "old eval" the same rows with the clones
removed.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from ber import ckpt as K
from ber import model as M
from ber import versions as V
from ber.aliases import learn_aliases
from ber.blocking import (candidates_at_k, feature_frame, generate_candidates, learn_region_merges, noaddr_recall_at_k,
                          recall_at_k, recall_report)
from ber.canon import apply_canon, learn_canon
from ber.data import dev_sample, load_split, load_truth
from ber import features as FT
from ber.features import (HOUSE_FEATS, NUMGAP_FEATS, align_features, consensus_features, context_features, name_idf,
                          string_features, support_features)
from ber.metrics import macro_f05
from ber.normalize import normalize, token_frames
from ber.progress import PROGRESS, cgroup_mem
from ber.twins import augment_twins, twin_links, twin_rates

FIT, ES, EVAL, REST = 0, 1, 2, 3
ROLE_NAMES = {FIT: "fit", ES: "early_stop", EVAL: "eval", REST: "rest"}
STAGES = ("prep", "block", "features", "train", "predict")
PARAMS = {"threshold": np.round(np.arange(0.20, 0.96, 0.025), 3), "expected_f": (0.05, 0.1, 0.2, 0.3, 0.4),
          "gated_ef": np.round(np.arange(0.30, 0.96, 0.05), 3)}
# test share of S1 records per country (1,732,544 test S1): the weighting of the leaderboard
TEST_WEIGHTS = {"India": 809986, "US": 663106, "France": 259452}
TWIN_FILTER_COUNTRIES = ("US", "France")  # SUB-A rule: India house numbers parse unreliably
CAND_TOP = 15

_T0 = time.time()


def peak_gb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # Linux: KiB
    except ImportError:  # Windows
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**30


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.0f}s  peak {peak_gb():5.1f} GB] {msg}", flush=True)


def headroom(where: Path, need_disk_gb: float, need_ram_gb: float, what: str) -> None:
    """Log free disk and RAM against the projected need; warn loudly when tight."""
    free = shutil.disk_usage(where).free / 2**30
    used, lim = cgroup_mem()
    ram_free = lim - used
    msg = (f"headroom before {what}: disk {free:.1f} GB free (need ~{need_disk_gb:.0f}), "
           f"RAM {ram_free:.1f} GB free of {lim:.1f} (need ~{need_ram_gb:.0f})")
    log(("WARNING TIGHT: " if free < need_disk_gb or ram_free < need_ram_gb else "") + msg)


class Paths:
    """Stage directories named by the versions each stage depends on."""

    def __init__(self, work: Path, rv: V.RunVersions):
        self.work = work
        self.prep = work / "prep" / rv.key("prep")
        self.block = work / "block" / rv.key("block")
        self.feat = work / "feat" / rv.key("features")
        self.run = work / "runs" / rv.key("train")
        self.rv = rv

    def stage_dir(self, stage: str) -> Path:
        return {"prep": self.prep, "block": self.block, "features": self.feat}.get(stage, self.run)

    def run_of(self, match_id: str) -> Path:
        return self.work / "runs" / "__".join([self.rv.norm.id, self.rv.block.id, self.rv.feat.id, match_id])


def read_json(f: Path) -> dict:
    return json.loads(f.read_text()) if f.exists() else {}


def write_json(f: Path, d: dict) -> None:
    K.write_json(d, f)


def data_scope(a) -> dict:
    return {"sample": a.sample, "seed": a.seed, "fit": a.fit, "es": a.es, "eval": a.eval}


def check_scope(a, P: Paths) -> None:
    cached = read_json(P.prep / "prep.json").get("scope")
    if cached is None:
        raise SystemExit(f"{P.prep} has no prep output yet: run the prep stage first.")
    if cached != data_scope(a):
        raise SystemExit(f"{P.prep} was built with {cached}, not {data_scope(a)}. Use another --work dir.")


def stage_complete(P: Paths, stage: str) -> bool:
    """A stage is done only if its json is good and every artifact it lists opens and matches its marker."""
    f = P.stage_dir(stage) / f"{stage}.json"
    if not K.good(f):
        return False
    arts = read_json(f).get("artifacts", [])
    return bool(arts) and all(K.good(P.stage_dir(stage) / x) for x in arts)


def learning_pairs(P: Paths) -> pl.DataFrame:
    """True links of the non-eval train entities: the only links aliases and region merges may learn from."""
    split = pl.read_parquet(P.prep / "split.parquet")
    return pl.read_parquet(P.prep / "pairs.parquet").join(
        split.filter(pl.col("role") != EVAL).select("q_rid"), on="q_rid", how="semi")


def n_real_targets(P: Paths) -> int:
    """Train target rows before the synthetic twin clones (which are appended at the end)."""
    return int(read_json(P.prep / "prep.json").get("train_targets_real", 1 << 40))


# ---------------------------------------------------------------------- prep
def stage_prep(a, rv: V.RunVersions, P: Paths) -> dict:
    nv = rv.norm
    PROGRESS.begin("prep", total=6, unit="steps")
    tr_s1, tr_t = load_split(a.data, "train")
    te_s1, te_t = load_split(a.data, "test")
    truth = load_truth(a.data)
    if a.sample < 1:
        tr_s1, tr_t, truth = dev_sample(tr_s1, tr_t, truth, a.sample, seed=a.seed)
        te_s1 = te_s1.sample(fraction=a.sample, seed=a.seed)
        te_t = te_t.sample(fraction=a.sample, seed=a.seed)
    K.require(tr_s1.height, "train S1"); K.require(tr_t.height, "train targets"); K.require(te_s1.height, "test S1")
    Q, T = tr_s1.with_row_index("rid"), tr_t.with_row_index("rid")
    Qt, Tt = te_s1.with_row_index("rid"), te_t.with_row_index("rid")
    pairs = (truth.join(Q.select(pl.col("entity_id").alias("s1_id"), pl.col("rid").alias("q_rid")), on="s1_id")
             .join(T.select(pl.col("entity_id").alias("t_id"), pl.col("rid").alias("t_rid")), on="t_id")
             .select("q_rid", "t_rid"))
    K.require(pairs.height, "train links after joining ids", minimum=truth.height // 2)
    out = {"scope": data_scope(a)}
    n_real = T.height
    if nv.twin_density:
        # target: the twin density measured on test, for each country that has training links
        rates = twin_rates(Qt, Tt)
        link_countries = Q.join(pairs.select(pl.col("q_rid").alias("rid")).unique(), on="rid", how="semi")["country"].unique().to_list()
        target = {c: r for c, r in rates.items() if c in link_countries}
        clones, rep = augment_twins(Q, T, pairs, target, seed=a.seed)
        log(f"twin augmentation: {rep}")
        out["twin_augmentation"] = rep
        out["twin_rates_test"] = rates
        if clones.height:
            T = pl.concat([T, clones.with_columns(pl.int_range(n_real, n_real + clones.height, dtype=T["rid"].dtype).alias("rid"))
                           .select(T.columns)])
    PROGRESS.step(note="loaded data, twin clones")
    log(f"train S1 {Q.height:,} targets {T.height:,} ({T.height - n_real:,} synthetic) links {pairs.height:,} | "
        f"test S1 {Qt.height:,} targets {Tt.height:,}")

    # one fixed split of the train S1 entities
    u = np.random.default_rng(a.seed).random(Q.height)
    role = np.full(Q.height, REST, np.int8)
    role[u < a.fit] = FIT
    role[(u >= a.fit) & (u < a.fit + a.es)] = ES
    role[u >= 1 - a.eval] = EVAL
    split = pl.DataFrame({"q_rid": np.arange(Q.height, dtype=np.uint32), "role": role})
    log("split: " + ", ".join(f"{ROLE_NAMES[r]} {int((role == r).sum()):,}" for r in ROLE_NAMES))

    canon = None
    if nv.unseen_canon or nv.legal_learned:
        link_countries = Q.join(pairs.select(pl.col("q_rid").alias("rid")).unique(), on="rid", how="semi")["country"].unique().to_list()
        canon = learn_canon(tr_s1, te_s1, te_t, link_countries, nv.unseen_canon, nv.legal_learned)
        canon["addr_components"] = nv.addr_components
        log(f"canon: keep_legal {canon['keep_legal']}, unseen {canon['unseen']}, "
            f"street fixes {canon['street_fix']}, geo keys {({c: len(g) for c, g in canon['geo'].items()})}")
        K.write_json(canon, P.prep / "canon_spec.json")
        Q, T, Qt, Tt = (apply_canon(d, canon) for d in (Q, T, Qt, Tt))
    PROGRESS.step(note="canon learned")

    learn = pairs.join(split.filter(pl.col("role") != EVAL).select("q_rid"), on="q_rid", how="semi")
    _, nq, aq = token_frames(Q, canon=canon)
    _, nt, at = token_frames(T, canon=canon)
    kw = dict(min_count=nv.alias_min_count, min_ratio=nv.alias_min_ratio, max_unaligned=nv.alias_max_unaligned)
    name_alias = learn_aliases(nq, nt, learn, **kw).select("country", "tok", "to")
    addr_alias = learn_aliases(aq, at, learn, **kw).select("country", "tok", "to")
    del nq, aq, nt, at
    log(f"aliases: {name_alias.height} name, {addr_alias.height} address")
    PROGRESS.step(note="aliases learned")

    P.prep.mkdir(parents=True, exist_ok=True)
    arts = []
    for name, df in (("Q_train", Q), ("T_train", T), ("Q_test", Qt), ("T_test", Tt)):
        f = P.prep / f"{name}.parquet"
        if not K.good(f, min_rows=df.height):
            n = normalize(df, name_alias, addr_alias, canon=canon).drop("business_name", "business_address")
            K.require(n.height, f"normalised {name}", minimum=df.height)
            K.write_parquet(n, f)
        arts.append(f.name)
        log(f"normalised {name}")
        PROGRESS.step(rows=df.height, note=f"normalised {name}")
    for name, df in (("pairs", pairs), ("split", split), ("name_alias", name_alias), ("addr_alias", addr_alias)):
        K.write_parquet(df, P.prep / f"{name}.parquet", allow_empty=name.endswith("alias"))
        arts.append(f"{name}.parquet")
    out.update({"train_s1": Q.height, "train_targets": T.height, "train_targets_real": n_real, "train_links": pairs.height,
                "test_s1": Qt.height, "test_targets": Tt.height,
                "split": {ROLE_NAMES[r]: int((role == r).sum()) for r in ROLE_NAMES},
                "aliases": {"name": name_alias.height, "address": addr_alias.height},
                "canon": None if canon is None else {k: canon[k] for k in ("keep_legal", "unseen", "street_fix", "legal_report")},
                "artifacts": arts})
    return out


# --------------------------------------------------------------------- block
def stage_block(a, rv: V.RunVersions, P: Paths) -> dict:
    bv = rv.block
    spec = None
    if bv.region_split:
        spec = learn_region_merges(pl.read_parquet(P.prep / "Q_train.parquet", columns=["rid", "country", "addr_n"]),
                                   pl.read_parquet(P.prep / "T_train.parquet", columns=["rid", "country", "addr_n"]), learning_pairs(P),
                                   min_share=bv.region_min_share, min_cross=bv.region_min_cross,
                                   min_last_ratio=bv.region_min_last_ratio)
        log(f"region spec: {spec}")
    P.block.mkdir(parents=True, exist_ok=True)
    write_json(P.block / "region_spec.json", spec or {})
    out = {"region_spec": spec}
    arts = ["region_spec.json"]
    K.clean_dir(P.block, "cand_*.parquet")
    BLOCK_COLS = ["rid", "country", "name_n", "addr_n", "name_nosp", "has_addr"]  # all that blocking reads
    for split in ("train", "test"):
        Q = pl.read_parquet(P.prep / f"Q_{split}.parquet", columns=BLOCK_COLS)
        T = pl.read_parquet(P.prep / f"T_{split}.parquet", columns=BLOCK_COLS + ["src"])
        countries = Q["country"].unique().sort().to_list()
        done = [c for c in countries if K.good(P.block / f"cand_{split}__{c}.parquet")]
        PROGRESS.begin(f"block {split}", total=len(countries), unit="countries", skipped=len(done))
        headroom(P.block, 10, 45, f"block {split}")
        parts = []
        for country in countries:  # one country at a time bounds memory
            f = P.block / f"cand_{split}__{country}.parquet"
            if K.good(f):
                parts.append(pl.read_parquet(f))
                log(f"block {split}/{country}: resumed from {f.name} ({parts[-1].height:,} rows)")
                continue
            Qc, Tc = Q.filter(pl.col("country") == country), T.filter(pl.col("country") == country)
            if Tc.height == 0:
                continue
            t0 = time.time()
            ff = lambda df: feature_frame(df, char_grams=bv.char_grams, joined_name=bv.joined_name)
            C = generate_candidates(Qc, Tc, ff(Qc), ff(Tc), k=bv.k, k_noaddr=bv.k_noaddr, max_df_frac=bv.max_df_frac,
                                    min_df_cap=bv.min_df_cap, region_spec=spec)
            K.require(C.height, f"candidates {split}/{country}", minimum=Qc.height // 2)
            K.write_parquet(C, f)
            parts.append(C)
            log(f"block {split}/{country}: S1 {Qc.height:,} x targets {Tc.height:,} -> "
                f"{C.height:,} candidates ({time.time() - t0:.0f}s)")
            PROGRESS.step(rows=C.height, note=f"{split}/{country} {C.height:,} candidates")
        C = pl.concat(parts).sort("q_rid", "src", "bscore", descending=[False, False, True])
        K.require(C.height, f"candidates {split}", minimum=Q.height)
        K.write_parquet(C, P.block / f"cand_{split}.parquet")
        arts.append(f"cand_{split}.parquet")
        log(f"block {split}: {C.height:,} candidates, {C.height / Q.height:.1f} per S1")
        out[split] = {"candidates": C.height, "per_s1": C.height / Q.height}
        if split == "train":
            pairs = pl.read_parquet(P.prep / "pairs.parquet")
            ev = pl.read_parquet(P.prep / "split.parquet").filter(pl.col("role") == EVAL).select("q_rid")
            pairs_ev = pairs.join(ev, on="q_rid", how="semi")
            out["train"]["recall_all"] = recall_report(C, pairs)
            out["train"]["recall_eval"] = recall_report(C.join(ev, on="q_rid", how="semi"), pairs_ev)
            Ce = C.join(ev, on="q_rid", how="semi")
            out["train"]["recall_at_k_eval"] = recall_at_k(Ce, pairs_ev)
            out["train"]["recall_at_k_noaddr_eval"] = noaddr_recall_at_k(Ce, pairs_ev)
            out["train"]["cand_per_s1_at_k"] = candidates_at_k(C, Q.height)
            out["train"]["cand_per_s1_at_k_noaddr"] = candidates_at_k(C, Q.height, noaddr=True)
            nr = n_real_targets(P)
            out["train"]["synthetic_candidates"] = int((C["t_rid"] >= nr).sum())
            log(f"blocking recall: all {out['train']['recall_all']['pair_recall']:.4f}, "
                f"eval {out['train']['recall_eval']['pair_recall']:.4f}; synthetic-target candidates "
                f"{out['train']['synthetic_candidates']:,}")
        del C, parts
    out["artifacts"] = arts
    return out


# ------------------------------------------------------------------ features
_W = {}


def _feat_worker_init(prep: str, split: str, fv_align: bool, rf_threads: int) -> None:
    FT.RF_WORKERS = rf_threads
    q = pl.read_parquet(Path(prep) / f"Q_{split}.parquet")
    t = pl.read_parquet(Path(prep) / f"T_{split}.parquet")
    _W.update(Q=q, T=t, idf=name_idf(q) if fv_align else None)


def _feat_worker(job: tuple) -> tuple[str, int, float]:
    inp, outp, number_gap, twins, align, addr_parts = job
    t0 = time.time()
    c = pl.read_parquet(inp)
    base = c.select("q_rid", "t_rid", pl.col("src").cast(pl.Float32), "bscore", "cos_name", "cos_addr",
                    pl.col("noaddr_pass").cast(pl.Float32))
    ctx = c.select([x for x in c.columns if x.startswith("ctx__")]).rename(lambda x: x[5:])
    cols = [base, string_features(c, _W["Q"], _W["T"], number_gap=number_gap, twins=twins, addr_parts=addr_parts), ctx]
    if align:
        cols.append(align_features(c, _W["Q"], _W["T"], _W["idf"]))
    part = pl.concat(cols, how="horizontal")
    if part.height != c.height:
        raise RuntimeError(f"feature rows {part.height} != input rows {c.height} for {outp}")
    K.write_parquet(part, outp)
    Path(inp).unlink(missing_ok=True)
    Path(str(inp) + ".ok").unlink(missing_ok=True)
    return outp, part.height, time.time() - t0


def stage_features(a, rv: V.RunVersions, P: Paths) -> dict:
    fv = rv.feat
    out, arts = {}, []
    for split in ("train", "test"):
        d = P.feat / split
        d.mkdir(parents=True, exist_ok=True)
        kept, deleted = K.clean_dir(d, "part-*.parquet")
        K.clean_dir(d, "in-*.parquet")
        if deleted:
            log(f"features {split}: deleted {len(deleted)} torn parts: {[p.name for p in deleted]}")
        C = pl.read_parquet(P.block / f"cand_{split}.parquet")
        n = a.part_rows
        n_parts = -(-C.height // n)
        todo = [i for i in range(n_parts) if not K.good(d / f"part-{i:04d}.parquet")]
        PROGRESS.begin(f"features {split}", total=n_parts, unit="parts", skipped=n_parts - len(todo))
        log(f"features {split}: {n_parts} parts, {n_parts - len(todo)} already good, {len(todo)} to compute")
        if todo:
            Q = pl.read_parquet(P.prep / f"Q_{split}.parquet")
            T = pl.read_parquet(P.prep / f"T_{split}.parquet")
            # the context features need every candidate at once; the rest are row-wise
            ctx = context_features(C, Q, T)
            del Q, T
            for i in todo:  # inputs for the workers: candidates + their context columns
                s = i * n
                K.write_parquet(pl.concat([C.slice(s, n), ctx.slice(s, n).rename(lambda x: "ctx__" + x)], how="horizontal"),
                                d / f"in-{i:04d}.parquet")
            del ctx
            workers = max(1, min(a.feat_workers, len(todo)))
            threads = max(1, (os.cpu_count() or 8) // workers)
            headroom(d, 0.25 * len(todo), 6 * workers + 6, f"features {split} ({workers} workers x {threads} threads)")
            os.environ["POLARS_MAX_THREADS"] = str(threads)
            jobs = [(str(d / f"in-{i:04d}.parquet"), str(d / f"part-{i:04d}.parquet"), fv.number_gap, fv.twins, fv.align, fv.addr_parts)
                    for i in todo]
            with mp.get_context("spawn").Pool(workers, initializer=_feat_worker_init,
                                              initargs=(str(P.prep), split, fv.align, threads)) as pool:
                for outp, rows, sec in pool.imap_unordered(_feat_worker, jobs):
                    PROGRESS.step(rows=rows, note=f"{Path(outp).name} {rows:,} rows in {sec:.0f}s")
                    log(f"features {split}: {Path(outp).name} {rows:,} rows ({sec:.0f}s)")
        parts = sorted(d.glob("part-*.parquet"))
        rows = sum(K._rows(p) for p in parts)
        if len(parts) != n_parts or rows != C.height or not all(K.good(p) for p in parts):
            raise RuntimeError(f"features {split}: {len(parts)} parts / {rows:,} rows, expected {n_parts} / {C.height:,}")
        width = len(pl.read_parquet_schema(parts[0]))
        arts += [f"{split}/{p.name}" for p in parts]
        out[split] = {"rows": C.height, "parts": n_parts, "features": width - 2}
    out["artifacts"] = arts
    return out


# --------------------------------------------------------------------- train
def _parts(P: Paths, split: str) -> list[Path]:
    parts = sorted((P.feat / split).glob("part-*.parquet"))
    bad = [p for p in parts if not K.good(p)]
    if bad or not parts:
        raise RuntimeError(f"feature parts not good for {split}: {[p.name for p in bad] or 'none found'}")
    return parts


def _labels(df: pl.DataFrame, pairs: pl.DataFrame) -> np.ndarray:
    return (df.select("q_rid", "t_rid")
            .join(pairs.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["q_rid", "t_rid"], how="left",
                  maintain_order="left")["y"].fill_null(0).to_numpy())


def _load_rows(parts: list[Path], keep: np.ndarray, cols: list[str], pairs: pl.DataFrame,
               extra: np.ndarray | None = None):
    """Feature rows flagged in ``keep`` (one flag per candidate, candidate order).

    ``extra`` (one row per candidate, in candidate order) is appended as further columns.
    Returns X (float32), y, q_rid, and each row's index in the full candidate order.
    """
    n = int(keep.sum())
    K.require(n, "training rows selected")
    width = len(cols) + (0 if extra is None else extra.shape[1])
    X = np.empty((n, width), np.float32)
    y = np.empty(n, np.int8)
    q = np.empty(n, np.uint32)
    rows = np.empty(n, np.int64)
    at = off = 0
    for p in parts:
        size = K._rows(p)
        m = keep[off:off + size]
        k = int(m.sum())
        if k:
            df = pl.read_parquet(p, columns=["q_rid", "t_rid"] + cols).filter(pl.Series(m))
            X[at:at + k, :len(cols)] = df.select(cols).to_numpy()
            y[at:at + k] = _labels(df, pairs)
            q[at:at + k] = df["q_rid"].to_numpy()
            rows[at:at + k] = off + np.flatnonzero(m)
            if extra is not None:
                X[at:at + k, len(cols):] = extra[rows[at:at + k]]
            at += k
        off += size
    return X, y, q, rows


def _predict(models, X: np.ndarray, q: np.ndarray, role: np.ndarray | None, fold: np.ndarray | None,
             margin: float | None = None, fit_roles=(FIT,)) -> np.ndarray:
    """Fit rows get the fold model that did not train on their entity; every other row gets the mean.

    ``margin`` turns on LightGBM's prediction early stopping (Match.pred_margin).
    """
    kw = {} if margin is None else {"pred_early_stop": True, "pred_early_stop_freq": 10, "pred_early_stop_margin": margin}
    P = np.stack([m.predict(X, num_threads=0, **kw) for m in models]).astype(np.float32)
    out = P.mean(0)
    if role is not None:
        fit = np.flatnonzero(np.isin(role[q], fit_roles))
        out[fit] = P[fold[q[fit]], fit]
    return out


def _predict_parts(models, parts: list[Path], cols: list[str], role=None, fold=None, extra=None,
                   margin: float | None = None, fit_roles=(FIT,), cache: Path | None = None, tag: str = "") -> np.ndarray:
    """Predict every candidate in order; ``extra`` holds columns appended to the part features.

    With ``cache`` each part's predictions are checkpointed, so a restart resumes mid-way.
    """
    out, off = [], 0
    PROGRESS.begin(f"predict {tag}", total=len(parts), unit="parts")
    for i, p in enumerate(parts):
        f = None if cache is None else cache / f"{tag}-{i:04d}.npy"
        size = K._rows(p)
        if f is not None and K.good(f, min_rows=size):
            out.append(np.load(f))
        else:
            df = pl.read_parquet(p, columns=["q_rid"] + cols)
            X = df.select(cols).to_numpy()
            if extra is not None:
                X = np.hstack([X, extra[off:off + len(X)]])
            out.append(_predict(models, X, df["q_rid"].to_numpy(), role, fold, margin, fit_roles))
            if f is not None:
                K.write_npy(out[-1], f)
        off += size
        PROGRESS.step(rows=size)
    return np.concatenate(out)


def _cross_fit(X, y, q, w, role, fold, cols, mv: V.Match, tag: str, P: Paths, fit_roles) -> list:
    import lightgbm as lgb
    is_fit, is_es = np.isin(role[q], fit_roles), role[q] == ES
    X_es, y_es, w_es = X[is_es], y[is_es], w[is_es]
    models = []
    PROGRESS.begin(f"train {tag}", total=mv.folds, unit="folds")
    for f in range(mv.folds):
        path = P.run / f"{tag}_fold{f}.txt"
        if K.good(path):
            m = lgb.Booster(model_file=str(path))
            if m.feature_name() != cols:
                raise RuntimeError(f"{path.name}: cached model columns differ from this run's")
            models.append(m)
            log(f"{tag} fold {f}: resumed from {path.name}")
            PROGRESS.step(note=f"fold {f} resumed")
            continue
        tr = is_fit & (fold[q] != f)
        K.require(int(tr.sum()), f"{tag} fold {f} training rows", minimum=1000)
        t0 = time.time()
        m = M.train_arrays(X[tr], y[tr], X_es, y_es, cols, params=mv.lgb_params(), threads=0, w_tr=w[tr], w_va=w_es)
        K.save_model(m, path)
        models.append(m)
        log(f"{tag} fold {f}: {int(tr.sum()):,} rows, best iteration {m.best_iteration} ({time.time() - t0:.0f}s)")
        PROGRESS.step(rows=int(tr.sum()), note=f"fold {f} best iteration {m.best_iteration}")
    return models


def _importance(models, top: int = 15) -> list:
    gain = sum(np.asarray(m.feature_importance("gain")) for m in models)
    order = np.argsort(-gain)[:top]
    names = models[0].feature_name()
    return [(names[i], round(float(gain[i] / gain.sum()), 4)) for i in order]


def _houses(P: Paths, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    cols = ["rid", "h_num", "h_key"]
    return (pl.read_parquet(P.prep / f"Q_{split}.parquet", columns=cols),
            pl.read_parquet(P.prep / f"T_{split}.parquet", columns=cols))


def _stage2_extra(C: pl.DataFrame, p1: np.ndarray, T: pl.DataFrame | None, houses=None) -> tuple[np.ndarray, list[str]]:
    """Stage-2 inputs beyond the stage-1 features: p1 and its context, and optionally support / consensus features."""
    S = C.with_columns(pl.Series("p1", p1))
    blocks = [M.probability_context(S)]  # p1 + its context columns
    if T is not None:
        blocks.append(support_features(S, T))
    if houses is not None:
        blocks.append(consensus_features(S, *houses))
    frame = pl.concat(blocks, how="horizontal")
    return frame.to_numpy().astype(np.float32), frame.columns


def _columns(P: Paths, mv: V.Match, parts: list[Path]) -> list[str]:
    cols = [c for c in pl.read_parquet_schema(parts[0]) if c not in M.ID_COLS]
    if mv.cols == "v2":
        cols = [c for c in cols if c not in set(NUMGAP_FEATS) | set(HOUSE_FEATS)]
    return cols


def _train_selection(P: Paths, mv: V.Match, role: np.ndarray, pairs: pl.DataFrame, parts: list[Path], seed: int):
    """Per-candidate training flag and weight: roles, synthetic twins, easy-negative downsampling, twin weight."""
    C = pl.read_parquet(P.block / "cand_train.parquet", columns=["q_rid", "t_rid"])
    q, t = C["q_rid"].to_numpy(), C["t_rid"].to_numpy()
    fit_roles = [FIT] + ([REST] if mv.fit_rest else [])
    keep = np.isin(role[q], fit_roles + [ES])
    info = {"candidates": len(q), "rows_by_role": int(keep.sum())}
    synth = t >= n_real_targets(P)
    if not mv.train_synth:
        keep &= ~synth
    info["synthetic_rows_excluded" if not mv.train_synth else "synthetic_rows_included"] = int((synth & np.isin(role[q], fit_roles + [ES])).sum())
    w = np.ones(len(q), np.float32)
    y = _labels(C, pairs)
    if mv.easy_neg_keep < 1.0 and mv.easy_from:
        f = P.run_of(mv.easy_from) / "scores_train.parquet"
        if K.good(f, min_rows=len(q)):
            p1e = pl.read_parquet(f, columns=["p1"])["p1"].to_numpy()
            u = np.random.default_rng(seed + 7).random(len(q))
            easy = (y == 0) & (p1e < 0.01) & np.isin(role[q], fit_roles)
            drop = easy & (u >= mv.easy_neg_keep)
            keep &= ~drop
            w[easy & ~drop] = 1.0 / mv.easy_neg_keep
            info.update({"easy_negatives": int(easy.sum()), "easy_dropped": int(drop.sum())})
        else:
            info["easy_negatives"] = f"skipped: {f} not good"
            log(f"WARNING: easy-negative downsampling skipped, {f} is not a good artifact")
    if mv.easy_neg_keep < 1.0 and mv.easy_screen:
        p1e = _screen(P, parts, role, fit_roles, pairs, seed)
        u = np.random.default_rng(seed + 7).random(len(q))
        easy = (y == 0) & (p1e < 0.01) & np.isin(role[q], fit_roles)
        drop = easy & (u >= mv.easy_neg_keep)
        keep &= ~drop
        w[easy & ~drop] = 1.0 / mv.easy_neg_keep
        info.update({"easy_negatives": int(easy.sum()), "easy_dropped": int(drop.sum()), "easy_source": "screen"})
    if mv.twin_weight != 1.0:
        hn = np.concatenate([pl.read_parquet(p, columns=["h_near"])["h_near"].to_numpy() for p in parts])
        tw = (y == 0) & (hn == 1)
        w[tw] *= mv.twin_weight
        info["twin_negatives_weighted"] = int((tw & keep).sum())
    info["rows_kept"] = int(keep.sum())
    log(f"training selection: {info}")
    return keep, w, info, fit_roles


def _screen(P: Paths, parts: list[Path], role: np.ndarray, fit_roles, pairs: pl.DataFrame, seed: int) -> np.ndarray:
    """Quick screening model (3% of fit rows, 300 trees) whose probabilities mark easy negatives (p < 0.01).

    Cached: the model and its per-part predictions resume like everything else.
    """
    import lightgbm as lgb
    cols = [c for c in pl.read_parquet_schema(parts[0]) if c not in M.ID_COLS]
    path = P.run / "screen.txt"
    if K.good(path):
        m = lgb.Booster(model_file=str(path))
    else:
        rng = np.random.default_rng(seed + 11)
        sel = []
        for p in parts:
            q = pl.read_parquet(p, columns=["q_rid"])["q_rid"].to_numpy()
            sel.append(np.isin(role[q], fit_roles) & (rng.random(len(q)) < 0.03))
        keep = np.concatenate(sel)
        X, y, q, _ = _load_rows(parts, keep, cols, pairs)
        K.require(len(y), "screening rows", minimum=10000)
        params = {**M.DEFAULT_PARAMS, "learning_rate": 0.1, "num_leaves": 127, "num_threads": 0}
        m = lgb.train(params, lgb.Dataset(X, label=y, feature_name=cols), 300)
        K.save_model(m, path)
        log(f"screening model: {len(y):,} rows ({y.mean():.3f} positive), 300 trees")
        del X
    pc = P.run / "pred"
    pc.mkdir(parents=True, exist_ok=True)
    return _predict_parts([m], parts, cols, cache=pc, tag="screen")


def stage_train(a, rv: V.RunVersions, P: Paths) -> dict:
    mv = rv.match
    P.run.mkdir(parents=True, exist_ok=True)
    role = pl.read_parquet(P.prep / "split.parquet")["role"].to_numpy()
    fold = np.random.default_rng(a.seed + 1).integers(0, mv.folds, len(role)).astype(np.int8)
    pairs = pl.read_parquet(P.prep / "pairs.parquet")
    tr_parts, te_parts = _parts(P, "train"), _parts(P, "test")
    cols = _columns(P, mv, tr_parts)
    keep, w_all, sel, fit_roles = _train_selection(P, mv, role, pairs, tr_parts, a.seed)
    headroom(P.run, 2, int(keep.sum()) * (len(cols) + 40) * 4 * 2.2 / 2**30, "train")

    X, y, q, rows = _load_rows(tr_parts, keep, cols, pairs)
    w = w_all[rows]
    log(f"training rows: {len(y):,} ({y.mean():.4f} positive), {len(cols)} features")
    s1 = _cross_fit(X, y, q, w, role, fold, cols, mv, "stage1", P, fit_roles)
    del X  # stage 2 re-reads its rows, so the two matrices are never held together
    pc = P.run / "pred"
    pc.mkdir(exist_ok=True)
    p1_tr = _predict_parts(s1, tr_parts, cols, role, fold, margin=mv.pred_margin, fit_roles=fit_roles, cache=pc, tag="p1_train")
    p1_te = None if mv.eval_only else _predict_parts(s1, te_parts, cols, margin=mv.pred_margin, cache=pc, tag="p1_test")
    log("stage 1 predicted train (out-of-fold) and test")
    C_tr = pl.read_parquet(P.block / "cand_train.parquet", columns=["q_rid", "t_rid", "src"])
    C_te = pl.read_parquet(P.block / "cand_test.parquet", columns=["q_rid", "t_rid", "src"])
    K.require(len(p1_tr), "stage-1 train predictions", minimum=C_tr.height)
    scores_tr = C_tr.with_columns(pl.Series("p1", p1_tr))
    scores_te = None if mv.eval_only else C_te.with_columns(pl.Series("p1", p1_te))
    out = {"rows": len(y), "positive_rate": float(y.mean()), "features_stage1": len(cols), "selection": sel,
           "best_iterations_stage1": [m.best_iteration for m in s1], "importance_stage1": _importance(s1)}

    if mv.stages == 2:
        text = lambda split: (pl.read_parquet(P.prep / f"T_{split}.parquet", columns=["rid", "name_core", "name_nosp", "addr_n"])
                              if mv.support else None)
        houses = lambda split: _houses(P, split) if mv.consensus else None
        extra_tr, extra_cols = _stage2_extra(C_tr, p1_tr, text("train"), houses("train"))
        cols2 = cols + extra_cols
        X2, _, _, _ = _load_rows(tr_parts, keep, cols, pairs, extra=extra_tr)  # same rows, same order as X
        s2 = _cross_fit(X2, y, q, w, role, fold, cols2, mv, "stage2", P, fit_roles)
        del X2
        scores_tr = scores_tr.with_columns(pl.Series("p2", _predict_parts(s2, tr_parts, cols, role, fold, extra=extra_tr,
                                                                          margin=mv.pred_margin, fit_roles=fit_roles, cache=pc, tag="p2_train")))
        del extra_tr
        if not mv.eval_only:
            extra_te, _ = _stage2_extra(C_te, p1_te, text("test"), houses("test"))  # built only now, to keep the peak down
            scores_te = scores_te.with_columns(pl.Series("p2", _predict_parts(s2, te_parts, cols, extra=extra_te,
                                                                              margin=mv.pred_margin, cache=pc, tag="p2_test")))
            del extra_te
        log("stage 2 predicted train (out-of-fold) and test")
        out.update({"features_stage2": len(cols2), "best_iterations_stage2": [m.best_iteration for m in s2],
                    "importance_stage2": _importance(s2)})
    K.write_parquet(scores_tr, P.run / "scores_train.parquet")
    out["artifacts"] = ["scores_train.parquet"]
    if scores_te is not None:
        K.write_parquet(scores_te, P.run / "scores_test.parquet")
        out["artifacts"].append("scores_test.parquet")
    return out


# ------------------------------------------------------------------- predict
def _exclusive(scores: pl.DataFrame) -> pl.DataFrame:
    """Every S2/S3 record belongs to at most one S1 entity: keep it only where it scores highest."""
    return scores.filter(pl.col("p") == pl.col("p").max().over("t_rid"))


def _decide(ex: pl.DataFrame, rule: str, param: float) -> pl.DataFrame:
    """Links from exclusive scores: a threshold, expected F (param = floor), or gated expected F (param = gate)."""
    if rule == "threshold":
        return ex.filter(pl.col("p") >= param).select("q_rid", "t_rid")
    if rule == "gated_ef":
        return M.decide_gated(ex, gate=param, exclusive=False)
    return M.decide_expected_f(ex, floor=param, exclusive=False)


def _write_lists(Q: pl.DataFrame, T: pl.DataFrame, links: pl.DataFrame, col: str, path: Path) -> None:
    """One row per S1 record (file order), with a comma-separated list of S2/S3 ids ('' if none)."""
    ids = (links.select("q_rid", "t_rid")
           .join(T.select(pl.col("rid").alias("t_rid"), pl.col("entity_id").alias("tid")), on="t_rid")
           .sort("q_rid", "tid").group_by("q_rid", maintain_order=True)
           .agg(pl.col("tid").unique(maintain_order=True).str.join(",").alias(col)))
    df = (Q.select(pl.col("rid").alias("q_rid"), pl.col("entity_id").alias("source1_entity_id"))
          .join(ids, on="q_rid", how="left", maintain_order="left").with_columns(pl.col(col).fill_null(""))
          .select("source1_entity_id", col))
    if df.height != Q.height:
        raise RuntimeError(f"{path.name}: {df.height} rows for {Q.height} S1 records")
    tmp = path.with_name(path.name + ".tmp")
    df.write_csv(tmp, separator="\t", quote_style="never")
    os.replace(tmp, path)
    K.mark_existing(path)


def _weighted(per_country: dict, weights: dict) -> float:
    ws = {c: weights[c] for c in per_country if c in weights}
    tot = sum(ws.values())
    return sum(per_country[c]["f05"] * v for c, v in ws.items()) / tot if tot else float("nan")


def _eval_block(pred: pl.DataFrame, truth: pl.DataFrame, q_eval: np.ndarray, country: pl.DataFrame) -> dict:
    m = macro_f05(pred, truth, q_eval)
    per = {}
    for c in country["country"].unique().sort().to_list():
        qc = np.intersect1d(q_eval, country.filter(pl.col("country") == c)["rid"].to_numpy()).astype(np.uint32)
        if len(qc):
            per[c] = macro_f05(pred, truth, qc)
    train_w = {c: int(np.isin(q_eval, country.filter(pl.col("country") == c)["rid"].to_numpy()).sum()) for c in per}
    m["per_country"] = per
    m["f05_train_weighted"] = _weighted(per, train_w)
    m["f05_test_weighted"] = _weighted(per, TEST_WEIGHTS)
    return m


def stage_predict(a, rv: V.RunVersions, P: Paths) -> dict:
    PROGRESS.begin("predict", total=4, unit="steps")
    role = pl.read_parquet(P.prep / "split.parquet")["role"].to_numpy()
    pairs = pl.read_parquet(P.prep / "pairs.parquet")
    S = pl.read_parquet(P.run / "scores_train.parquet")
    nr = n_real_targets(P)
    score_cols = [c for c in ("p1", "p2") if c in S.columns]
    q_eval = np.flatnonzero(role == EVAL).astype(np.uint32)
    ids_eval = pl.DataFrame({"q_rid": q_eval})
    truth_eval = pairs.join(ids_eval, on="q_rid", how="semi")
    K.require(truth_eval.height, "eval truth links")
    country = pl.read_parquet(P.prep / "Q_train.parquet", columns=["rid", "country"])
    S_new = S.join(ids_eval, on="q_rid", how="semi")                  # test density (with synthetic twins)
    S_old = S_new.filter(pl.col("t_rid") < nr)                          # the same rows without the clones
    K.require(S_new.height, "eval candidate rows")
    oracle = macro_f05(S_new.select("q_rid", "t_rid").join(truth_eval, on=["q_rid", "t_rid"]), truth_eval, q_eval)
    log(f"eval: {len(q_eval):,} S1 entities; {S_new.height - S_old.height:,} synthetic-twin candidate rows; "
        f"perfect matcher F0.5={oracle['f05']:.4f}")

    table = []
    for score in score_cols:
        ex = _exclusive(S_new.select("q_rid", "t_rid", pl.col(score).alias("p")))
        for rule in rv.match.rules:
            for v in PARAMS[rule]:
                table.append({"score": score, "rule": rule, "param": float(v),
                              **macro_f05(_decide(ex, rule, float(v)), truth_eval, q_eval)})
    best = max(table, key=lambda r: r["f05"])  # chosen on the test-density (new) eval
    log(f"chosen on new eval: {best['score']} {best['rule']}={best['param']} F0.5={best['f05']:.4f}")
    PROGRESS.step(note="rule chosen")

    Hq, Ht = (_houses(P, "train") if "h_num" in pl.read_parquet_schema(P.prep / "Q_train.parquet") else (None, None))
    res = {}
    for tag, Sx in (("new_eval", S_new), ("old_eval", S_old)):
        ex = _exclusive(Sx.select("q_rid", "t_rid", pl.col(best["score"]).alias("p")))
        pred = _decide(ex, best["rule"], best["param"])
        res[tag] = _eval_block(pred, truth_eval, q_eval, country)
        if Hq is not None:  # the SUB-A rule on the same predictions
            fl = twin_links(pred, Hq, Ht, 2, 30)
            cq = pred.join(country.rename({"rid": "q_rid"}), on="q_rid", how="left")["country"]
            drop = fl & cq.is_in(list(TWIN_FILTER_COUNTRIES))
            pf = pred.filter(~drop)
            res[tag + "_twinfilter"] = _eval_block(pf, truth_eval, q_eval, country)
            res[tag + "_twinfilter"]["links_removed"] = int(drop.sum())
            res[tag + "_twinfilter"]["removed_true"] = int(pred.filter(drop).join(truth_eval, on=["q_rid", "t_rid"], how="semi").height)
        log(f"{tag}: F0.5={res[tag]['f05']:.4f} P={res[tag]['precision']:.4f} R={res[tag]['recall']:.4f} "
            f"train-w {res[tag]['f05_train_weighted']:.4f} test-w {res[tag]['f05_test_weighted']:.4f} | "
            + ", ".join(f"{c} {m['f05']:.4f}" for c, m in res[tag]["per_country"].items())
            + (f" | twin filter -> {res[tag + '_twinfilter']['f05']:.4f} (removed {res[tag + '_twinfilter']['links_removed']:,}, "
               f"{res[tag + '_twinfilter']['removed_true']:,} of them true)" if tag + "_twinfilter" in res else ""))
    best_old = max((dict(r, **macro_f05(_decide(_exclusive(S_old.select("q_rid", "t_rid", pl.col(r["score"]).alias("p"))),
                                                    r["rule"], r["param"]), truth_eval, q_eval))
                    for r in sorted(table, key=lambda r: -r["f05"])[:5]), key=lambda r: r["f05"])
    PROGRESS.step(note="eval scored")
    if rv.match.eval_only:
        return {"eval": {"oracle": oracle, "best": best, "best_on_old_eval": best_old, **res,
                         "table": sorted(table, key=lambda r: -r["f05"])[:25]}, "test": None, "artifacts": []}

    # test
    St = pl.read_parquet(P.run / "scores_test.parquet").select("q_rid", "t_rid", pl.col(best["score"]).alias("p"))
    links = _decide(_exclusive(St), best["rule"], best["param"])
    # full data: ~94% of train S1 have a match, so fewer links than half the S1 records means a broken run
    K.require(links.height, "test links", minimum=St["q_rid"].n_unique() // 2 if a.sample >= 1 else 1)
    Qt = pl.read_parquet(P.prep / "Q_test.parquet", columns=["rid", "entity_id", "country"])
    Tt = pl.read_parquet(P.prep / "T_test.parquet", columns=["rid", "entity_id"])
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_lists(Qt, Tt, links, "matched_entity_ids", out_dir / "matching_results.tsv")
    # candidates: the matcher's top CAND_TOP per S1 by probability, plus every predicted link (forced inclusion)
    top = St.with_columns(pl.col("p").rank("ordinal", descending=True).over("q_rid").alias("r")).filter(pl.col("r") <= CAND_TOP)
    forced = links.join(top.select("q_rid", "t_rid"), on=["q_rid", "t_rid"], how="anti")
    cand = pl.concat([top.select("q_rid", "t_rid"), forced])
    _write_lists(Qt, Tt, cand, "candidate_entity_ids", out_dir / "candidate_pairs.tsv")
    retained = St.join(links, on=["q_rid", "t_rid"], how="semi").height / max(links.height, 1)
    test = {"links": links.height, "s1_with_matches": links["q_rid"].n_unique(), "s1_total": Qt.height,
            "candidates_full": St.height, "candidates_written": cand.height, "cand_per_s1": cand.height / Qt.height,
            "forced_inclusions": forced.height, "links_in_scored_set": retained}
    per = (Qt.rename({"rid": "q_rid"}).join(links.group_by("q_rid").len(), on="q_rid", how="left").with_columns(pl.col("len").fill_null(0))
           .group_by("country").agg(pl.len().alias("s1"), (pl.col("len") > 0).mean().alias("share_linked"),
                                    pl.col("len").mean().alias("links_per_s1")).sort("country"))
    test["per_country"] = {r["country"]: {k: r[k] for k in ("s1", "share_linked", "links_per_s1")} for r in per.iter_rows(named=True)}
    if Hq is not None:
        Hqt, Htt = _houses(P, "test")
        fl = twin_links(links, Hqt, Htt, 2, 30)
        cq = links.join(Qt.select(pl.col("rid").alias("q_rid"), "country"), on="q_rid", how="left")["country"]
        test["near_number_links"] = {c: int((fl & (cq == c)).sum()) for c in sorted(cq.unique().to_list())}
        drop = fl & cq.is_in(list(TWIN_FILTER_COUNTRIES))
        _write_lists(Qt, Tt, links.filter(~drop), "matched_entity_ids", out_dir / "matching_results.twinfilter.tsv")
        test["twinfilter_removed"] = int(drop.sum())
    log(f"test: {links.height:,} links for {test['s1_with_matches']:,} of {Qt.height:,} S1; candidates "
        f"{cand.height:,} ({test['cand_per_s1']:.1f}/S1, {forced.height} forced) -> {out_dir}; profile {test['per_country']}")
    PROGRESS.step(note="test written")
    if a.validator and a.sample >= 1:
        r = subprocess.run([sys.executable, a.validator, "--matching", str(out_dir / "matching_results.tsv"),
                            "--candidate", str(out_dir / "candidate_pairs.tsv"), "--test-dir", str(Path(a.data) / "test")],
                           capture_output=True, text=True)
        test["validator"] = (r.stdout + r.stderr).strip()[-2000:]
        test["validator_exit"] = r.returncode
        log(f"validator exit {r.returncode}: {r.stdout.strip()[-300:]}")
    return {"eval": {"oracle": oracle, "best": best, "best_on_old_eval": best_old, **res,
                     "table": sorted(table, key=lambda r: -r["f05"])[:25]}, "test": test,
            "artifacts": []}


# ---------------------------------------------------------------------- main
FNS = {"prep": stage_prep, "block": stage_block, "features": stage_features, "train": stage_train,
       "predict": stage_predict}


def list_versions() -> None:
    for title, reg in (("NORM", V.NORM), ("BLK", V.BLOCK), ("FEAT", V.FEAT), ("MATCH", V.MATCH)):
        print(title)
        for vid, v in reg.items():
            print(f"  {vid:12s} {v.note}")
    print("PRESETS (end-to-end versions)")
    for name, ids in V.PRESETS.items():
        print(f"  {name:6s} {' + '.join(ids)}{'   (default)' if name == V.DEFAULT_PRESET else ''}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print every registered version and exit")
    ap.add_argument("--data", help="dataset dir containing train/ and test/")
    ap.add_argument("--work", help="dir for cached stage outputs (one per data scope)")
    ap.add_argument("--out", help="dir for matching_results.tsv and candidate_pairs.tsv")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--preset", default=V.DEFAULT_PRESET, choices=sorted(V.PRESETS))
    ap.add_argument("--norm", choices=sorted(V.NORM), help="override the preset's normalisation version")
    ap.add_argument("--block", choices=sorted(V.BLOCK), help="override the preset's blocking version")
    ap.add_argument("--feat", choices=sorted(V.FEAT), help="override the preset's feature version")
    ap.add_argument("--match", choices=sorted(V.MATCH), help="override the preset's matcher version")
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of the data, for smoke runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fit", type=float, default=0.30, help="share of train S1 entities used to fit models")
    ap.add_argument("--es", type=float, default=0.05, help="share used for early stopping")
    ap.add_argument("--eval", type=float, default=0.20, help="share held out to score and choose the decision rule")
    ap.add_argument("--part-rows", type=int, default=4_000_000)
    ap.add_argument("--feat-workers", type=int, default=4, help="processes for the feature stage")
    ap.add_argument("--force", action="store_true", help="recompute stages even if their artifacts are good")
    ap.add_argument("--validator", default=None, help="path to utils/validate_submission.py")
    a = ap.parse_args()
    if a.list:
        list_versions()
        return 0
    if not (a.data and a.work and a.out):
        ap.error("--data, --work and --out are required")

    rv = V.resolve(a.preset, a.norm, a.block, a.feat, a.match)
    P = Paths(Path(a.work), rv)
    PROGRESS.start(Path(a.work) / "PROGRESS.txt")
    log(f"versions: {rv.preset or 'custom'} = {rv.key('train').replace('__', ' + ')}")
    run_file = P.run / "metrics.json"
    for stage in a.stages.split(","):
        if stage != "prep":
            check_scope(a, P)
        if not a.force and stage != "predict" and stage_complete(P, stage):
            log(f"==== {stage}: complete and verified in {P.stage_dir(stage)}, skipped")
            continue
        t0 = time.time()
        log(f"==== {stage}  -> {P.stage_dir(stage)}")
        result = FNS[stage](a, rv, P)
        result.update({"seconds": round(time.time() - t0), "peak_gb": round(peak_gb(), 1)})
        write_json(P.stage_dir(stage) / f"{stage}.json", result)
        # the run's metrics.json gathers every stage it depends on, including cached ones
        write_json(run_file, {"versions": rv.describe(), "scope": data_scope(a),
                              **{s: read_json(P.stage_dir(s) / f"{s}.json") for s in STAGES
                                 if (P.stage_dir(s) / f"{s}.json").exists()}})
        log(f"==== {stage} done in {time.time() - t0:.0f}s")
    PROGRESS.begin("finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
