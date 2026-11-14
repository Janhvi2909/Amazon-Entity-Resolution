"""Cross-encoder reranker for M-v5's uncertain pairs (experimental branch exp-neural).

Fine-tunes microsoft/mdeberta-v3-base (MIT, 278M parameters) as a binary pair classifier on the
raw text "name ; address" of (S1, candidate) pairs whose M-v5 stage-2 probability lies in
[0.02, 0.98]. Trained on the fit entities' pairs, validated on the early-stop entities', then
scores the rest, eval and test pairs; a LightGBM stacker (stack.py) combines this score with
M-v5's. Only the provided training data is used; the model runs offline.

Checkpoints: the model is saved every --ckpt-min minutes (atomic rename) and training resumes
from it; inference writes parts with markers and skips good parts on restart.

    python ce.py --band band.parquet --out /n6/ce --model <mdeberta dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

FIT, ES, EVAL, REST = 0, 1, 2, 3


def text(n, a):
    return [f"{x or ''} ; {y or ''}" for x, y in zip(n, a)]


class Pairs(Dataset):
    def __init__(self, df: pl.DataFrame):
        self.a = text(df["qn"].to_list(), df["qa"].to_list())
        self.b = text(df["tn"].to_list(), df["ta"].to_list())
        self.y = df["y"].to_numpy().astype(np.float32) if "y" in df.columns else np.zeros(df.height, np.float32)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i], self.y[i]


class Progress:
    def __init__(self, path: Path):
        self.path, self.hist, self.t0 = path, [], time.time()

    def update(self, stage: str, done: int, total: int, extra: str = ""):
        now = time.time()
        self.hist = [(t, d) for t, d in self.hist if t > now - 600] + [(now, done)]
        (t_a, d_a), (t_b, d_b) = self.hist[0], self.hist[-1]
        rate = (d_b - d_a) / (t_b - t_a) if t_b > t_a else 0
        eta = f"{(total - done) / rate / 60:.1f} min (rate over last {(t_b - t_a) / 60:.1f} min)" if rate > 0 else "unknown"
        mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        self.path.write_text(f"updated {time.strftime('%H:%M:%S', time.gmtime())} UTC\nstage {stage}\nprogress {done:,} / {total:,}\n"
                             f"rate {rate:,.0f}/s\neta {eta}\ngpu peak {mem:.1f} GB\nelapsed {(now - self.t0) / 60:.1f} min\n{extra}\n")


def collate(tok, maxlen):
    def f(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), truncation=True, max_length=maxlen, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y)
        return enc
    return f


def train(a, B: pl.DataFrame, tok, prog: Progress):
    final = Path(a.out) / "model_final"
    if (final / "done.json").exists():
        print("training: final model exists, skipped", flush=True)
        return
    tr = B.filter((pl.col("split") == "train") & (pl.col("role") == FIT))
    if tr.height > a.max_train:
        tr = tr.sample(a.max_train, seed=42)
    va = B.filter((pl.col("split") == "train") & (pl.col("role") == ES))
    va = va.sample(min(40000, va.height), seed=1, shuffle=True)
    if tr.height == 0 or va.height == 0:
        raise RuntimeError("FATAL: empty training or validation band")
    print(f"training rows {tr.height:,} (positive {tr['y'].mean():.3f}); validation {va.height:,}", flush=True)
    dev = "cuda"
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = math.ceil(tr.height / a.bs) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    ck = Path(a.out) / "ckpt.pt"
    step = 0
    if ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"]); step = st["step"]
        print(f"resumed from step {step}", flush=True)
    g = torch.Generator().manual_seed(42)
    dl = DataLoader(Pairs(tr), batch_size=a.bs, shuffle=True, generator=g, num_workers=6, collate_fn=collate(tok, a.maxlen), drop_last=True)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    last_ck, ema = time.time(), None
    it = iter(dl)
    for _ in range(step % len(dl)):  # skip batches already seen (same shuffle order via the fixed generator)
        next(it)
    while step < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        y = batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**batch).logits.squeeze(-1)
        loss = lossf(logit.float(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        step += 1
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 50 == 0:
            prog.update("train", step * a.bs, steps * a.bs, f"loss(ema) {ema:.4f}")
        if step % 500 == 0:
            print(f"step {step}/{steps} loss {ema:.4f}", flush=True)
        if time.time() - last_ck > a.ckpt_min * 60:
            tmp = ck.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "step": step}, tmp)
            os.replace(tmp, ck); last_ck = time.time()
            print(f"checkpoint at step {step}", flush=True)
    model.eval()
    pv = predict(model, tok, va, a, prog, "validate")
    yv = va["y"].to_numpy()
    eps = 1e-6
    ll = float(-np.mean(yv * np.log(pv + eps) + (1 - yv) * np.log(1 - pv + eps)))
    p2v = va["p2"].to_numpy()
    ll2 = float(-np.mean(yv * np.log(p2v + eps) + (1 - yv) * np.log(1 - p2v + eps)))
    acc = float(((pv > 0.5) == (yv > 0.5)).mean()); acc2 = float(((p2v > 0.5) == (yv > 0.5)).mean())
    rep = {"train_rows": tr.height, "steps": steps, "val_rows": va.height, "val_logloss_ce": ll, "val_logloss_m5": ll2,
           "val_acc_ce": acc, "val_acc_m5": acc2}
    print(json.dumps(rep), flush=True)
    tmpd = final.with_name("model_final.tmp")
    model.save_pretrained(tmpd); tok.save_pretrained(tmpd)
    (tmpd / "done.json").write_text(json.dumps(rep))
    os.replace(tmpd, final)


@torch.no_grad()
def predict(model, tok, df: pl.DataFrame, a, prog: Progress | None, stage: str) -> np.ndarray:
    ds = Pairs(df)
    order = np.argsort([len(x) + len(y) for x, y in zip(ds.a, ds.b)])
    out = np.empty(len(ds), np.float32)
    col = collate(tok, a.maxlen)
    bs = a.bs * 6
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = col([ds[i] for i in idx]); enc.pop("labels")
        enc = {k: v.to("cuda", non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[idx] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
        if prog is not None and (s // bs) % 20 == 0:
            prog.update(stage, s, len(order))
    return out


def infer(a, B: pl.DataFrame, tok, prog: Progress):
    final = Path(a.model_dir) if a.model_dir else Path(a.out) / "model_final"
    model = AutoModelForSequenceClassification.from_pretrained(final).to("cuda").eval()
    X = B.filter((pl.col("split") == "test") | pl.col("role").is_in([ES, EVAL, REST]))
    if a.skip_scored:  # row ids are identical across NORM-v3t and NORM-v4t (same files, same order, same clones)
        import glob
        done = pl.concat([pl.read_parquet(f, columns=["split", "q_rid", "t_rid"]) for f in sorted(glob.glob(a.skip_scored))]).unique()
        n0 = X.height
        X = X.join(done, on=["split", "q_rid", "t_rid"], how="anti")
        print(f"skip-scored: {n0 - X.height:,} of {n0:,} pairs already scored, {X.height:,} to score", flush=True)
    d = Path(a.out) / "scores"; d.mkdir(parents=True, exist_ok=True)
    n = a.part_rows
    for i, s in enumerate(range(0, X.height, n)):
        f = d / f"part-{i:04d}.parquet"; m = d / f"part-{i:04d}.ok"
        chunk = X.slice(s, n)
        if f.exists() and m.exists() and json.loads(m.read_text())["rows"] == chunk.height:
            try:
                if pl.scan_parquet(f).select(pl.len()).collect().item() == chunk.height:
                    continue
            except Exception:
                pass
        p = predict(model, tok, chunk, a, prog, f"infer part {i} of {-(-X.height // n)}")
        res = chunk.select("split", "q_rid", "t_rid").with_columns(pl.Series("ce", p))
        tmp = f.with_suffix(".tmp"); res.write_parquet(tmp); os.replace(tmp, f)
        m.write_text(json.dumps({"rows": chunk.height}))
        print(f"inference part {i}: {chunk.height:,} rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stages", default="train,infer")
    ap.add_argument("--max-train", type=int, default=2_000_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--ckpt-min", type=float, default=20)
    ap.add_argument("--part-rows", type=int, default=500_000)
    ap.add_argument("--model-dir", default="", help="infer with this trained model_final dir instead of <out>/model_final")
    ap.add_argument("--skip-scored", default="", help="glob of score parts (split,q_rid,t_rid) already scored: those pairs are not re-scored")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    prog = Progress(Path(a.out) / "PROGRESS.txt")
    B = pl.read_parquet(a.band)
    print(f"band rows {B.height:,}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if "train" in a.stages:
        train(a, B, tok, prog)
    if "infer" in a.stages:
        infer(a, B, tok, prog)
    prog.update("finished", 1, 1)


if __name__ == "__main__":
    main()
