"""Create explicit, reproducible CE band subsets without overwriting artifacts.

review: all train ES/EVAL, up to --rest-rows train REST sampled with --seed,
and all test rows in --review-country. Input must already be the high-score band.
country: keep only --country, for example India-only MuRIL training/inference.
This implements the documented selection; it does not certify historical row order.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import polars as pl


def select(df, mode, country='India', review_country='France', rest_rows=1000000, seed=7):
    keys = ['split', 'q_rid', 't_rid']
    if df.select(keys).unique().height != df.height:
        raise ValueError('Input contains duplicate pair keys')
    if mode == 'country':
        result = df.filter(pl.col('country') == country)
    else:
        if not df['p2'].is_between(.98, 1.).fill_null(False).all():
            raise ValueError('Review input must contain only p2 in [.98, 1]')
        rest = df.filter((pl.col('split') == 'train') & (pl.col('role') == 3)).sort(keys)
        rest = rest.sample(n=min(rest_rows, rest.height), seed=seed)
        held = df.filter((pl.col('split') == 'train') & pl.col('role').is_in([1, 2]))
        test = df.filter((pl.col('split') == 'test') & (pl.col('country') == review_country))
        result = pl.concat([held, rest, test])
    return result.sort(keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=['country', 'review'], required=True)
    ap.add_argument('--country', default='India')
    ap.add_argument('--review-country', default='France')
    ap.add_argument('--rest-rows', type=int, default=1000000)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    if a.rest_rows < 0:
        ap.error('--rest-rows must be nonnegative')
    dest = Path(a.out)
    manifest = dest.with_suffix(dest.suffix + '.manifest.json')
    if dest.exists() or manifest.exists():
        ap.error('Refusing to overwrite an existing band or manifest; use a new output path')
    df = pl.read_parquet(a.input)
    result = select(df, a.mode, a.country, a.review_country, a.rest_rows, a.seed)
    if not result.height:
        ap.error('Selection is empty')
    digest = hashlib.sha256()
    for split, q, t in result.select('split', 'q_rid', 't_rid').iter_rows():
        digest.update(f'{split}\t{q}\t{t}\n'.encode())
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.tmp')
    result.write_parquet(tmp)
    os.replace(tmp, dest)
    info = {'arguments': vars(a), 'input_rows': df.height, 'selected_rows': result.height,
            'ordered_pair_keys_sha256': digest.hexdigest(), 'polars_version': pl.__version__,
            'groups': result.group_by('split', 'role', 'country').len().sort('split', 'role', 'country').to_dicts(),
            'historical_equivalence': 'Not asserted; verify historical manifest before reusing CE parts.'}
    manifest.write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
