"""Read-only SUB-N3 validation audit; never changes frozen pipeline versions.

Reuses the saved stacker and CE scores. Reports legacy eval-only competition
and full-pool competition separately; missing fit-entity CE scores and in-sample
REST stacker predictions mean the latter is a diagnostic, not unbiased validation.
Low scores below .02 cannot enter the existing decision (floor .05), and are
filtered before joins to bound memory. CE context is computed on all CE rows.
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from ber.metrics import macro_f05
from ber.model import decide_gated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--prep', required=True)
    ap.add_argument('--ce-root', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    def log(s):
        print(f'[{time.time()-t0:.1f}s] {s}', flush=True)
    role = pl.read_parquet(Path(a.prep)/'split.parquet')
    truth = pl.read_parquet(Path(a.prep)/'pairs.parquet')
    country = pl.read_parquet(Path(a.prep)/'Q_train.parquet').rename({'rid':'q_rid'})
    ce = None
    for folder, col in [('ce2','ce'), ('ce_bge','ce_b'), ('ce_muril_in','ce_c')]:
        d = (pl.scan_parquet(str(Path(a.ce_root)/folder/'scores'/'part-*.parquet'))
             .filter(pl.col('split')=='train').select('q_rid','t_rid',pl.col('ce').alias(col))
             .collect(engine='streaming').unique(['q_rid','t_rid']))
        ce = d if ce is None else ce.join(d,on=['q_rid','t_rid'],how='left')
    ce = ce.with_columns(
        pl.col('ce').rank('ordinal',descending=True).over('q_rid').cast(pl.Float32).alias('ce_rank_q'),
        pl.len().over('q_rid').cast(pl.Float32).alias('band_n_q'),
        pl.when(pl.col('ce')==pl.col('ce').max().over('t_rid'))
        .then(pl.col('ce').sort(descending=True).slice(1,1).first().over('t_rid'))
        .otherwise(pl.col('ce').max().over('t_rid')).fill_null(0).alias('ce_other_t'))
    log(f'CE context {ce.height:,}')
    s = (pl.scan_parquet(Path(a.run)/'scores_train.parquet').filter(pl.col('p2')>=.02)
         .collect(engine='streaming').join(role,on='q_rid'))
    log(f'Retained decision-relevant rows {s.height:,}')
    b = s.filter(pl.col('p2')<=.98).join(ce,on=['q_rid','t_rid'],how='left')
    model = lgb.Booster(model_file=a.model)
    p = model.predict(b.select(model.feature_name()).to_numpy(),num_threads=6)
    updates = b.select('q_rid','t_rid').with_columns(pl.Series('p3',p))
    del b,ce
    s = s.join(updates,on=['q_rid','t_rid'],how='left').with_columns(pl.coalesce('p3','p2').alias('p3'))
    del updates
    s.write_parquet(out/'audit_scores.parquet')
    log('Stacker predictions saved')
    ids = role.filter(pl.col('role')==2)['q_rid'].to_numpy()
    report = {'caveats':['Full pool has missing FIT CE and in-sample REST stacker scores; diagnostic only.',
                         'Legacy scores tune the gate on eval; this is selection-biased.',
                         'p2 below .02 omitted: cannot enter existing .05 decision floor.'], 'results':{}}
    for col in ['p2','p3']:
        for pool in ['eval_only','all_entities']:
            x = s if pool=='all_entities' else s.filter(pl.col('role')==2)
            x = x.select('q_rid','t_rid',pl.col(col).alias('p'))
            ex = x.filter(pl.col('p')==pl.col('p').max().over('t_rid'))
            ties = ex.height-ex['t_rid'].n_unique()
            ex = ex.join(role.filter(pl.col('role')==2).select('q_rid'),on='q_rid',how='semi')
            for gate in [.45,.6,.75]:
                pred = decide_gated(ex,gate=gate,exclusive=False)
                r = macro_f05(pred,truth,ids)
                r['per_country']={c:macro_f05(pred,truth,np.intersect1d(ids,country.filter(pl.col('country')==c)['q_rid'].to_numpy()))['f05']
                                  for c in country['country'].unique().to_list()}
                r['ties_in_pool']=ties
                key=f'{col}/{pool}/gate{gate}'
                report['results'][key]=r
                log(f'{key}: {r}')
                if col=='p3' and gate==.6:
                    pred.write_parquet(out/f'pred_{pool}.parquet')
            (out/'audit.json').write_text(json.dumps(report,indent=2))
    log('Done')


if __name__=='__main__':
    main()
