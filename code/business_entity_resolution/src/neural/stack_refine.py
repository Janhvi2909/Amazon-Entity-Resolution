"""Isolated bounded-memory stacker experiments over frozen M-v5 caches.

Train on REST, early-stop on ES, report on EVAL; gate selected on ES.
Writes score updates, never mutates existing versions or submissions.
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

KEY=['q_rid','t_rid']
BASE=['p1','p2','ce','ce_rank_q','band_n_q','ce_other_t','ce_b','ce_c']
NUM=['h_gap','h_loggap','h_edit','h_diffpos','h_near','h_same_key','h_suf_conflict',
     'legal_equal','legal_conflict','num_x_edit','num_x_loggap','num_jacc','a_tset']
CONTEXT=['base_other_t','base_margin_t','base_sum_q','base_count_q']


def context(b):
    return b.with_columns(
        pl.col('ce').rank('ordinal',descending=True).over('q_rid').cast(pl.Float32).alias('ce_rank_q'),
        pl.len().over('q_rid').cast(pl.Float32).alias('band_n_q'),
        pl.when(pl.col('ce')==pl.col('ce').max().over('t_rid'))
        .then(pl.col('ce').sort(descending=True).slice(1,1).first().over('t_rid'))
        .otherwise(pl.col('ce').max().over('t_rid')).fill_null(0).alias('ce_other_t'))


def main():
    ap=argparse.ArgumentParser()
    for k in ['run','prep','feat','ce_root','out']:
        ap.add_argument('--'+k.replace('_','-'),required=True)
    ap.add_argument('--threads',type=int,default=16)
    ap.add_argument('--test',action='store_true')
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    def log(x): print(f'[{time.time()-t0:.0f}s] {x}',flush=True)
    role=pl.read_parquet(Path(a.prep)/'split.parquet')
    truth=pl.read_parquet(Path(a.prep)/'pairs.parquet')
    countries=pl.read_parquet(Path(a.prep)/'Q_train.parquet',columns=['rid','country']).rename({'rid':'q_rid'})

    def prepare(split):
        cache=out/f'band_{split}.parquet'
        scores=pl.scan_parquet(Path(a.run)/f'scores_{split}.parquet').filter(pl.col('p2')>=.02).collect(engine='streaming')
        if cache.exists(): return scores,pl.read_parquet(cache)
        ce=None
        for d,c in [('ce2','ce'),('ce_bge','ce_b'),('ce_muril_in','ce_c')]:
            z=(pl.scan_parquet(str(Path(a.ce_root)/d/'scores'/'part-*.parquet')).filter(pl.col('split')==split)
               .select(*KEY,pl.col('ce').alias(c)).collect(engine='streaming').unique(KEY))
            ce=z if ce is None else ce.join(z,on=KEY,how='left')
        ce=context(ce)
        # Context over all decision-relevant candidates, including confident rivals.
        ctx=scores.with_columns(
            pl.when(pl.col('p2')==pl.col('p2').max().over('t_rid'))
              .then(pl.col('p2').sort(descending=True).slice(1,1).first().over('t_rid'))
              .otherwise(pl.col('p2').max().over('t_rid')).fill_null(0).alias('base_other_t'),
            pl.col('p2').sum().over('q_rid').alias('base_sum_q'),
            (pl.col('p2')>.5).sum().over('q_rid').alias('base_count_q'))
        b=ctx.filter(pl.col('p2')<=.98).with_columns((pl.col('p2')-pl.col('base_other_t')).alias('base_margin_t'))
        del ctx
        b=b.join(ce,on=KEY,how='left'); del ce
        keys=b.select(KEY); chunks=[]
        for i,f in enumerate(sorted(Path(a.feat,split).glob('part-*.parquet'))):
            chunks.append(pl.read_parquet(f).join(keys,on=KEY,how='semi'))
            if i%20==0: log(f'{split} feature part {i}')
        feats=pl.concat(chunks); del chunks
        b=b.join(feats,on=KEY,how='left'); del feats
        if split=='train':
            b=b.join(role,on='q_rid').join(truth.with_columns(pl.lit(1).alias('y')),on=KEY,how='left').with_columns(pl.col('y').fill_null(0))
        b.write_parquet(cache); log(f'{split} compact band {b.height:,}')
        return scores,b

    scores,b=prepare('train')
    allfeat=[c for c in b.columns if c not in KEY+['role','y','src']]
    configs={'numeric':BASE+[c for c in NUM if c in b.columns],
             'context_numeric':BASE+CONTEXT+[c for c in NUM if c in b.columns],
             'all_features':allfeat}
    params=dict(objective='binary',learning_rate=.03,num_leaves=63,min_data_in_leaf=200,
                feature_fraction=.9,bagging_fraction=.8,bagging_freq=1,lambda_l2=1.,verbosity=-1,seed=42,num_threads=a.threads)
    report={}
    models={}
    for name,cols in configs.items():
        tr=b.filter(pl.col('role')==3); es=b.filter(pl.col('role')==1)
        m=lgb.train(params,lgb.Dataset(tr.select(cols).to_numpy(),label=tr['y'].to_numpy(),feature_name=cols),2000,
                    valid_sets=[lgb.Dataset(es.select(cols).to_numpy(),label=es['y'].to_numpy())],
                    callbacks=[lgb.early_stopping(80,verbose=False)])
        m.save_model(str(out/f'{name}.txt')); models[name]=m
        del tr,es
        log(f'{name}: trained {m.best_iteration} trees')
        r={'iteration':m.best_iteration,'feature_gain':dict(zip(cols,m.feature_importance('gain').tolist()))}
        # Legacy comparable validation; explicitly retain this label in the report.
        for rid,tag in [(1,'es'),(2,'eval')]:
            ids=role.filter(pl.col('role')==rid).select('q_rid')
            sb=b.filter(pl.col('role')==rid)
            upd=sb.select(KEY).with_columns(pl.Series('stack',m.predict(sb.select(cols).to_numpy(),num_threads=a.threads)))
            ev=(scores.join(ids,on='q_rid',how='semi').join(upd,on=KEY,how='left')
                .with_columns(pl.coalesce('stack','p2').alias('p')))
            ev=ev.filter(pl.col('p')==pl.col('p').max().over('t_rid')).select(*KEY,'p')
            gates=[.45,.6,.75] if tag=='es' else [r['gate']]
            metrics={}
            for gate in gates:
                pred=decide_gated(ev,gate=gate,exclusive=False)
                z=macro_f05(pred,truth,ids['q_rid'].to_numpy())
                z['per_country']={c:macro_f05(pred,truth,ids.join(countries.filter(pl.col('country')==c),on='q_rid')['q_rid'].to_numpy())['f05'] for c in countries['country'].unique()}
                metrics[gate]=z
            if tag=='es': r['gate']=max(metrics,key=lambda g:metrics[g]['f05'])
            r[tag]=metrics; log(f'{name} {tag}: {metrics}')
            if tag=='eval': upd.write_parquet(out/f'{name}_eval_updates.parquet')
        report[name]=r
        (out/'experiments.json').write_text(json.dumps(report,indent=2))
    if a.test:
        del b,scores
        _,bt=prepare('test')
        for name,m in models.items():
            p=m.predict(bt.select(m.feature_name()).to_numpy(),num_threads=a.threads)
            bt.select(KEY).with_columns(pl.Series('stack',p)).write_parquet(out/f'{name}_test_updates.parquet')
        log('Test score updates complete')
    log('Done; eval uses legacy within-split competition for comparability')


if __name__=='__main__': main()
