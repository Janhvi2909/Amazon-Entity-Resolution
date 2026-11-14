"""Ablate macro-entity weighting and stacker training on mean-fold ES scores.

Entity weights use training truth only. Gates are selected on early-stop entities.
The ES-only variant splits ES deterministically into fit and stopping halves;
it never trains on evaluation entities.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb
from ber.metrics import macro_f05
from ber.model import decide_gated

ap=argparse.ArgumentParser()
for x in ['band','scores','prep','out']:ap.add_argument('--'+x,required=True)
a=ap.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True);t0=time.time()
def log(x):print(f'[{time.time()-t0:.0f}s] {x}',flush=True)
b=pl.read_parquet(a.band)
role=pl.read_parquet(Path(a.prep)/'split.parquet'); truth=pl.read_parquet(Path(a.prep)/'pairs.parquet')
count=truth.group_by('q_rid').len().rename({'len':'ntrue'})
b=b.join(count,on='q_rid',how='left').with_columns(pl.col('ntrue').fill_null(0))
cols=[c for c in b.columns if c not in ['q_rid','t_rid','src','role','y','ntrue']]
scores=pl.read_parquet(a.scores).select('q_rid','t_rid','p2','role')
report={}
for name,exp,esfit in [('sqrt_entity',.5,False),('inverse_entity',1.,False),('mean_fold_es',0.,True)]:
 tr=b.filter((pl.col('role')==1)&(pl.col('q_rid')%2==0)) if esfit else b.filter(pl.col('role')==3)
 es=b.filter((pl.col('role')==1)&(pl.col('q_rid')%2==1)) if esfit else b.filter(pl.col('role')==1)
 w=1/np.maximum(tr['ntrue'].to_numpy(),1)**exp; w=w/w.mean()
 params=dict(objective='binary',learning_rate=.03,num_leaves=31 if esfit else 63,min_data_in_leaf=100 if esfit else 200,
             feature_fraction=.9,bagging_fraction=.8,bagging_freq=1,lambda_l2=1.,verbosity=-1,seed=42,num_threads=6)
 m=lgb.train(params,lgb.Dataset(tr.select(cols).to_numpy(),label=tr['y'].to_numpy(),weight=w,feature_name=cols),1500,
             valid_sets=[lgb.Dataset(es.select(cols).to_numpy(),label=es['y'].to_numpy())],callbacks=[lgb.early_stopping(80,verbose=False)])
 m.save_model(str(out/f'{name}.txt'));log(f'{name} trained {m.best_iteration}')
 r={}
 for rid,tag in [(1,'es'),(2,'eval')]:
  ids=role.filter(pl.col('role')==rid)
  if esfit and rid==1:ids=ids.filter(pl.col('q_rid')%2==1)
  sb=b.join(ids.select('q_rid'),on='q_rid',how='semi')
  u=sb.select('q_rid','t_rid').with_columns(pl.Series('new',m.predict(sb.select(cols).to_numpy(),num_threads=6)))
  s=scores.join(ids.select('q_rid'),on='q_rid',how='semi').join(u,on=['q_rid','t_rid'],how='left').with_columns(pl.coalesce('new','p2').alias('p'))
  ex=s.filter(pl.col('p')==pl.col('p').max().over('t_rid')).select('q_rid','t_rid','p')
  grid=[.3,.45,.6,.75,.9] if tag=='es' else [r['gate']]
  rr={g:macro_f05(decide_gated(ex,g,exclusive=False),truth,ids['q_rid'].to_numpy()) for g in grid}
  if tag=='es':r['gate']=max(rr,key=lambda g:rr[g]['f05'])
  r[tag]=rr;log(f'{name} {tag} {rr}')
  if tag=='eval':u.write_parquet(out/f'{name}_updates.parquet')
 report[name]=r;(out/'results.json').write_text(json.dumps(report,indent=2))
