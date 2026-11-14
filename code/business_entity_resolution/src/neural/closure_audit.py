"""Evaluate conservative target-to-target retrieval from confident predictions.

No truth is used to generate candidates. This is a diagnostic, not an approved
postprocessing rule: exact signatures can still collide across businesses.
"""
import argparse,json,time
from pathlib import Path
import polars as pl
import lightgbm as lgb
from ber.model import decide_gated
from ber.metrics import macro_f05

ap=argparse.ArgumentParser()
for k in ['prep','run','refine','out']: ap.add_argument('--'+k,required=True)
a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
t0=time.time()
def log(s): print(f'[{time.time()-t0:.0f}s] {s}',flush=True)
K=['q_rid','t_rid']
role=pl.read_parquet(Path(a.prep)/'split.parquet')
truth=pl.read_parquet(Path(a.prep)/'pairs.parquet')
ids=role.filter(pl.col('role')==2)['q_rid'].to_numpy()
b=pl.read_parquet(Path(a.refine)/'band_train.parquet')
m=lgb.Booster(model_file=str(Path(a.refine)/'all_features.txt'))
u=b.select(K).with_columns(pl.Series('stack',m.predict(b.select(m.feature_name()).to_numpy(),num_threads=16)))
del b
s=(pl.scan_parquet(Path(a.run)/'scores_train.parquet').filter(pl.col('p2')>=.02).collect(engine='streaming')
   .join(u,on=K,how='left').with_columns(pl.coalesce('stack','p2').alias('p')))
ex=s.filter(pl.col('p')==pl.col('p').max().over('t_rid'))
pred=decide_gated(ex.select(*K,'p'),gate=.45,exclusive=False)
anchors=ex.filter(pl.col('p')>=.995).select(K).join(pred,on=K,how='semi')
del s,ex,u
nr=json.loads((Path(a.prep)/'prep.json').read_text())['train_targets_real']
cols=['rid','country','name_core','name_nosp','addr_n','has_addr','legal']
t=pl.read_parquet(Path(a.prep)/'T_train.parquet',columns=cols).filter(pl.col('rid')<nr).rename({'rid':'t_rid'})
at=anchors.join(t,on='t_rid'); unmatched=t.join(pred.select('t_rid'),on='t_rid',how='anti')
report={'baseline':macro_f05(pred,truth,ids),'variants':{}}
log(f'baseline {report["baseline"]}; anchors {anchors.height:,}; unassigned {unmatched.height:,}')
for name,keys,hasaddr in [('full_name_address',['country','name_nosp','addr_n'],True),
                         ('core_name_address',['country','name_core','addr_n'],True),
                         ('unique_alias_noaddr',['country','name_nosp'],False)]:
    idx=(at.filter(pl.col('has_addr')).group_by(keys).agg(pl.col('q_rid').n_unique().alias('owners'),pl.col('q_rid').first())
         .filter(pl.col('owners')==1).drop('owners'))
    add=unmatched.filter(pl.col('has_addr')==hasaddr).join(idx,on=keys).select(K).unique()
    # Exclude conflicting proposals; no arbitrary ID tie-breaking.
    add=add.filter(pl.len().over('t_rid')==1)
    ae=add.join(role.filter(pl.col('role')==2).select('q_rid'),on='q_rid',how='semi')
    ntrue=ae.join(truth,on=K,how='semi').height
    metrics=macro_f05(pl.concat([pred,add]).unique(),truth,ids)
    report['variants'][name]={'added_eval':ae.height,'added_true':ntrue,'added_precision':ntrue/max(ae.height,1),'metrics':metrics}
    add.write_parquet(out/f'{name}.parquet')
    log(f'{name}: {report["variants"][name]}')
    (out/'closure.json').write_text(json.dumps(report,indent=2))
log('Done')
