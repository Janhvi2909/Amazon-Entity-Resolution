"""Paired entity-level comparison and error decomposition on held-out labels."""
import argparse,json
from pathlib import Path
import numpy as np
import polars as pl
from ber.model import decide_gated

ap=argparse.ArgumentParser()
for k in ['scores','updates','prep','out']:ap.add_argument('--'+k,required=True)
a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
k=['q_rid','t_rid']
role=pl.read_parquet(Path(a.prep)/'split.parquet');ids=role.filter(pl.col('role')==2).select('q_rid')
truth=pl.read_parquet(Path(a.prep)/'pairs.parquet').join(ids,on='q_rid',how='semi')
country=pl.read_parquet(Path(a.prep)/'Q_train.parquet').rename({'rid':'q_rid'})
s=pl.read_parquet(a.scores).filter(pl.col('role')==2)
u=pl.read_parquet(a.updates).rename({'stack':'new'})
s=s.join(u,on=k,how='left').with_columns(pl.coalesce('new','p2').alias('new'))
stats=ids.join(truth.group_by('q_rid').len().rename({'len':'nt'}),on='q_rid',how='left').fill_null(0)
preds={}
for name,col,gate in [('old','p3',.6),('new','new',.45)]:
 ex=s.select(*k,pl.col(col).alias('p'));ex=ex.filter(pl.col('p')==pl.col('p').max().over('t_rid'))
 pred=decide_gated(ex,gate=gate,exclusive=False);preds[name]=pred
 n=pred.group_by('q_rid').len().rename({'len':f'np_{name}'})
 tp=pred.join(truth,on=k,how='semi').group_by('q_rid').len().rename({'len':f'tp_{name}'})
 stats=stats.join(n,on='q_rid',how='left').join(tp,on='q_rid',how='left').fill_null(0)
 stats=stats.with_columns(pl.when(pl.col('nt')==0).then((pl.col(f'np_{name}')==0).cast(pl.Float64))
    .otherwise(5*pl.col(f'tp_{name}')/(4*pl.col(f'np_{name}')+pl.col('nt'))).alias(f'f_{name}'))
stats=stats.join(country,on='q_rid').with_columns((pl.col('f_new')-pl.col('f_old')).alias('delta'))
report={}
for c in ['all']+sorted(stats['country'].unique().to_list()):
 x=stats if c=='all' else stats.filter(pl.col('country')==c)
 d=x['delta'].to_numpy();se=d.std(ddof=1)/np.sqrt(len(d))
 report[c]={'n':len(d),'old':x['f_old'].mean(),'new':x['f_new'].mean(),'delta':d.mean(),
            'approx_paired_95_ci':[d.mean()-1.96*se,d.mean()+1.96*se],
            'improved':int((d>0).sum()),'worsened':int((d<0).sum())}
miss=truth.join(preds['new'],on=k,how='anti');fp=preds['new'].join(truth,on=k,how='anti')
present=s.select(k)
report['error_links']={'missed':miss.height,'false_positive':fp.height,
  'missed_without_retained_p2_ge_002':miss.join(present,on=k,how='anti').height}
for name,x in [('misses',miss),('false_positives',fp)]:x.write_parquet(out/f'{name}.parquet')
stats.write_parquet(out/'entity_delta.parquet')
(out/'paired.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
