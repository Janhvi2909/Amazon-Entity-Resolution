"""Export an isolated refinement and the exact original scored candidate set.

Candidate output is streamed by S1 chunks to bound string-join memory. It is
not pruned using model scores. All source rows, including empty lists, remain.
"""
import argparse,json,time
from pathlib import Path
import polars as pl
from ber.model import decide_gated

ap=argparse.ArgumentParser()
for k in ['run','prep','updates','out']: ap.add_argument('--'+k,required=True)
ap.add_argument('--gate',type=float,default=.45)
ap.add_argument('--candidate',action='store_true')
a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
t0=time.time()
def log(x): print(f'[{time.time()-t0:.0f}s] {x}',flush=True)
q=pl.read_parquet(Path(a.prep)/'Q_test.parquet',columns=['rid','entity_id','country']).rename({'rid':'q_rid','entity_id':'source1_entity_id'})
t=pl.read_parquet(Path(a.prep)/'T_test.parquet',columns=['rid','entity_id']).rename({'rid':'t_rid','entity_id':'tid'})
u=pl.read_parquet(a.updates)
s=(pl.scan_parquet(Path(a.run)/'scores_test.parquet').filter(pl.col('p2')>=.02).collect(engine='streaming')
   .join(u,on=['q_rid','t_rid'],how='left').with_columns(pl.coalesce('stack','p2').alias('p')))
ex=s.filter(pl.col('p')==pl.col('p').max().over('t_rid'))
links=decide_gated(ex.select('q_rid','t_rid','p'),gate=a.gate,exclusive=False)
assert links['t_rid'].n_unique()==links.height,'Decision assigned a target to multiple S1s'
links.write_parquet(out/'links.parquet')
def lists(rows,qq,col):
    ids=(rows.join(t,on='t_rid').sort('q_rid','tid').group_by('q_rid').agg(pl.col('tid').unique(maintain_order=True).str.join(',').alias(col)))
    return qq.join(ids,on='q_rid',how='left',maintain_order='left').select('source1_entity_id',pl.col(col).fill_null(''))
lists(links,q,'matched_entity_ids').write_csv(out/'matching_results.tsv',separator='\t',quote_style='never')
log(f'Matches {links.height:,}')
report={'links':links.height,'gate':a.gate,'country_links':dict(links.join(q.select('q_rid','country'),on='q_rid').group_by('country').len().iter_rows())}
del s,u,ex
if a.candidate:
    count=0
    with (out/'candidate_pairs.tsv').open('wb') as f:
        for start in range(0,q.height,100000):
            qq=q.slice(start,100000)
            rows=(pl.scan_parquet(Path(a.run)/'scores_test.parquet').select('q_rid','t_rid')
                  .filter(pl.col('q_rid').is_between(qq['q_rid'].min(),qq['q_rid'].max())).collect(engine='streaming'))
            count+=rows.height
            lists(rows,qq,'candidate_entity_ids').write_csv(f,separator='\t',quote_style='never',include_header=start==0)
            log(f'Candidates S1 {start:,}; cumulative {count:,}')
    report['candidate_pairs']=count
(out/'export.json').write_text(json.dumps(report,indent=2))
log(report)
