"""Streaming test prediction differences with raw examples; no test labels inferred."""
import argparse,csv,json
from pathlib import Path
import polars as pl
from rapidfuzz.fuzz import token_set_ratio
from anyascii import anyascii

ap=argparse.ArgumentParser()
for x in ['base','new','data','out']:ap.add_argument('--'+x,required=True)
a=ap.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
changes=[]; counts={}; affected={}
with open(a.base) as bf,open(a.new) as nf,open(Path(a.data)/'test/test_source1.tsv') as qf:
 b=csv.reader(bf,delimiter='\t',quoting=csv.QUOTE_NONE);n=csv.reader(nf,delimiter='\t',quoting=csv.QUOTE_NONE);q=csv.DictReader(qf,delimiter='\t',quoting=csv.QUOTE_NONE)
 next(b);next(n)
 for br,nr,qr in zip(b,n,q,strict=True):
  assert br[0]==nr[0]==qr['entity_id']
  old=set(filter(None,br[1].split(',')));new=set(filter(None,nr[1].split(',')));c=qr['country']
  if old!=new:affected[c]=affected.get(c,0)+1
  for kind,ts in [('added',new-old),('removed',old-new)]:
   counts.setdefault(c,{}).setdefault(kind,0);counts[c][kind]+=len(ts)
   changes.extend((br[0],t,c,kind,qr['business_name'],qr['business_address']) for t in sorted(ts))
d=pl.DataFrame(changes,schema=['s1','tid','country','kind','qn','qa'],orient='row')
ids=d['tid'].unique().implode()
targets=pl.concat([pl.scan_csv(Path(a.data)/f'test/test_source{k}.tsv',separator='\t',quote_char=None,infer_schema=False)
                  .filter(pl.col('entity_id').is_in(ids)).select(pl.col('entity_id').alias('tid'),pl.col('business_name').alias('tn'),pl.col('business_address').fill_null('').alias('ta'))
                  .collect(engine='streaming') for k in [2,3]])
d=d.join(targets,on='tid')
d=d.with_columns(pl.Series('name_tset',[token_set_ratio(anyascii(x).lower(),anyascii(y).lower()) for x,y in zip(d['qn'],d['tn'])]),
                 pl.Series('address_tset',[token_set_ratio(anyascii(x).lower(),anyascii(y).lower()) for x,y in zip(d['qa'],d['ta'])]))
d.write_parquet(out/'changed_pairs.parquet')
summary=d.group_by('country','kind').agg(pl.len().alias('links'),(pl.col('ta')=='').sum().alias('no_address'),pl.col('name_tset').mean(),pl.col('address_tset').mean()).sort('country','kind')
report={'note':'Label-free profile, not an accuracy estimate.','counts':counts,'affected_s1':affected,'summary':summary.to_dicts()}
(out/'profile.json').write_text(json.dumps(report,indent=2))
for c in d['country'].unique():
 d.filter(pl.col('country')==c).sample(n=min(100,d.filter(pl.col('country')==c).height),seed=42).write_csv(out/f'{c}_sample.tsv',separator='\t')
print(json.dumps(report,indent=2))
