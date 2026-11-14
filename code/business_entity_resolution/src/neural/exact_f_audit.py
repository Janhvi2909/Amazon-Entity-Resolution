"""Compare exact independent-Bernoulli expected F0.5 with ratio approximation.

Gauss-Legendre quadrature integrates a polynomial exactly (up to floating-point
roundoff). For top k, E[F]=5 integral x^(4k) product_j(1-p_j+p_j*x)
sum_{i<=k} p_i/(1-p_i+p_i*x) dx. k=0 scores prod(1-p).
No labels enter the decision. This does not assume independence is true in data.
"""
import argparse,json
from pathlib import Path
import numpy as np
import polars as pl
from ber.metrics import macro_f05
from ber.model import decide_gated


def exact_decide(ex,gate=.6,floor=.05,batch_size=2000):
    s=ex.filter(pl.col('p')>=floor).sort(['q_rid','p'],descending=[False,True])
    grouped=s.group_by('q_rid',maintain_order=True).agg(pl.col('p'),pl.col('t_rid'))
    chosen=[]
    for start in range(0,grouped.height,batch_size):
        b=grouped.slice(start,batch_size)
        lists=b['p'].to_list(); n=max(map(len,lists)); count=np.array(list(map(len,lists)))
        p=np.zeros((len(lists),n))
        for i,values in enumerate(lists): p[i,:len(values)]=values
        # Polynomial degree <= 5*n-1, choose sufficient nodes for exactness.
        x,w=np.polynomial.legendre.leggauss((5*n+1)//2)
        x=(x+1)/2; w=w/2
        d=1-p[:,:,None]+p[:,:,None]*x
        product=d.prod(axis=1)
        cumulative=np.cumsum(p[:,:,None]/d,axis=1)
        k=np.arange(1,n+1)
        expected=5*np.sum(cumulative*product[:,None,:]*x[None,None,:]**(4*k[None,:,None])*w,axis=2)
        expected[k[None,:]>count[:,None]]=-1
        best=expected.argmax(axis=1)
        keep=(expected[np.arange(len(p)),best]>(1-p).prod(axis=1)) & (p[:,0]>=gate)
        qs=b['q_rid'].to_list(); tids=b['t_rid'].to_list()
        chosen.extend((qs[i],t) for i in range(len(p)) if keep[i] for t in tids[i][:best[i]+1])
    return pl.DataFrame(chosen,schema={'q_rid':pl.UInt32,'t_rid':pl.UInt32},orient='row')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scores',required=True); ap.add_argument('--prep',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); prep=Path(a.prep)
    s=pl.read_parquet(a.scores).filter(pl.col('role')==2).select('q_rid','t_rid',pl.col('p3').alias('p'))
    ex=s.filter(pl.col('p')==pl.col('p').max().over('t_rid'))
    truth=pl.read_parquet(prep/'pairs.parquet'); ids=pl.read_parquet(prep/'split.parquet').filter(pl.col('role')==2)['q_rid'].to_numpy()
    report={'approx':macro_f05(decide_gated(ex,.6,exclusive=False),truth,ids)}
    for floor in [.05,.02]:
        pred=exact_decide(ex,.6,floor)
        report[f'exact_floor_{floor}']=macro_f05(pred,truth,ids)
        print(report,flush=True)
    Path(a.out).write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
