import csv, sys, numpy as np
rows=[r for r in csv.DictReader(open(sys.argv[1])) if r['kind'] in('rollout','repeat') and r['K']=='6' and int(r['R'])>=8 and int(r['D'])>=4]
R=np.log2([float(r['R']) for r in rows]); D=np.log2([float(r['D']) for r in rows]); y=np.array([float(r['elo_vs_greedy_joint']) for r in rows])
def X(R,D): return np.stack([np.ones_like(R),R,D,R*R,D*D,R*D],1)
def fit(R,D,y): return np.linalg.lstsq(X(R,D),y,rcond=None)[0]
c=fit(R,D,y); res=y-X(R,D)@c
print("n",len(y),"coef",np.round(c,2),"resid sd",res.std().round(1))
def ridge(c,logB):
    # maximize over D with R=logB-D
    Ds=np.linspace(1,8,701); Rs=logB-Ds; v=X(Rs,Ds)@c; i=v.argmax(); return Ds[i],Rs[i],v[i]
Bs=np.log2([320,640,1280,2560,5120,10240])
for b in Bs: d,r,v=ridge(c,b); print(f"B=2^{b:.2f} D*={2**d:.1f} R*={2**r:.0f} Elo={v:.0f}")
# slope of D* vs logB, bootstrap
rng=np.random.default_rng(0); sl=[];gain=[]
for _ in range(2000):
    i=rng.integers(0,len(y),len(y)); cb=fit(R[i],D[i],y[i])
    ds=[ridge(cb,b)[0] for b in Bs]; vs=[ridge(cb,b)[2] for b in Bs]
    sl.append(np.polyfit(Bs,ds,1)[0]); gain.append(np.polyfit(Bs,vs,1)[0])
print("dlogD*/dlogB",np.percentile(sl,[5,50,95]).round(2)," Elo/doubling",np.percentile(gain,[5,50,95]).round(1))
