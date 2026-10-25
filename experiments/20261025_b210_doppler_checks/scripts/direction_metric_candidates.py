import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/home/usrpb210/sdr_workspace/RealtimeISAC/realtime")
from rt_config import RtConfig; from rt_dsp import DopplerEngine; from rt_sync import TddSync
iq=np.load("/home/usrpb210/sdr_workspace/RealtimeISAC/debug/iq_20261025_160836.npz")["iq"]
cfg=RtConfig(gain=20.0); ns=cfg.n_step; K=iq.shape[1]//2//ns
eng=DopplerEngine(cfg); sync=TddSync(cfg); P=[]
for k in range(K):
    raw=np.ascontiguousarray(iq[:,k*ns*2:(k+1)*ns*2]); sync.update(raw); eng.ingest(raw); P.append(eng.process(sync.phase,sync.n_integrate).copy())
P=np.array(P); fr=eng.freqs; t=np.arange(K)*0.02
pb=(fr>=20)&(fr<=500); nb=(fr<=-20)&(fr>=-500); N=pb.sum()
st=(t>0.4)&(t<2.7)
fl=P[st][:,pb|nb].mean()            # mean noise power per bin (static)
print("noise floor per bin (mean |S|^2 in static, 20-500Hz):",fl)
def sm(x,n=5): return np.convolve(x,np.ones(n)/n,"same")
Pp=P[:,pb].sum(1);Pn=P[:,nb].sum(1); Sp=np.sqrt(P[:,pb]).sum(1);Sn=np.sqrt(P[:,nb]).sum(1)
M={}
M["R_abs (|S|)"]=(Sp-Sn)/(Sp+Sn)
M["R_pow (|S|^2)"]=(Pp-Pn)/(Pp+Pn)
M["R_shrunk: D/(P++P-+N*floor)"]=(Pp-Pn)/(Pp+Pn+N*fl*2)       # 2 bands
ex=lambda tau:(np.where(P[:,pb]>tau*fl,P[:,pb],0).sum(1),np.where(P[:,nb]>tau*fl,P[:,nb],0).sum(1))
a,b=ex(10); M["R_gate10dB (bins>10x floor)"]=np.divide(a-b,a+b,out=np.zeros_like(a),where=(a+b)>0)
D=(Pp-Pn)/(N*fl); M["D_dB = sign*10log10(1+|P+-P-|/(N*floor))"]=np.sign(D)*10*np.log10(1+np.abs(D))
a,b=ex(10); D2=(a-b)/(N*fl); M["D_gate_dB"]=np.sign(D2)*10*np.log10(1+np.abs(D2))
pw=np.where(P>10*fl,P,0)*(pb|nb); M["centroid (gated, Hz)"]=np.divide((pw*fr).sum(1),pw.sum(1),out=np.zeros(K),where=pw.sum(1)>0)
pre=[(6.5,7.9),(10.5,12.0),(16.2,17.7)]; post=[(8.4,9.9),(12.6,15.5),(18.6,19.0)]
sel=lambda w:np.concatenate([np.flatnonzero((t>=a)&(t<b)) for a,b in w])
ip,iq_=sel(pre),sel(post)
print(f"{'metric':46s} {'pre>0 %':>8s} {'post<0 %':>9s} {'static p95|m|':>14s} {'motion med|m|':>14s} {'sep(ratio)':>10s}")
for k,m in M.items():
    m=sm(m); s95=np.percentile(np.abs(m[st]),95); mm=np.median(np.abs(m[np.r_[ip,iq_]]))
    print(f"{k:46s} {np.mean(m[ip]>0)*100:8.0f} {np.mean(m[iq_]<0)*100:9.0f} {s95:14.3f} {mm:14.3f} {mm/s95:10.1f}")
fig,ax=plt.subplots(len(M),1,figsize=(12,2*len(M)),sharex=True)
for a_,(k,m) in zip(ax,M.items()):
    a_.plot(t,sm(m),lw=1);a_.axhline(0,c="k",lw=.4);a_.set_ylabel(k[:14],fontsize=7);a_.set_title(k,fontsize=8,loc="left")
    for x in (8.1,12.4,18.0): a_.axvline(x,c="r",lw=.5,ls="--")
plt.tight_layout();plt.savefig("/home/usrpb210/sdr_workspace/RealtimeISAC/debug/cand_160836.png",dpi=70)
