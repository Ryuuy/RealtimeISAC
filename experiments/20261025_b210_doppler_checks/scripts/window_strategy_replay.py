import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/home/usrpb210/sdr_workspace/RealtimeISAC/realtime")
from rt_config import RtConfig; from rt_dsp import DopplerEngine; from rt_sync import TddSync
d=np.load("/home/usrpb210/sdr_workspace/RealtimeISAC/debug/iq_20261025_155600.npz"); iq=d["iq"]
cfg=RtConfig(); ns=cfg.n_step; K=iq.shape[1]//2//ns
def run(mode):
    eng=DopplerEngine(cfg); sync=TddSync(cfg); out=[]; ph=[]; ni=[]
    for k in range(K):
        raw=np.ascontiguousarray(iq[:,k*ns*2:(k+1)*ns*2])
        if mode=="live": sync.update(raw)
        elif mode=="frozen": sync.update(raw, force=(k==0)) if k==0 else None
        elif mode=="fixed_end": sync.phase,sync.locked,sync._n_on=2150,True,6830
        elif mode=="narrow":   # live phase, but only central 300us
            sync.update(raw)
        eng.ingest(raw)
        if mode=="full": p=eng.process(0,cfg.n_period)
        elif mode=="narrow":
            p=eng.process(sync.phase+(sync.n_integrate-3000)//2 if sync.locked else 0, 3000 if sync.locked else cfg.n_period)
        else: p=eng.process(sync.phase,sync.n_integrate)
        out.append(p.copy()); ph.append(sync.phase); ni.append(sync.n_integrate)
    return np.array(out),np.array(ph),np.array(ni),eng
res={m:run(m) for m in ["live","frozen","fixed_end","full","narrow"]}
eng=res["live"][3]; mask=eng.dc_mask; fr=eng.freqs
t=np.arange(K)*0.02
st=(t>0.4)&(t<4.4); mo=(t>5)&(t<12.5)
ref=np.median(res["live"][0][st][:,mask])   # common static floor
print(f"{'mode':10s} {'floor dB':>9s} {'motion mean dB':>15s} {'motion-floor':>13s} {'static p99-floor':>17s} {'corr w/ live':>13s}")
L=10*np.log10(res["live"][0]+1e-20)
for m,(P,ph,ni,_) in res.items():
    D=10*np.log10(P+1e-20)
    fl=np.median(D[st][:,mask]); mm=10*np.log10(P[mo][:,mask].mean()); sp=np.percentile(D[st][:,mask],99)
    c=np.corrcoef(L[mo][:,mask].ravel(),D[mo][:,mask].ravel())[0,1]
    print(f"{m:10s} {fl:9.1f} {mm:15.1f} {mm-fl:13.1f} {sp-fl:17.1f} {c:13.3f}")
P,ph,ni,_=res["live"]; ch=np.flatnonzero(np.diff(ph)!=0)
print("live: phase %d->%d, n_int %d..%d, #window moves %d, locked_frac=%.3f"%(ph[0],ph[-1],ni.min(),ni.max(),len(ch),(ni<cfg.n_period).mean()))
fig,ax=plt.subplots(5,1,figsize=(12,13),sharex=True)
for a,m in zip(ax[:4],["live","frozen","full","narrow"]):
    D=10*np.log10(res[m][0]+1e-20); D=D-L.max()
    a.pcolormesh(t,fr,D.T,cmap="jet",vmin=-70,vmax=0,shading="auto"); a.set_ylabel("Hz"); a.set_title(m)
ax[4].plot(t,ph/10,label="live phase (us)"); ax[4].set_ylabel("us"); ax[4].set_xlabel("s"); ax[4].legend()
plt.tight_layout(); plt.savefig("/home/usrpb210/sdr_workspace/RealtimeISAC/debug/winreplay_155600.png",dpi=80)
