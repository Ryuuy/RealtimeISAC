import sys, numpy as np
sys.path.insert(0,"/home/usrpb210/sdr_workspace/RealtimeISAC/realtime")
from rt_config import RtConfig; from rt_dsp import DopplerEngine; from rt_sync import TddSync
r=np.load("/tmp/claude-1000/-home-usrpb210-sdr-workspace/d5275eac-6f68-4b1c-a405-2c9f99e6186d/scratchpad/rtout/doppler_20261025_165557.npz",allow_pickle=True)
print({k:(r[k].shape,r[k].dtype) for k in r.files})
iq=np.load("/home/usrpb210/sdr_workspace/RealtimeISAC/debug/iq_20261025_165202.npz")["iq"]
cfg=RtConfig(gain=20.0); ns=cfg.n_step; K=iq.shape[1]//2//ns
eng=DopplerEngine(cfg); sync=TddSync(cfg); S=[]
for k in range(K):
    raw=np.ascontiguousarray(iq[:,k*ns*2:(k+1)*ns*2]); sync.update(raw); eng.ingest(raw); eng.process(sync.phase,sync.n_integrate); S.append(eng.last_spectrum.copy())
S=np.array(S)
key=[k for k in r.files if r[k].ndim==2 and np.iscomplexobj(r[k])][0]; R=r[key]; print("rt_main spectra key",key,R.shape,"offline",S.shape)
n=min(len(R),len(S)); R=R[:n]; S=S[:n]
d=np.abs(R-S); print("max |rt_main - offline| over all frames/bins: %.3g   (max |S| = %.3g)"%(d.max(),np.abs(S).max()))
same=np.mean(np.all(np.isclose(R,S,rtol=1e-4,atol=1e-6),axis=1)); print("frames identical (rtol 1e-4): %.1f%%"%(same*100))
