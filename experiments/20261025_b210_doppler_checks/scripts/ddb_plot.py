import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/home/usrpb210/sdr_workspace/RealtimeISAC/realtime")
from rt_config import RtConfig; from rt_dsp import DopplerEngine; from rt_sync import TddSync
iq=np.load("/home/usrpb210/sdr_workspace/RealtimeISAC/debug/iq_20261025_163944.npz")["iq"]
cfg=RtConfig(gain=20.0); ns=cfg.n_step; K=iq.shape[1]//2//ns
eng=DopplerEngine(cfg); sync=TddSync(cfg); P=[]
for k in range(K):
    raw=np.ascontiguousarray(iq[:,k*ns*2:(k+1)*ns*2]); sync.update(raw); eng.ingest(raw); P.append(eng.process(sync.phase,sync.n_integrate).copy())
P=np.array(P); fr=eng.freqs; t=np.arange(K)*0.02
pb=(fr>=20)&(fr<=500); nb=(fr<=-20)&(fr>=-500); N=pb.sum()
st=(t>0.4)&(t<2.7); fl=P[st][:,pb|nb].mean()
Pp,Pn=P[:,pb].sum(1),P[:,nb].sum(1); D=Pp-Pn
ref=N*fl                                   # noise-only sum of one band
Ddb=np.sign(D)*10*np.log10(np.abs(D)/ref+1e-12)   # signed dB re noise-band sum
Ddb_c=np.where(np.abs(D)>ref,Ddb,0)               # <0 dB (below noise) -> 0
def sm(x,n=5): return np.convolve(x,np.ones(n)/n,"same")
mag=10*np.log10(np.abs(D)/ref+1e-12)
Ddb=np.sign(D)*mag
fig,ax=plt.subplots(2,1,figsize=(12,7),sharex=True,constrained_layout=True)
Dd=10*np.log10(P+1e-20);im=ax[0].pcolormesh(t,fr,(Dd-Dd.max()).T,cmap="jet",vmin=-60,vmax=0,shading="auto");ax[0].set_ylabel("Doppler Hz");ax[0].set_title("Doppler spectrum |S[f]|^2, gain 20 dB")
fig.colorbar(im,ax=ax,location="right",fraction=0.025,pad=0.01,label="dB rel. max")
ax[1].plot(t,Ddb,lw=.6,alpha=.5,label="raw");ax[1].plot(t,sm(Ddb),lw=1.5,label="5-step mean (0.1 s)");ax[1].axhline(0,c="k",lw=.6);ax[1].set_ylabel("D_dB  (re noise)");ax[1].set_xlabel("s");ax[1].legend(loc="upper left",fontsize=8);ax[1].grid(alpha=.3)
pass
ax[0].set_xlim(0,t[-1])
fig.text(0.01,0.0,"D = P+ - P-,  P+ = sum_{f=+20..+500Hz}|S[f]|^2,  P- = sum_{f=-500..-20Hz}|S[f]|^2  (linear)\nD_dB = sign(D)*10*log10(|D|/(N*floor)),  N*floor = noise-only sum of one band;  D>0: positive Doppler larger.  No threshold applied.",fontsize=7.5,va="bottom",family="monospace")
plt.savefig("/home/usrpb210/sdr_workspace/RealtimeISAC/debug/ddb_163944.png",dpi=85)
