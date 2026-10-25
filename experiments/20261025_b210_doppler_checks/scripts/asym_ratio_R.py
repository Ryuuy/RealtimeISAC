import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/home/usrpb210/sdr_workspace/RealtimeISAC/realtime")
from rt_config import RtConfig; from rt_dsp import DopplerEngine; from rt_sync import TddSync
tag="160836"; iq=np.load(f"/home/usrpb210/sdr_workspace/RealtimeISAC/debug/iq_20261025_{tag}.npz")["iq"]
for c in range(2):
    m=np.abs(iq[c].astype(np.int32)).max(); print(f"ch{c} max={m} ({m/32768*100:.1f}%FS) n>=0.9FS={(np.abs(iq[c])>=29491).sum()}")
cfg=RtConfig(gain=20.0); ns=cfg.n_step; K=iq.shape[1]//2//ns
eng=DopplerEngine(cfg); sync=TddSync(cfg); P=[]; A=[]
for k in range(K):
    raw=np.ascontiguousarray(iq[:,k*ns*2:(k+1)*ns*2]); sync.update(raw); eng.ingest(raw)
    P.append(eng.process(sync.phase,sync.n_integrate).copy()); A.append(eng.last_amp.mean(0))
P=np.array(P); A=np.array(A); fr=eng.freqs; t=np.arange(K)*0.02
print("locked",sync.locked,"ring primed after 0.2s")
band=(np.abs(fr)>=20)&(np.abs(fr)<=500)
Pp=P[:,band&(fr>0)].sum(1); Pn=P[:,band&(fr<0)].sum(1)
R=(Pp-Pn)/(Pp+Pn)        # linear power, in [-1,1]
np.save(f"/tmp/R_{tag}.npy",R)
# amplitude (|S|) variant
Sp=np.sqrt(P[:,band&(fr>0)]).sum(1); Sn=np.sqrt(P[:,band&(fr<0)]).sum(1); Ra=(Sp-Sn)/(Sp+Sn)
fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True)
D=10*np.log10(P+1e-20); im=ax[0].pcolormesh(t,fr,(D-D.max()).T,cmap="jet",vmin=-60,vmax=0,shading="auto");ax[0].set_ylabel("Hz");plt.colorbar(im,ax=ax[0]);ax[0].set_title("gain 20 dB, Doppler spectrum |S[f]|^2 (dB, relative)")
ax[1].plot(t,10*np.log10(A[:,0]**2+1e-20),label="ch0");ax[1].plot(t,10*np.log10(A[:,1]**2+1e-20),label="ch1");ax[1].set_ylabel("dBFS");ax[1].legend()
ax[2].plot(t,Ra,lw=1.6,label=r"R$_{abs}$ (main): uses |S|");ax[2].plot(t,R,lw=.8,alpha=.6,label=r"R$_{pow}$: uses |S|$^2$");ax[2].axhline(0,c="k",lw=.5);ax[2].set_ylim(-1,1);ax[2].set_ylabel("R  (-1..+1)");ax[2].set_xlabel("s");ax[2].legend()
ax[2].legend(loc="upper right",fontsize=8)
fig.text(0.01,0.005,"S[f] = FFT{ Blackman window x dec(t) }  (200 ms, 20 ms step);  dec(t) = sum(conj(ch0)*ch1) / (||ch0||*||ch1||) per TDD frame, 1 kHz\n"
 "A+ = sum over f = +20..+500 Hz of |S[f]|     A- = sum over f = -500..-20 Hz of |S[f]|     R_abs = (A+ - A-) / (A+ + A-)     (linear, not dB; R>0: positive Doppler larger)\n"
 "R_pow = same formula with |S[f]|^2 instead of |S[f]|",fontsize=7.5,va="bottom",family="monospace")
plt.tight_layout(rect=(0,0.06,1,1));plt.savefig(f"/home/usrpb210/sdr_workspace/RealtimeISAC/debug/asym_{tag}.png",dpi=85)
print(" t(s)  R_abs(|S|)  A+/A-   R_pow(|S|^2)  P+/P-")
for s in range(0,20):
    i=slice(max(s*50,10),(s+1)*50)
    pp,pn=Pp[i].sum(),Pn[i].sum(); sp,sn=Sp[i].sum(),Sn[i].sum()
    r=(pp-pn)/(pp+pn); ra=(sp-sn)/(sp+sn); print(f"{s:2d}-{s+1:2d}  {ra:+8.3f}   {sp/sn:6.2f}x   {r:+8.3f}   {pp/pn:6.2f}x")
i=slice(10,None); pp,pn=Pp[i].sum(),Pn[i].sum(); print(f"整段: R_abs={(Sp[i].sum()-Sn[i].sum())/(Sp[i].sum()+Sn[i].sum()):+.3f} (A+/A-={Sp[i].sum()/Sn[i].sum():.2f}x)   R_pow={(pp-pn)/(pp+pn):+.3f} (P+/P-={pp/pn:.2f}x)")
