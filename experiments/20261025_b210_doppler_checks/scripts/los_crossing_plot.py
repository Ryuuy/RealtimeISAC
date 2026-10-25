import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/home/usrpb210/sdr_workspace/RealtimeISAC/realtime")
from rt_config import RtConfig; from rt_dsp import DopplerEngine; from rt_sync import TddSync
f="/home/usrpb210/sdr_workspace/RealtimeISAC/debug/iq_20261025_160336.npz"
iq=np.load(f)["iq"]; cfg=RtConfig(); ns=cfg.n_step; K=iq.shape[1]//2//ns
eng=DopplerEngine(cfg); sync=TddSync(cfg); P=[]; amp=[]; ok=[]; lk=[]; ph=[]
for k in range(K):
    raw=np.ascontiguousarray(iq[:,k*ns*2:(k+1)*ns*2]); sync.update(raw); eng.ingest(raw)
    P.append(eng.process(sync.phase,sync.n_integrate).copy()); amp.append(eng.last_amp.mean(0)); lk.append(sync.locked); ph.append(sync.phase)
P=np.array(P); amp=np.array(amp); lk=np.array(lk); fr=eng.freqs; t=np.arange(K)*0.02
print("locked frac %.3f, phase %d->%d"%(lk.mean(),ph[0],ph[-1]))
a=iq.reshape(2,-1,2)[:,:,:] ; print("max|IQ| ch0 %d ch1 %d"%(np.abs(iq[0]).max(),np.abs(iq[1]).max()))
D=10*np.log10(P+1e-20); m=eng.dc_mask
# signed centroid of non-DC power
pw=P*m; cen=(pw*fr).sum(1)/(pw.sum(1)+1e-20)
pos=P[:,fr>20].sum(1); neg=P[:,fr<-20].sum(1)
ad=10*np.log10(amp.mean(1)**2+1e-20)
fig,ax=plt.subplots(4,1,figsize=(12,11),sharex=True)
im=ax[0].pcolormesh(t,fr,(D-D.max()).T,cmap="jet",vmin=-60,vmax=0,shading="auto");ax[0].set_ylabel("Doppler Hz");ax[0].set_title("Live-sync replay of saved IQ (LOS crossing)");plt.colorbar(im,ax=ax[0])
ax[1].plot(t,10*np.log10(amp[:,0]**2+1e-20),label="ch0");ax[1].plot(t,10*np.log10(amp[:,1]**2+1e-20),label="ch1");ax[1].set_ylabel("dBFS");ax[1].legend()
ax[2].plot(t,10*np.log10(pos/neg));ax[2].axhline(0,c="k",lw=.5);ax[2].set_ylabel("pos/neg power dB")
ax[3].plot(t,cen);ax[3].axhline(0,c="k",lw=.5);ax[3].set_ylabel("centroid Hz");ax[3].set_xlabel("s")
plt.tight_layout();plt.savefig("/home/usrpb210/sdr_workspace/RealtimeISAC/debug/los_160336.png",dpi=85)
for s in range(0,20):
    i=slice(s*50,(s+1)*50); print(f"{s:2d}s ch0={ad[i].mean():6.1f}dB  centroid={cen[i].mean():7.1f}Hz  pos/neg={10*np.log10(pos[i].sum()/neg[i].sum()):5.1f}dB  peakdb={D[i][:,m].max()-np.median(D[:, m]):5.1f}")
