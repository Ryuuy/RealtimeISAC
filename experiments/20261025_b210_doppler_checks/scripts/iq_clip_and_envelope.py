import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=np.load("/home/usrpb210/sdr_workspace/RealtimeISAC/debug/iq_20261025_154643.npz")
iq=d["iq"]; fs=float(d["fs"]); ph=int(d["phase"]); ni=int(d["n_int"]); npr=int(d["n_period"])
print("fs",fs,"phase",ph,"n_int",ni,"n_period",npr,"cal",d["rx_tracking_cal"],"gain",d["gain"])
N=iq.shape[1]//2; P=N//npr
x=iq[:,:P*npr*2].reshape(2,P,npr,2)
for c in range(2):
    a=np.abs(x[c].astype(np.int32)); 
    print(f"ch{c}: max|I/Q|={a.max()}  >=32000: {(a>=32000).mean()*100:.5f}%  >=30000: {(a>=30000).mean()*100:.5f}%  p99.9={np.percentile(a[::50,::50],99.9):.0f}  DC I={x[c,...,0].mean():.1f} Q={x[c,...,1].mean():.1f}")
# envelope inside 1ms period, 50us bins
nb=npr//500
env=np.zeros((2,nb))
for c in range(2):
    p=(x[c,:2000].astype(np.float32)**2).sum(-1)       # (2000,npr)
    env[c]=p.reshape(2000,nb,500).mean((0,2))
envdb=10*np.log10(env/32768**2+1e-20)
print("envelope dBFS per 50us bin (ch0):",np.round(envdb[0],0).astype(int).tolist())
print("envelope dBFS per 50us bin (ch1):",np.round(envdb[1],0).astype(int).tolist())
# dec stream
sl=x[:,:,ph:ph+ni,:].astype(np.float32)
a=sl[0,...,0]+1j*sl[0,...,1]; b=sl[1,...,0]+1j*sl[1,...,1]
num=(np.conj(a)*b).sum(1); dec=num/np.sqrt((abs(a)**2).sum(1)*(abs(b)**2).sum(1))
print("|dec| median",np.median(abs(dec)),"phase std(rad)",np.std(np.angle(dec)))
# spectrogram 200 ms window, 20ms step, Blackman, nfft256
nr,nfft=200,256; w=np.blackman(nr); rows=[]
for s in range(0,P-nr,20):
    S=np.fft.fftshift(np.fft.fft(dec[s:s+nr]*w,nfft)); rows.append(abs(S)**2)
R=10*np.log10(np.array(rows).T+1e-20); fr=np.fft.fftshift(np.fft.fftfreq(nfft,1e-3))
pos=np.array(rows)[:,fr>20].sum(1); neg=np.array(rows)[:,fr<-20].sum(1)
asym=10*np.log10(pos/neg)
print("pos/neg power ratio dB per-step: mean %.2f, std %.2f, |mean abs| %.2f"%(asym.mean(),asym.std(),abs(asym).mean()))
# real-valued test: how much of dec fluctuation is amplitude (real after phase rotation) vs phase
d0=dec-dec.mean(); u=dec.mean()/abs(dec.mean())
rad=(d0*np.conj(u)).real; tan=(d0*np.conj(u)).imag
print("fluct power along mean-phasor (amplitude-like): %.3g ; orthogonal (phase-like): %.3g"%(rad.var(),tan.var()))
t=np.arange(P)*1e-3
fig,ax=plt.subplots(4,1,figsize=(12,11),sharex=False)
ax[0].plot(np.arange(nb)*50,envdb[0],label="ch0");ax[0].plot(np.arange(nb)*50,envdb[1],label="ch1");ax[0].axvspan(ph/fs*1e6,(ph+ni)/fs*1e6,alpha=.2);ax[0].set_xlabel("us in 1ms period");ax[0].set_ylabel("dBFS");ax[0].legend();ax[0].set_title("Mean power over TDD period (shade=integration window)")
im=ax[1].pcolormesh(np.arange(R.shape[1])*0.02,fr,R-R.max(),cmap="jet",vmin=-60,vmax=0,shading="auto");ax[1].set_ylabel("Hz");ax[1].set_title("Doppler (recomputed offline from saved IQ)");plt.colorbar(im,ax=ax[1])
ax[2].plot(t,np.angle(dec),lw=.4);ax[2].set_ylabel("angle(dec) rad")
ax[3].plot(t,abs(dec),lw=.4);ax[3].set_ylabel("|dec|");ax[3].set_xlabel("s")
plt.tight_layout();plt.savefig("/home/usrpb210/sdr_workspace/RealtimeISAC/debug/iqcheck_154643.png",dpi=90)
