import sys, time, numpy as np
sys.path.insert(0,"/home/usrpb210/sdr_workspace/RealtimeISAC/realtime")
from rt_config import RtConfig; from rt_sync import TddSync; from rt_source import UsrpSource
SEC=30; cfg=RtConfig(gain=20.0); n=int(SEC/cfg.step_sec); ns=cfg.n_step
buf=np.empty((2,n*ns*2),dtype=np.int16); sync=TddSync(cfg); src=UsrpSource(cfg).open(); k=0
try:
    for raw in src.steps():
        if k>=n: break
        buf[:,k*ns*2:(k+1)*ns*2]=raw; sync.update(raw,force=True); k+=1
    err,dropped=str(src.errors),src.dropped
finally: src.close()
path="/home/usrpb210/sdr_workspace/RealtimeISAC/debug/iq_%s.npz"%time.strftime("%Y%m%d_%H%M%S")
np.savez(path,iq=buf[:,:k*ns*2],fs=cfg.sample_rate,phase=sync.phase,n_int=sync.n_integrate,n_period=cfg.n_period,locked=sync.locked,contrast=sync.contrast,gain=cfg.gain,center_freq=cfg.center_freq,rx_tracking_cal=cfg.rx_tracking_cal)
print("steps",k,"UHD",err,"dropped",dropped); print(sync.status()); print("SAVED",path)
