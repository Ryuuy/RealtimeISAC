import sys, numpy as np
sys.path.insert(0,"/home/usrpb210/sdr_workspace/RealtimeISAC/realtime")
import rt_main
iq=np.load("/home/usrpb210/sdr_workspace/RealtimeISAC/debug/iq_20261025_165202.npz")["iq"]
def replay(cfg,duration,target_period=4.0):
    ns=cfg.n_step
    for k in range(iq.shape[1]//2//ns): yield np.ascontiguousarray(iq[:,k*ns*2:(k+1)*ns*2])
rt_main._synthetic_steps=replay
out="/tmp/claude-1000/-home-usrpb210-sdr-workspace/d5275eac-6f68-4b1c-a405-2c9f99e6186d/scratchpad/rtout"
sys.argv=["rt_main","--dry-run","--duration","60","--debug","--debug-dir",out,"--debug-max-sec","40"]
rt_main.main()
