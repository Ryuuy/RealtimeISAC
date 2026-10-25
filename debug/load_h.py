#!/usr/bin/env python3
"""读 rt_main --save-h 的输出。用法: python3 load_h.py <prefix>  (prefix 不带扩展名, 如 h_logs/h_20261019_160000)

作为模块: from load_h import load_h; d = load_h(prefix)
  d['h']    complex64 (N,)    H 序列, 1kHz
  d['amp']  float32   (N,2)   每样本 ch0/ch1 RMS 幅度 (满量程=1)
  d['t']    float64   (N,)    每样本的 time.time() (秒), 同 isac_bridge session / 回车打点 / NTG csv 的钟
  d['ok']   bool      (N,)    环已灌满且未处于丢步冻结 (False 的样本别用来做分析)
  d['meta'] dict
"""
import json
import sys

import numpy as np


def load_h(prefix):
    meta = json.load(open(prefix + ".json", encoding="utf-8"))
    n_fr = meta["n_frames_per_step"]; fs = meta["fs_out"]
    t_step = np.fromfile(prefix + ".t.bin", dtype=np.float64)
    flag = np.fromfile(prefix + ".flag.bin", dtype=np.uint8)
    h = np.fromfile(prefix + ".h.bin", dtype=np.complex64)
    amp = np.fromfile(prefix + ".amp.bin", dtype=np.float32).reshape(-1, 2)
    ns = min(len(t_step), len(flag), len(h) // n_fr, len(amp) // n_fr)   # 崩溃时最后一块可能不齐
    t_step, flag = t_step[:ns], flag[:ns]
    h, amp = h[:ns * n_fr], amp[:ns * n_fr]
    t = (t_step[:, None] - (n_fr - 1 - np.arange(n_fr))[None, :] / fs).reshape(-1)
    ok = np.repeat((flag & 1 == 1) & (flag & 2 == 0), n_fr)
    return dict(h=h, amp=amp, t=t, ok=ok, meta=meta)


if __name__ == "__main__":
    d = load_h(sys.argv[1])
    print(f"{len(d['h'])} 样本 = {len(d['h'])/d['meta']['fs_out']:.1f}s, 可用 {d['ok'].mean()*100:.1f}%, "
          f"|H| 中位 {np.median(np.abs(d['h'])):.3f}, 时间跨度 {d['t'][-1]-d['t'][0]:.1f}s")
