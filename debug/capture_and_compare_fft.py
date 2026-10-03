#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键：从 USRP 现抓一段 IQ，立刻做 reference(ch0)/sensing(ch1) FFT 对比。

把 rt_iqdump.py 的采集逻辑接到 compare_ref_sensing_fft.py 的分析逻辑
（analyze_and_plot）上，省去"抓完存 npz 再手动跑分析脚本"这两步。
适合"每次都现采一次、现看一次"的排查节奏——反复跑这一个脚本就行。

默认不落盘原始 IQ（只出图 + 打印峰值表），加 --save-iq 才顺手存一份 npz
留档（留档后可以用 compare_ref_sensing_fft.py 再回放/换参数重画）。

    cd RealtimeISAC/debug
    /usr/bin/python3 capture_and_compare_fft.py                 # 抓 0.2s，出图
    /usr/bin/python3 capture_and_compare_fft.py --ms 500 --fmax 3000
    /usr/bin/python3 capture_and_compare_fft.py --save-iq        # 顺便存 npz 留档
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REALTIME_DIR = os.path.join(os.path.dirname(HERE), "realtime")
sys.path.insert(0, REALTIME_DIR)

from rt_config import RtConfig    # noqa: E402
from rt_source import UsrpSource  # noqa: E402
from rt_sync import TddSync       # noqa: E402

from compare_ref_sensing_fft import analyze_and_plot  # noqa: E402


def capture(cfg: RtConfig, n_cap: int):
    """跟 rt_iqdump.py 一样的采集方式：开流、连拉 n_cap 个 step、关流。"""
    sync = TddSync(cfg)
    src = UsrpSource(cfg).open()
    try:
        chunks = []
        for k, raw in enumerate(src.steps()):
            if k >= n_cap:
                break
            chunks.append(raw.copy())
            sync.update(raw, force=True)
        errors, dropped = str(src.errors), src.dropped
    finally:
        src.close()
    iq = np.concatenate(chunks, axis=1)      # (2, N*2) int16
    return iq, sync, errors, dropped


def main() -> int:
    p = argparse.ArgumentParser(description="现抓 IQ + reference/sensing FFT 对比，一次到位")
    p.add_argument("--ms", type=float, default=200.0, help="抓多少毫秒（默认 200ms）")
    p.add_argument("--fs", type=float, default=10.0, help="采样率 MHz")
    p.add_argument("--gain", type=float, default=30.0)
    p.add_argument("--serial", type=str, default="32392D3")
    p.add_argument("--no-cal", action="store_true",
                   help="关掉 AD9361 后台跟踪校准（一般别关，见 rt_config.py 注释）")
    p.add_argument("--fmax", type=float, default=3000.0, help="只画 ±fmax Hz（默认 3000）")
    p.add_argument("--window", choices=["none", "hann", "blackman"], default="none")
    p.add_argument("--save-iq", action="store_true", help="顺便把原始 IQ 存成 npz 留档")
    p.add_argument("--out-dir", type=str, default=HERE)
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()

    cfg = RtConfig(serial=a.serial, sample_rate=a.fs * 1e6, gain=a.gain,
                   rx_tracking_cal=not a.no_cal)
    n_cap = max(1, int(np.ceil(a.ms * 1e-3 / cfg.step_sec)))

    print(f"抓取 {a.ms:.0f}ms @ {cfg.sample_rate/1e6:.0f}MHz gain={cfg.gain:.0f}dB "
          f"serial={a.serial} ...")
    t0 = time.time()
    iq, sync, errors, dropped = capture(cfg, n_cap)
    fs = cfg.sample_rate
    total = iq.shape[1] // 2
    print(f"抓到 {total/fs*1e3:.0f} ms x2ch，用时 {time.time()-t0:.2f}s  "
          f"UHD: {errors} 丢步={dropped}")
    print(sync.status())
    if not sync.locked:
        print("⚠️ TDD 未锁定（TX 没开？）—— 不影响下面的原始 FFT 对比，只是没有 TDD 参考信息")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(a.out_dir, exist_ok=True)
    if a.save_iq:
        npz_path = os.path.join(a.out_dir, f"iq_{stamp}.npz")
        np.savez(npz_path, iq=iq, fs=fs, phase=sync.phase, n_int=sync.n_integrate,
                 n_period=cfg.n_period, locked=sync.locked, contrast=sync.contrast,
                 gain=cfg.gain, center_freq=cfg.center_freq,
                 rx_tracking_cal=cfg.rx_tracking_cal)
        print(f"\U0001f4be 原始 IQ 已存 {npz_path} ({os.path.getsize(npz_path)/1e6:.0f} MB)")

    show = (not a.no_show) and bool(os.environ.get("DISPLAY")) and sys.stdout.isatty()
    lock_txt = (f"  TDD locked phase={sync.phase}" if sync.locked
                else "  TDD not locked")
    out = os.path.join(a.out_dir, f"cmpfft_ref_sensing_{stamp}.png")
    print()
    analyze_and_plot(iq, fs, f"iq_{stamp} (live)", out, fmax=a.fmax,
                      window=a.window, show=show, extra_title=lock_txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
