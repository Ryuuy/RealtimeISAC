#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 ablate_doppler.run_chain 的几个关键配置画出来，直观展示 ±50·k Hz 梳的成因：
逐 step(0.02s=50Hz) 做一次的**块状去DC**。

    cd RealtimeISAC/realtime
    python ../debug/comb_rootcause.py ../../experiment_30MHz_static_20260710_155243 --sec 18
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "realtime"))
sys.path.insert(0, HERE)

from rt_config import RtConfig            # noqa: E402
from ablate_doppler import run_chain, load_params   # noqa: E402


def excess(spec, freqs):
    rows = np.any(spec != 0, axis=1)
    P = (np.abs(spec[rows]) ** 2).mean(axis=0)
    Pdb = 10 * np.log10(P + 1e-20)
    k = 31
    base = np.median(sliding_window_view(np.pad(Pdb, k // 2, mode="edge"), k), axis=-1)
    return Pdb - base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--sec", type=float, default=18.0)
    ap.add_argument("--fs", type=float, default=None)
    a = ap.parse_args()

    params = load_params(a.folder)
    sr = (a.fs * 1e6) if a.fs else float(params.get("sample_rate_hz", 30e6))
    cfg = RtConfig(sample_rate=sr, center_freq=float(params.get("center_freq_hz", 1.89e9)),
                   gain=float(params.get("gain_db", 30.0)))
    bin_path = os.path.join(a.folder, "2ch_iq_data.bin")
    n_steps = int(round(a.sec / cfg.step_sec))

    cases = [
        ("baseline: per-step (50Hz) block DC removal  [rt_dsp now]", dict(norm="coeff", gate="on", dc="ema",          win="blackman"), "tab:red"),
        ("DC EMA updated per-frame (1kHz) instead",                  dict(norm="coeff", gate="on", dc="ema_perframe", win="blackman"), "tab:green"),
        ("subtract current 0.2s window mean (one-shot)",             dict(norm="coeff", gate="on", dc="winmean",      win="blackman"), "tab:blue"),
        ("no DC removal at all",                                     dict(norm="coeff", gate="on", dc="none",         win="blackman"), "0.4"),
    ]

    specs = []
    for name, kw, _ in cases:
        spec, freqs, sync = run_chain(bin_path, cfg, n_steps, **kw)
        specs.append(spec)
        print(f"done: {name}", flush=True)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.18)

    # 上：时间平均谱（相对局部本底），四个配置叠一起
    ax = fig.add_subplot(gs[0, :])
    for (name, _, col), spec in zip(cases, specs):
        ax.plot(freqs, excess(spec, freqs), lw=1.0, color=col, label=name)
    for h in range(-450, 451, 50):
        ax.axvline(h, color="k", lw=0.4, alpha=0.15)
    ax.axhline(5, color="k", ls=":", lw=0.8)
    ax.set_xlim(-500, 500)
    ax.set_ylim(-4, 16)
    ax.set_xlabel("Doppler (Hz)")
    ax.set_ylabel("dB above local median")
    ax.set_title("Time-averaged Doppler spectrum: the +-50k Hz comb appears ONLY with "
                 "per-step block DC removal (step=0.02s -> 50Hz)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)

    # 下：baseline vs 完全不去DC 的谱图
    for j, idx in enumerate((0, 3)):
        axs = fig.add_subplot(gs[1, j])
        spec = specs[idx]
        rows = np.any(spec != 0, axis=1)
        Sdb = 10 * np.log10(np.abs(spec[rows].T) ** 2 + 1e-20)
        t = np.arange(Sdb.shape[1]) * cfg.step_sec
        vmin, vmax = np.percentile(Sdb, 5), np.percentile(Sdb, 97)
        axs.pcolormesh(t, freqs, Sdb, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        for h in range(-450, 451, 50):
            axs.axhline(h, color="w", lw=0.3, alpha=0.25)
        axs.set_title(cases[idx][0], fontsize=10)
        axs.set_xlabel("Time (s)")
        axs.set_ylabel("Doppler (Hz)")
        axs.set_ylim(-500, 500)

    fig.suptitle(f"Root cause of the +-50k Hz Doppler comb — {os.path.basename(a.folder)}  "
                 f"({a.sec:.0f}s, fs={sr/1e6:.0f}MHz)", fontsize=13)
    out = os.path.join(HERE, "comb_rootcause_dc_perstep.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"图: {out}")


if __name__ == "__main__":
    main()
