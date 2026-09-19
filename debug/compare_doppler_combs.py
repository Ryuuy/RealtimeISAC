#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""并排对比几份 doppler_*.npz 的多普勒谱 + 各自的 ±k·50Hz 线谱强度曲线。

用来回答"±50/100/200Hz 那几条横线是信号自带的还是算法造的"：
把「0816 真机录制」「155243 用同一算法离线重放」「合成对照(无周期调幅)」放一起看。
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    d = np.load(path)
    meta = json.loads(str(d["meta_json"]))
    r = d["ready"].astype(bool)
    if r.sum() < 5:
        r = np.ones(len(d["t"]), bool)
    spec, t, freqs = d["spec"][r], d["t"][r], d["freqs"]
    return spec, t, freqs, meta


def comb_profile(spec, freqs):
    P = (np.abs(spec) ** 2).mean(axis=0)
    Pdb = 10 * np.log10(P + 1e-20)
    k = 31
    from numpy.lib.stride_tricks import sliding_window_view
    base = np.median(sliding_window_view(np.pad(Pdb, k // 2, mode="edge"), k), axis=-1)
    return freqs, Pdb - base


def main(argv):
    # (label, path)
    items = []
    for a in argv:
        if "=" in a:
            lab, p = a.split("=", 1)
        else:
            lab, p = os.path.basename(a), a
        if not os.path.isabs(p):
            p = os.path.join(HERE, p)
        items.append((lab, p))
    if not items:
        print("用法: compare_doppler_combs.py  标签=xxx.npz  标签=yyy.npz ...", file=sys.stderr)
        return 1

    n = len(items)
    fig, axes = plt.subplots(2, n, figsize=(6.2 * n, 9),
                             gridspec_kw={"height_ratios": [2.4, 1.0]})
    if n == 1:
        axes = axes.reshape(2, 1)

    for c, (lab, path) in enumerate(items):
        spec, t, freqs, meta = load(path)
        Sdb = 10 * np.log10(np.abs(spec.T) ** 2 + 1e-20)
        vmin, vmax = np.percentile(Sdb, 5), np.percentile(Sdb, 95)
        ax = axes[0, c]
        ax.pcolormesh(t, freqs, Sdb, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        lk = meta.get("sync_locked")
        ax.set_title(f"{lab}\nfs={meta['sample_rate']/1e6:.0f}MHz  "
                     f"TDD {'LOCKED' if lk else 'free-run'}  "
                     f"duty={meta.get('sync_duty',0)*100:.0f}%", fontsize=10)
        ax.set_xlabel("Time (s)")
        if c == 0:
            ax.set_ylabel("Doppler (Hz)")
        for h in range(-450, 451, 50):
            ax.axhline(h, color="w", lw=0.3, alpha=0.25)

        fr, ex = comb_profile(spec, freqs)
        axp = axes[1, c]
        axp.plot(fr, ex, lw=0.8, color="tab:blue")
        for h in range(-450, 451, 50):
            axp.axvline(h, color="tab:red", lw=0.5, alpha=0.3)
        axp.axhline(8, color="k", ls=":", lw=0.8)
        axp.set_xlim(-500, 500)
        axp.set_ylim(-3, max(20, float(np.percentile(ex, 99.5))))
        axp.set_xlabel("Doppler (Hz)")
        if c == 0:
            axp.set_ylabel("dB above local floor")
        axp.grid(alpha=0.3)
        # 标 ±50k 处的值
        for f0 in (50, 100, 150, 200):
            i = int(np.argmin(np.abs(fr - f0)))
            v = float(ex[max(0, i - 1):i + 2].max())
            axp.annotate(f"{v:+.0f}", (f0, v), fontsize=7, ha="center",
                         va="bottom", color="tab:red")

    fig.suptitle("+-(50k) Hz Doppler comb: SAME algorithm, different input data", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(HERE, "compare_doppler_combs.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"🖼  {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
