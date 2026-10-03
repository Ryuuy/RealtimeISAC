#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 selfcaf_doppler.py 输出的基础上加第4行：带下限保护的归一化互相关。

用户想看的对比：rt_dsp.py / selfcaf_doppler.py 里现在的归一化是
    dec = conj(ch0)·ch1 / (‖ch0‖·‖ch1‖)
分母没有下限，能量很小时分母跟着一起塌，噪声段的 dec 幅度不会干净地趋于 0
（09-01 debug 的结论：ch1 弱的时候 √P1 由噪声定，杂散线相对更突出）。

这里加一个假设性的修法做对比，不改 rt_dsp.py/selfcaf_doppler.py 本体：
    dec_floor = conj(ch0)·ch1 / max(‖ch0‖·‖ch1‖, N0)
N0 取这份录制里 ‖ch0‖·‖ch1‖ 时间序列的某个低分位数（--floor-pct，默认10%），
当一个粗略的"本底噪声能量"估计——这是个简化选择，不是真拿静默段标定出来的，
只用于快速对比看形状差异，要精确结论还是得找一段真没信号的数据单独标 N0。

直接吃 selfcaf_doppler.py 已经存好的 selfcaf_*.npz（不用重新扫一遍640MB的
原始IQ，s00/s11/s01 已经是逐帧算好的量，重建出未归一化的互相关和分母就够了），
前三行图（ch0 self / ch1 self / 原版 cross-CAF）直接复用已存的谱，只重算第4行。

用法：
    /usr/bin/python3 debug/selfcaf_floor_compare.py selfcaf_20260930_150643.npz [--floor-pct 10]
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_RING, HOP, NFFT, FS_OUT = 200, 20, 256, 1000.0
DC_ALPHA, DC_GUARD_HZ = 0.01, 20.0
LINE_HZ = (50.0, 100.0, 150.0, 200.0, 300.0)


def ema_dc_remove(z, hop=HOP, alpha=DC_ALPHA):
    z = z.astype(np.complex128).copy()
    dc, dc_n = 0.0 + 0.0j, 0
    for i0 in range(0, len(z), hop):
        blk = slice(i0, i0 + hop)
        m = z[blk].mean()
        z[blk] -= dc
        dc_n += 1
        a = max(alpha, 1.0 / dc_n)
        dc = (1.0 - a) * dc + a * m
    return z


def stft_short(z):
    win = np.blackman(N_RING)
    buf = np.zeros(NFFT, dtype=np.complex128)
    freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1.0 / FS_OUT))
    specs, times = [], []
    for c in range(N_RING, len(z) + 1, HOP):
        buf[:N_RING] = z[c - N_RING:c] * win
        buf[N_RING:] = 0.0
        S = np.fft.fftshift(np.fft.fft(buf))
        specs.append(S.real ** 2 + S.imag ** 2)
        times.append(c / FS_OUT)
    return np.asarray(times), freqs, np.asarray(specs)


def line_excess(g, f_axis):
    base_m = np.abs(f_axis) >= DC_GUARD_HZ
    for f0 in LINE_HZ:
        base_m &= np.abs(np.abs(f_axis) - f0) >= 8.0
    floor = np.median(g[base_m])
    out = {}
    for f0 in LINE_HZ:
        m = np.abs(np.abs(f_axis) - f0) <= 4.0
        out[f0] = float(g[m].max() - floor) if m.any() else np.nan
    return out, floor


def main() -> int:
    p = argparse.ArgumentParser(description="加一行带下限保护归一化的互相关，跟原版对比")
    p.add_argument("npz", help="selfcaf_doppler.py 生成的 selfcaf_*.npz（不是 iq_*.npz）")
    p.add_argument("--fmax", type=float, default=300.0)
    p.add_argument("--floor-pct", type=float, default=10.0,
                   help="N0 取 ‖ch0‖·‖ch1‖ 序列的这个百分位数，默认10%%（粗略估计，不是真标定）")
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()

    path = a.npz
    if not os.path.isabs(path):
        path = os.path.join(HERE, path)
    d = np.load(path)
    s00, s11, s01 = d["s00"], d["s11"], d["s01"]
    n_int = int(d["n_int"])
    freqs_full, t_full = d["freqs"], d["t"]
    S_ch0, S_ch1, S_cross = d["S_ch0_self"], d["S_ch1_self"], d["S_cross"]

    # 重建未归一化的互相关和分母：s01 = raw01/norm, s00=p0/n_int, s11=p1/n_int
    # => norm = sqrt(s00*n_int)*sqrt(s11*n_int),  raw01 = s01 * norm
    norm = np.sqrt(s00 * n_int) * np.sqrt(s11 * n_int)
    raw01 = s01.astype(np.complex128) * norm

    n0 = np.percentile(norm, a.floor_pct)
    norm_floored = np.maximum(norm, n0)
    dec_floor = raw01 / norm_floored
    frac_floored = float(np.mean(norm < n0))
    print(f"N0(第{a.floor_pct:.0f}百分位) = {n0:.3g}   原始 norm 中位数 = {np.median(norm):.3g}   "
          f"被下限顶住的帧占比 = {100*frac_floored:.1f}%")

    tt4, freqs4, S4 = stft_short(ema_dc_remove(dec_floor))
    fmask = np.abs(freqs_full) <= a.fmax
    fmask4 = np.abs(freqs4) <= a.fmax
    f_axis = freqs_full[fmask]

    show = bool(os.environ.get("DISPLAY")) and sys.stdout.isatty()
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        ("ch0 self  |ch0|^2  (reference x itself)", t_full, freqs_full, S_ch0, fmask),
        ("ch1 self  |ch1|^2  (sensing x itself)", t_full, freqs_full, S_ch1, fmask),
        ("ch0 x ch1  cross-CAF  (original, no floor)", t_full, freqs_full, S_cross, fmask),
        (f"ch0 x ch1  cross-CAF  (floored, N0=p{a.floor_pct:.0f})", tt4, freqs4, S4, fmask4),
    ]

    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(4, 2, width_ratios=[3, 1], hspace=0.35, wspace=0.22)
    print("\n时间积分谱里 TX 线相对本底 (dB):")
    for row, (name, tt, freqs, S, fm) in enumerate(rows):
        fa = freqs[fm]
        D = 10 * np.log10(S[:, fm].T + 1e-20)
        vmin, vmax = np.percentile(D, 5), np.percentile(D, 95)
        ax = fig.add_subplot(gs[row, 0])
        ax.pcolormesh(tt, fa, D, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_ylabel("Doppler (Hz)")
        ax.set_title(name)
        ax.set_ylim(-a.fmax, a.fmax)
        for sgn in (-DC_GUARD_HZ, DC_GUARD_HZ):
            ax.axhline(sgn, color="w", ls="--", lw=0.6, alpha=0.5)
        if row == 3:
            ax.set_xlabel("Time (s)")

        g = 10 * np.log10(S[:, fm].mean(axis=0) + 1e-20)
        ex, floor = line_excess(g, fa)
        axg = fig.add_subplot(gs[row, 1])
        axg.plot(g - g.max(), fa, lw=1.0, color="tab:red" if row == 3 else "tab:blue")
        for f0 in LINE_HZ:
            for sgn in (-1, 1):
                axg.axhline(sgn * f0, color="0.7", lw=0.5, ls=":")
        axg.set_title("time-integrated")
        axg.set_xlabel("dB (norm)")
        axg.set_ylim(-a.fmax, a.fmax)
        axg.grid(alpha=0.3)
        print(f"  {name[:40]:40s} floor={floor:6.1f}  "
              + " ".join(f"{int(f)}:{ex[f]:+5.1f}" for f in LINE_HZ))

    base = os.path.basename(path)
    out = a.out or os.path.join(HERE, base.replace("selfcaf_", "selfcaf_floor4_").replace(".npz", ".png"))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n🖼  {out}")
    if show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
