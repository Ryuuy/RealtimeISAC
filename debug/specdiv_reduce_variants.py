#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为什么频谱除法还是有 ±100/±200Hz 那几条 TX 线？中位数换成平均会怎样？

在同一份原始 IQ 上，把「每帧 H(f) 塌成一个复数」这步换几种做法，其余（切窗、
采样率、EMA 去 DC、滑动 STFT）全部和 compare_caf_vs_specdiv.py 一致：

  median   : median(Re)+j·median(Im)          —— 老脚本 / 现在用的
  mean     : mean(Re)+j·mean(Im)              —— 全频段直接平均
  band_mean: 只在 |TX(f)| > 0.1·max 的 bin 上平均（避开 TX 没功率的地方）
  lsq      : Σ RX·conj(TX) / Σ|TX|²           —— 最小二乘标量估计
             （Parseval：这一项精确等于时域 Σ ch1·conj(ch0)/Σ|ch0|² = CAF）
  caf      : rt_dsp 在用的相关系数归一化 CAF，做参照

还画一栏「AM 诊断」：逐帧积分功率 P0(k)、P1(k)、比值 P1/P0 的慢时频谱——
±100/±200Hz 线的根源是**两路各自的超帧幅度调制深度不同**（P1 抖 ~2%、
P0 抖 ~0.2%），除法只消掉两路共有的东西，消不掉这个差。

    /usr/bin/python3 debug/specdiv_reduce_variants.py [iq_xxx.npz] [--no-show]
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_RING, HOP, NFFT, FS_OUT = 200, 20, 256, 1000.0
DC_ALPHA, DC_GUARD_HZ = 0.01, 20.0
LINE_HZ = (50.0, 100.0, 150.0, 200.0, 300.0, 400.0)


def newest_iq(d):
    f = sorted(glob.glob(os.path.join(d, "iq_*.npz")))
    return f[-1] if f else None


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


def stft(z):
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


def line_excess_db(g, f_axis):
    """时间积分谱 g(f) 里，各已知线频点相对"非线非DC本底"高多少 dB。"""
    base_m = (np.abs(f_axis) >= DC_GUARD_HZ)
    for f0 in LINE_HZ:
        base_m &= np.abs(np.abs(f_axis) - f0) >= 8.0
    floor = np.median(g[base_m])
    out = {}
    for f0 in LINE_HZ:
        m = np.abs(np.abs(f_axis) - f0) <= 3.0
        out[f0] = float(g[m].max() - floor) if m.any() else np.nan
    return out, floor


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="?", default=None)
    p.add_argument("--fmax", type=float, default=300.0)
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()

    path = a.npz or newest_iq(HERE)
    if not path:
        print("没有 iq_*.npz", file=sys.stderr)
        return 1
    if not os.path.isabs(path):
        path = os.path.join(HERE, path)

    show = (not a.no_show) and bool(os.environ.get("DISPLAY")) and sys.stdout.isatty()
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(path, mmap_mode="r")
    iq = d["iq"]
    fs = float(d["fs"])
    phase, n_int, n_per = int(d["phase"]), int(d["n_int"]), int(d["n_period"])
    total = iq.shape[1] // 2
    n_fr = (total - phase - n_int) // n_per
    starts = phase + np.arange(n_fr) * n_per
    base = os.path.basename(path)
    print(f"=== {base} ===  {n_fr} 帧 @ {FS_OUT:.0f}Hz  n_int={n_int}({n_int/fs*1e6:.0f}µs)")

    def load_ch(ch, s):
        seg = np.asarray(iq[ch, s * 2:(s + n_int) * 2], dtype=np.float64)
        return seg[0::2] + 1j * seg[1::2]

    methods = ("median", "mean", "band_mean", "lsq", "caf")
    red = {m: np.empty(n_fr, np.complex128) for m in methods}
    P0 = np.empty(n_fr)          # 逐帧积分功率
    P1 = np.empty(n_fr)
    Hmag_frac_lo = np.empty(n_fr)   # |TX(f)| < 0.1max 的 bin 占比（噪声主导区）

    print("逐帧算 5 种塌缩 …")
    t0 = time.time()
    for k, s in enumerate(starts):
        a0 = load_ch(0, s)
        b0 = load_ch(1, s)
        P0[k] = np.vdot(a0, a0).real
        P1[k] = np.vdot(b0, b0).real
        # CAF（参照）
        red["caf"][k] = np.vdot(a0, b0) / (np.sqrt(P0[k]) * np.sqrt(P1[k]))
        # 频谱除法
        tx = a0 - a0.mean()
        rx = b0 - b0.mean()
        TX = np.fft.fftshift(np.fft.fft(tx))
        RX = np.fft.fftshift(np.fft.fft(rx))
        atx = np.abs(TX)
        floor = max(atx.max() * 1e-9, 1e-12)
        band = atx > 0.1 * atx.max()
        Hmag_frac_lo[k] = 1.0 - band.mean()
        H = RX / np.where(atx < floor, floor, TX)
        red["median"][k] = np.median(H.real) + 1j * np.median(H.imag)
        red["mean"][k] = H.real.mean() + 1j * H.imag.mean()
        red["band_mean"][k] = (H.real[band].mean() + 1j * H.imag[band].mean()
                               if band.any() else 0.0)
        red["lsq"][k] = np.vdot(TX, RX) / np.vdot(TX, TX).real
        if (k + 1) % 1000 == 0:
            print(f"  {k+1}/{n_fr}")
    print(f"  用时 {time.time()-t0:.1f}s")
    print(f"  每帧 |TX(f)|<0.1·max 的 bin 占比：均值 {Hmag_frac_lo.mean():.1%}"
          f"（这些 bin 上 RX/TX≈噪声/噪声，mean 会被它们主导，median 不会）")

    # ---- AM 诊断：P0/P1/比值 的慢时频谱 ----
    def slow_spec(x):
        y = x / x.mean() - 1.0                 # 相对起伏
        Y = np.abs(np.fft.rfft(y * np.blackman(len(y))))
        fr = np.fft.rfftfreq(len(y), 1.0 / FS_OUT)
        return fr, 20 * np.log10(Y + 1e-20)
    fr, A0 = slow_spec(P0)
    _, A1 = slow_spec(P1)
    _, AR = slow_spec(P1 / P0)
    for nm, arr in (("P0(ch0)", A0), ("P1(ch1)", A1), ("P1/P0", AR)):
        base_m = np.ones(len(fr), bool)
        for f0 in LINE_HZ:
            base_m &= np.abs(fr - f0) >= 8.0
        fl = np.median(arr[base_m])
        peaks = {int(f0): round(float(arr[np.abs(fr - f0) <= 3].max() - fl), 1)
                 for f0 in (100, 200)}
        print(f"  {nm:8s} 慢时起伏 100/200Hz 高出本底: {peaks} dB")

    # ---- 每种塌缩过共用后端 ----
    res = {}
    for m in methods:
        tt, freqs, S = stft(ema_dc_remove(red[m]))
        res[m] = (tt, S)
    fmask = np.abs(freqs) <= a.fmax
    f_axis = freqs[fmask]

    print("\n各法时间积分谱里 TX 线高出本底 (dB):")
    print(f"  {'method':10s} {'floor':>7s} " + " ".join(f"{int(f):>5d}" for f in LINE_HZ))
    gg = {}
    for m in methods:
        _, S = res[m]
        g = 10 * np.log10(S[:, fmask].mean(axis=0) + 1e-20)
        gg[m] = g
        ex, fl = line_excess_db(g, f_axis)
        print(f"  {m:10s} {fl:7.1f} "
              + " ".join(f"{ex[f]:5.1f}" for f in LINE_HZ))

    # ---- 出图 ----
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.28)

    # 上排左中右：3 个多普勒谱（median / mean / lsq(=CAF)）
    for j, m in enumerate(("median", "mean", "lsq")):
        ax = fig.add_subplot(gs[0, j])
        tt, S = res[m]
        D = 10 * np.log10(S[:, fmask].T + 1e-20)
        vmin, vmax = np.percentile(D, 5), np.percentile(D, 95)
        ax.pcolormesh(tt, f_axis, D, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_title(f"spec-div / {m}" + ("  (= CAF, Parseval)" if m == "lsq" else ""))
        ax.set_ylabel("Doppler (Hz)")
        ax.set_xlabel("Time (s)")

    # 中排：时间积分谱叠一起
    ax = fig.add_subplot(gs[1, :])
    for m, c in zip(methods, ("tab:blue", "tab:red", "tab:green", "k", "tab:orange")):
        ax.plot(f_axis, gg[m] - gg[m].max(), lw=1.0, label=m, color=c)
    for f0 in LINE_HZ:
        for s in (-1, 1):
            ax.axvline(s * f0, color="0.7", lw=0.6, ls=":")
    ax.set_xlim(-a.fmax, a.fmax)
    ax.set_xlabel("Doppler (Hz)")
    ax.set_ylabel("dB (norm to peak)")
    ax.set_title("time-integrated Doppler spectrum — all reductions "
                 "(dotted = known TX line freqs)")
    ax.legend(ncol=5, fontsize=9)
    ax.grid(alpha=0.3)

    # 下排：AM 诊断
    ax = fig.add_subplot(gs[2, :])
    ax.plot(fr, A0 - np.median(A0), lw=0.8, label="P0 = per-frame power ch0", color="tab:blue")
    ax.plot(fr, A1 - np.median(A1), lw=0.8, label="P1 = per-frame power ch1", color="tab:red")
    ax.plot(fr, AR - np.median(AR), lw=1.1, label="P1/P0 ratio", color="k")
    for f0 in (100, 200):
        ax.axvline(f0, color="0.7", lw=0.7, ls=":")
    ax.set_xlim(0, a.fmax)
    ax.set_xlabel("slow-time frequency (Hz)")
    ax.set_ylabel("dB")
    ax.set_title("AM diagnostic: super-frame amplitude modulation of each channel's "
                 "per-frame power. The 100/200 Hz lines survive division because the "
                 "modulation depth differs between ch0 and ch1 (P1/P0 still oscillates).")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    out = os.path.join(HERE, base.replace("iq_", "specdiv_variants_").replace(".npz", ".png"))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n🖼  {out}")
    if show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
