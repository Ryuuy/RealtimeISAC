#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自相关（自己 × 自己*）的多普勒谱：conj(Rx)·Rx = |Rx|²，逐帧积分。

平时 CAF 是 **互相关** conj(ch0)·ch1（参考 × 感知）。这里换成 **自相关**：
  ch0 自己： s00[k] = Σ_t |ch0(t)|²   （参考通道自己的逐帧功率）
  ch1 自己： s11[k] = Σ_t |ch1(t)|²   （感知通道自己的逐帧功率）
再拿这条 1kHz 的实数流走和 compare_three_doppler.py 一样的后端
（EMA 去 DC + 200帧/0.2s 滑动 Blackman STFT）。同时画互相关 s01 做对照。

用途：把感知通道堵住之后，看 ±100/±200Hz 那几条 TX 线到底是不是**只在
参考通道自己的幅度里**——如果 ch0 自相关还有线、ch1 自相关和互相关都塌了，
就说明那几条线是 TX 发射信号自带的调制，跟感知路径无关。

⚠️ 自相关流是**实数**，多普勒谱天然共轭对称（±f 一样），分不出趋近/远离，
只反映 |调制|。这是它和互相关 CAF 的本质区别。

    /usr/bin/python3 debug/selfcaf_doppler.py [iq_xxx.npz] [--fmax 300] [--no-show]
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
LINE_HZ = (50.0, 100.0, 150.0, 200.0, 300.0)


def newest_iq(dirp):
    f = sorted(glob.glob(os.path.join(dirp, "iq_*.npz")))
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
    p = argparse.ArgumentParser(description="自相关多普勒谱")
    p.add_argument("npz", nargs="?", default=None)
    p.add_argument("--fmax", type=float, default=300.0)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()

    path = a.npz or newest_iq(HERE)
    if not path:
        print("没有 iq_*.npz。先采：cd realtime && /usr/bin/python3 rt_iqdump.py --ms 5000",
              file=sys.stderr)
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
    locked, contrast = bool(d["locked"]), float(d["contrast"])
    total = iq.shape[1] // 2
    base = os.path.basename(path)
    print(f"=== {base} ===  {total/fs*1e3:.0f}ms x2ch  fs={fs/1e6:.0f}MHz  "
          f"phase={phase} n_int={n_int}({n_int/fs*1e6:.0f}µs)  "
          f"{'锁定' if locked else '未锁定'} contrast={contrast:.2f}")
    if not locked:
        print("⚠️ TDD 未锁定：n_int 退化成整周期，'门控'名存实亡", file=sys.stderr)

    n_fr = (total - phase - n_int) // n_per
    starts = phase + np.arange(n_fr) * n_per

    def load_frame(ch, s):
        seg = np.asarray(iq[ch, s * 2:(s + n_int) * 2], dtype=np.float64)
        return seg[0::2] + 1j * seg[1::2]

    print(f"逐帧算自相关 s00/s11 + 互相关 s01（{n_fr} 帧）…")
    t0 = time.time()
    s00 = np.empty(n_fr, np.float64)      # Σ|ch0|²  实数
    s11 = np.empty(n_fr, np.float64)      # Σ|ch1|²  实数
    s01 = np.empty(n_fr, np.complex128)   # Σ conj(ch0)·ch1  复数
    for k, s in enumerate(starts):
        a0 = load_frame(0, s)
        b0 = load_frame(1, s)
        p0 = np.vdot(a0, a0).real
        p1 = np.vdot(b0, b0).real
        s00[k] = p0 / n_int
        s11[k] = p1 / n_int
        s01[k] = np.vdot(a0, b0) / (np.sqrt(p0) * np.sqrt(p1))
    print(f"  用时 {time.time()-t0:.1f}s")
    print(f"  ch0 功率 {10*np.log10(s00.mean()/32767**2):.1f}dBFS  "
          f"ch1 功率 {10*np.log10(s11.mean()/32767**2):.1f}dBFS")

    streams = [
        ("ch0 self  |ch0|^2  (reference x itself)", s00.astype(np.complex128)),
        ("ch1 self  |ch1|^2  (sensing x itself)", s11.astype(np.complex128)),
        ("ch0 x ch1  cross-CAF  (normal Doppler)", s01),
    ]
    res = []
    for name, z in streams:
        tt, freqs, S = stft_short(ema_dc_remove(z))
        res.append((name, tt, freqs, S))

    fmask = np.abs(res[0][2]) <= a.fmax
    f_axis = res[0][2][fmask]

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1], hspace=0.33, wspace=0.22)
    print("\n时间积分谱里 TX 线相对本底 (dB):")
    for row, (name, tt, freqs, S) in enumerate(res):
        D = 10 * np.log10(S[:, fmask].T + 1e-20)
        vmin, vmax = np.percentile(D, 5), np.percentile(D, 95)
        ax = fig.add_subplot(gs[row, 0])
        ax.pcolormesh(tt, f_axis, D, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_ylabel("Doppler (Hz)")
        ax.set_title(name)
        ax.set_ylim(-a.fmax, a.fmax)
        for sgn in (-DC_GUARD_HZ, DC_GUARD_HZ):
            ax.axhline(sgn, color="w", ls="--", lw=0.6, alpha=0.5)
        if row == 2:
            ax.set_xlabel("Time (s)")

        g = 10 * np.log10(S[:, fmask].mean(axis=0) + 1e-20)
        ex, floor = line_excess(g, f_axis)
        axg = fig.add_subplot(gs[row, 1])
        axg.plot(g - g.max(), f_axis, lw=1.0, color="tab:blue")
        for f0 in LINE_HZ:
            for sgn in (-1, 1):
                axg.axhline(sgn * f0, color="0.7", lw=0.5, ls=":")
        axg.set_title("time-integrated")
        axg.set_xlabel("dB (norm)")
        axg.set_ylim(-a.fmax, a.fmax)
        axg.grid(alpha=0.3)
        print(f"  {name[:34]:34s} floor={floor:6.1f}  "
              + " ".join(f"{int(f)}:{ex[f]:+5.1f}" for f in LINE_HZ))

    out = a.out or os.path.join(HERE, base.replace("iq_", "selfcaf_").replace(".npz", ".png"))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n🖼  {out}")

    npz_out = out.replace(".png", ".npz")
    np.savez(npz_out, freqs=res[0][2], t=res[0][1],
             S_ch0_self=res[0][3].astype(np.float32),
             S_ch1_self=res[1][3].astype(np.float32),
             S_cross=res[2][3].astype(np.float32),
             s00=s00, s11=s11, s01=s01.astype(np.complex64),
             fs=fs, phase=phase, n_int=n_int, src=base)
    print(f"💾 {npz_out}")

    if show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
