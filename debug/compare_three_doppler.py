#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三种多普勒算法对比，同一份原始 IQ：

  1. CAF + TDD门控   : 每个 ON 窗积分成一个复数 -> 1kHz 流 -> 短 STFT
                       （RealtimeISAC 现在在用的，rt_dsp.DopplerEngine）
  2. 频谱除法 + TDD门控: 每个 ON 窗 H(f)=FFT(ch1)/FFT(ch0) 取 median -> 1kHz 流 -> 短 STFT
                       （老 Udoppler_spectrum_division_20MHz.py 的做法）
  3. 连续CAF / 不切窗  : 整段流按 decim 抽取 -> conj(ch0)·ch1 -> Tw 窗 STFT
                       （reference/validate_capture_and_caf_fc32test.py 的做法，
                        没有 TDD 门控，OFF 段也进去了）

前两种和 compare_caf_vs_specdiv.py 一样：同样的切窗（采集时存的 phase/n_int/
n_period）、同样的 EMA 去 DC、同一套 200帧/0.2s 滑动 STFT。
第三种是另一路子：不按 TDD 帧切，直接抽取整段连续流做长窗 STFT——它自带
Tw=0.4s、decim=16，跟 validate 脚本一致（所以频率分辨率/窗长跟前两种不同，
这正是"切窗 vs 不切窗"要看的差别）。

    /usr/bin/python3 debug/compare_three_doppler.py [iq_xxx.npz] [--fmax 300] [--no-show]
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
from scipy.signal import spectrogram, windows

HERE = os.path.dirname(os.path.abspath(__file__))

# 前两法的后端（和 rt_config.RtConfig 默认一致）
N_RING, HOP, NFFT, FS_OUT = 200, 20, 256, 1000.0
DC_ALPHA, DC_GUARD_HZ = 0.01, 20.0
# 第三法（validate_capture_and_caf_fc32test.py 的参数）
NG_DECIM = 16
NG_TW = 0.4
NG_STEP = 0.02


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


def stft_short(z):
    """前两法共用后端：200帧窗、hop 20、Blackman、256补零，返回线性 |S|^2。"""
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


def main() -> int:
    p = argparse.ArgumentParser(description="三种多普勒算法对比")
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
          f"phase={phase} n_int={n_int}({n_int/fs*1e6:.0f}µs) n_per={n_per}  "
          f"{'锁定' if locked else '未锁定'} contrast={contrast:.2f}")

    n_fr = (total - phase - n_int) // n_per
    starts = phase + np.arange(n_fr) * n_per

    def load_frame(ch, s):
        seg = np.asarray(iq[ch, s * 2:(s + n_int) * 2], dtype=np.float64)
        return seg[0::2] + 1j * seg[1::2]

    # ---- 1 & 2：TDD 门控，逐 ON 窗塌成一个复数 ----
    print(f"[1,2] 逐 TDD 帧（{n_fr} 帧 @ {FS_OUT:.0f}Hz）算 CAF + 频谱除法 …")
    t0 = time.time()
    caf = np.empty(n_fr, np.complex128)
    sdv = np.empty(n_fr, np.complex128)
    for k, s in enumerate(starts):
        a0 = load_frame(0, s)
        b0 = load_frame(1, s)
        P0 = np.vdot(a0, a0).real
        P1 = np.vdot(b0, b0).real
        caf[k] = np.vdot(a0, b0) / (np.sqrt(P0) * np.sqrt(P1))
        tx = a0 - a0.mean()
        rx = b0 - b0.mean()
        TX = np.fft.fftshift(np.fft.fft(tx))
        RX = np.fft.fftshift(np.fft.fft(rx))
        atx = np.abs(TX)
        floor = max(atx.max() * 1e-9, 1e-12)
        H = RX / np.where(atx < floor, floor, TX)
        sdv[k] = np.median(H.real) + 1j * np.median(H.imag)
    print(f"  用时 {time.time()-t0:.1f}s")
    t_caf, freqs_s, S_caf = stft_short(ema_dc_remove(caf))
    t_sdv, _, S_sdv = stft_short(ema_dc_remove(sdv))

    # ---- 3：不切窗，整段抽取 + 连续 conj 乘积 + Tw 窗 STFT ----
    # decim=16 复数样点 = int16 上 stride 32。纯子采样（不加抗混叠滤波），
    # 和 validate_capture_and_caf_fc32test.build_decimated_channel 一致。
    print(f"[3] 不切窗：整段按 decim={NG_DECIM} 抽取，Tw={NG_TW}s 连续 STFT …")
    t0 = time.time()
    st = 2 * NG_DECIM
    c0 = (np.asarray(iq[0, 0:2 * total:st], np.float64)
          + 1j * np.asarray(iq[0, 1:2 * total:st], np.float64))
    c1 = (np.asarray(iq[1, 0:2 * total:st], np.float64)
          + 1j * np.asarray(iq[1, 1:2 * total:st], np.float64))
    n = min(len(c0), len(c1))
    prod = np.conj(c0[:n]) * c1[:n]
    eff_fs = fs / NG_DECIM
    nperseg = int(eff_fs * NG_TW)
    noverlap = int(eff_fs * (NG_TW - NG_STEP))
    win = windows.blackman(nperseg)
    f3, t3, Sxx = spectrogram(prod, fs=eff_fs, window=win, nperseg=nperseg,
                              noverlap=noverlap, return_onesided=False, mode="complex")
    f3 = np.fft.fftshift(f3)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    S_ng = (np.abs(Sxx) ** 2).T          # (n_time, n_freq)
    print(f"  用时 {time.time()-t0:.1f}s   eff_fs={eff_fs/1e3:.0f}kHz  nperseg={nperseg}  "
          f"bin={eff_fs/nperseg:.1f}Hz  {S_ng.shape[0]} 时间帧")

    # ---- 出图：三行时频 + 三条时间积分谱 ----
    def crop(f, S):
        m = np.abs(f) <= a.fmax
        return f[m], S[:, m]

    fa_s, Sc = crop(freqs_s, S_caf)
    _, Sd = crop(freqs_s, S_sdv)
    fa_3, S3 = crop(f3, S_ng)

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1], hspace=0.33, wspace=0.22)
    panels = [
        ("1. CAF + TDD gating  (rt_dsp)  Tw=0.2s", t_caf, fa_s, Sc),
        ("2. Spectrum division + TDD gating  (median)  Tw=0.2s", t_sdv, fa_s, Sd),
        (f"3. Continuous CAF, NO gating  (validate_..fc32)  "
         f"decim={NG_DECIM}, Tw={NG_TW}s", t3, fa_3, S3),
    ]
    for row, (name, tt, fa, S) in enumerate(panels):
        D = 10 * np.log10(S.T + 1e-20)
        vmin, vmax = np.percentile(D, 5), np.percentile(D, 95)
        ax = fig.add_subplot(gs[row, 0])
        ax.pcolormesh(tt, fa, D, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_ylabel("Doppler (Hz)")
        ax.set_title(name)
        ax.set_ylim(-a.fmax, a.fmax)
        for sgn in (-DC_GUARD_HZ, DC_GUARD_HZ):
            ax.axhline(sgn, color="w", ls="--", lw=0.6, alpha=0.5)
        if row == 2:
            ax.set_xlabel("Time (s)")

        axg = fig.add_subplot(gs[row, 1])
        g = 10 * np.log10(S.mean(axis=0) + 1e-20)
        axg.plot(g - g.max(), fa, lw=1.0, color="tab:blue")
        for f0 in (50, 100, 150, 200, 300):
            for sgn in (-1, 1):
                axg.axhline(sgn * f0, color="0.7", lw=0.5, ls=":")
        axg.set_title("time-integrated")
        axg.set_xlabel("dB (norm)")
        axg.set_ylim(-a.fmax, a.fmax)
        axg.grid(alpha=0.3)

    out = a.out or os.path.join(HERE, base.replace("iq_", "cmp3_").replace(".npz", ".png"))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n🖼  {out}")

    npz_out = out.replace(".png", ".npz")
    np.savez(npz_out, t_caf=t_caf, t_sdv=t_sdv, t_ng=t3,
             freqs_short=freqs_s, freqs_ng=f3,
             S_caf=S_caf.astype(np.float32), S_sdv=S_sdv.astype(np.float32),
             S_ng=S_ng.astype(np.float32),
             fs=fs, phase=phase, n_int=n_int, n_period=n_per,
             ng_decim=NG_DECIM, ng_tw=NG_TW, src=base)
    print(f"💾 {npz_out}")

    # 数字：DC 与 ±1000Hz TDD 谐波泄漏（不切窗那条会有）
    def excess_at(f, S, f0, bw=8.0):
        g = 10 * np.log10(S.mean(axis=0) + 1e-20)
        base = np.median(g[np.abs(np.abs(f) - f0) > 40])
        m = np.abs(np.abs(f) - f0) <= bw
        return float(g[m].max() - base) if m.any() else np.nan

    print("\n各法时间积分谱里几条特征线相对本底 (dB):")
    for nm, f, S in (("1 CAF+gate", freqs_s, S_caf),
                     ("2 specdiv+gate", freqs_s, S_sdv),
                     ("3 CAF nogate", f3, S_ng)):
        parts = [f"{nm:16s}"]
        for f0 in (100.0, 200.0):
            parts.append(f"{f0:.0f}Hz {excess_at(f, S, f0):+5.1f}")
        print("  " + "   ".join(parts))

    if show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
