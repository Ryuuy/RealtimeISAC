#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""频谱除法 vs CAF 的多普勒对比实验（同一份原始 IQ，同样的切窗、同样的采样率）。

只有「每个 TDD 帧塌缩成一个复数」这一步不同，其余全部共用：
  · 切窗    : 用采集时存下的 phase / n_int / n_period，两法逐帧完全一样的样本区间
  · 采样率  : npz 里的 fs（本机 10 MHz），帧率 = 1/tdd_period = 1000 Hz
  · 后端    : 同一套滑动 STFT（ring=200 帧=0.2s，hop=20 帧=0.02s，Blackman，
              256 点 FFT 补零，同一套 EMA 去 DC，同样的 5/95 分位色标）

两种「帧 -> 复数」的算法：
  CAF      : dec = Σ conj(ch0)·ch1 / (‖ch0‖·‖ch1‖)   —— RealtimeISAC 现在在用的
             （rt_dsp.DopplerEngine.process，含 2026-08-16 的相关系数归一化）
  频谱除法 : H(f) = FFT(ch1 - mean) / FFT(ch0 - mean)，取 median(Re)+j·median(Im)
             —— 老 Udoppler_spectrum_division_20MHz.py 的 LimeSDR 做法

用法：
    /usr/bin/python3 debug/compare_caf_vs_specdiv.py                 # debug/ 下最新 iq_*.npz
    /usr/bin/python3 debug/compare_caf_vs_specdiv.py iq_xxx.npz
    /usr/bin/python3 debug/compare_caf_vs_specdiv.py --fmax 300 --no-show

原始 IQ 先用 realtime/rt_iqdump.py 采：
    cd RealtimeISAC/realtime && /usr/bin/python3 rt_iqdump.py --ms 5000
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# 后端 STFT 参数：和 rt_config.RtConfig 的默认值一致
N_RING = 200          # window_sec 0.2s * fs_out 1000Hz
HOP = 20              # step_sec 0.02s * fs_out 1000Hz
NFFT = 256
FS_OUT = 1000.0
DC_ALPHA = 0.01
DC_GUARD_HZ = 20.0


def newest_iq(d):
    f = sorted(glob.glob(os.path.join(d, "iq_*.npz")))
    return f[-1] if f else None


def frame_bounds(phase, n_int, n_per, total):
    """逐 TDD 帧的样本区间 [s, s+n_int)。两种方法用的是同一批区间。"""
    n_fr = (total - phase - n_int) // n_per
    s = phase + np.arange(n_fr) * n_per
    return s, n_int, n_fr


def ema_dc_remove(z, hop=HOP, alpha=DC_ALPHA):
    """按 hop 粒度做 EMA 去 DC，严格历史序 + alpha 预热。

    复刻 rt_dsp.process 的 DC 抑制：用上一步留下的估计去减本步，再拿本步均值刷新。
    两条流用同一套处理，比较才公平。
    """
    z = z.astype(np.complex128).copy()
    dc = 0.0 + 0.0j
    dc_n = 0
    for i0 in range(0, len(z), hop):
        blk = slice(i0, i0 + hop)
        m = z[blk].mean()
        z[blk] -= dc
        dc_n += 1
        a = max(alpha, 1.0 / dc_n)
        dc = (1.0 - a) * dc + a * m
    return z


def stft(z, n_ring=N_RING, hop=HOP, nfft=NFFT):
    """共用后端：滑动 Blackman STFT，256 点补零，返回线性功率 |S|^2。"""
    win = np.blackman(n_ring)
    buf = np.zeros(nfft, dtype=np.complex128)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / FS_OUT))
    specs, times = [], []
    for c in range(n_ring, len(z) + 1, hop):
        buf[:n_ring] = z[c - n_ring:c] * win
        buf[n_ring:] = 0.0
        S = np.fft.fftshift(np.fft.fft(buf))
        specs.append((S.real ** 2 + S.imag ** 2))
        times.append(c / FS_OUT)
    return np.asarray(times), freqs, np.asarray(specs)   # spec: (n_time, n_freq)


def main() -> int:
    p = argparse.ArgumentParser(description="频谱除法 vs CAF 多普勒对比")
    p.add_argument("npz", nargs="?", default=None, help="iq_*.npz（省略=debug/下最新）")
    p.add_argument("--fmax", type=float, default=None, help="只画 ±fmax Hz")
    p.add_argument("--out", type=str, default=None, help="PNG 输出路径")
    p.add_argument("--no-show", action="store_true", help="只存图不弹窗")
    a = p.parse_args()

    path = a.npz or newest_iq(HERE)
    if not path:
        print("debug/ 下没有 iq_*.npz。先采：\n"
              "  cd RealtimeISAC/realtime && /usr/bin/python3 rt_iqdump.py --ms 5000",
              file=sys.stderr)
        return 1
    if not os.path.isabs(path):
        path = os.path.join(HERE, path)

    show = (not a.no_show) and bool(os.environ.get("DISPLAY")) and sys.stdout.isatty()
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # mmap 加载：iq 有几百 MB，这台机器内存小，别整块进内存
    d = np.load(path, mmap_mode="r")
    iq = d["iq"]                                  # (2, N*2) int16, on disk
    fs = float(d["fs"])
    phase, n_int, n_per = int(d["phase"]), int(d["n_int"]), int(d["n_period"])
    locked, contrast = bool(d["locked"]), float(d["contrast"])
    total = iq.shape[1] // 2

    base = os.path.basename(path)
    print(f"=== {base} ===")
    print(f"{total/fs*1e3:.0f} ms x2ch  fs={fs/1e6:.0f}MHz  "
          f"TDD phase={phase} n_int={n_int}({n_int/fs*1e6:.0f}µs) n_period={n_per}  "
          f"{'锁定' if locked else '未锁定'} contrast={contrast:.2f}")
    if not locked:
        print("⚠️ TDD 未锁定，切窗位置无意义，结果不可信", file=sys.stderr)

    # 整条通道转 complex 太大（10MHz*5s*16B*2ch = 1.6GB），逐帧从 mmap 上切小段转
    starts, n_int, n_fr = frame_bounds(phase, n_int, n_per, total)
    print(f"逐 TDD 帧: {n_fr} 帧 @ {FS_OUT:.0f}Hz（{n_fr/FS_OUT:.1f}s 有效流）")

    def load_ch(ch, s):
        seg = np.asarray(iq[ch, s * 2:(s + n_int) * 2], dtype=np.float64)
        return seg[0::2] + 1j * seg[1::2]

    print("逐帧算 CAF + 频谱除法 …")
    t0 = time.time()
    caf = np.empty(n_fr, dtype=np.complex128)
    sdv = np.empty(n_fr, dtype=np.complex128)
    eps = 1e-12
    for k, s in enumerate(starts):
        a0 = load_ch(0, s)
        b0 = load_ch(1, s)
        # --- CAF（相关系数归一化，复刻 rt_dsp）---
        cc = np.vdot(a0, b0)
        nrm = np.sqrt(np.vdot(a0, a0).real) * np.sqrt(np.vdot(b0, b0).real)
        caf[k] = cc / nrm if nrm > 0 else 0.0
        # --- 频谱除法（复刻老脚本）---
        tx = a0 - a0.mean()
        rx = b0 - b0.mean()
        TX = np.fft.fftshift(np.fft.fft(tx))
        RX = np.fft.fftshift(np.fft.fft(rx))
        # 老脚本用 TX + eps*sign(TX) 防除零；sign(0)=0 挡不住真的 0 bin，这里
        # 对 |TX| 过小的 bin 直接抬到一个下限（等价于老脚本的 1e-6 兜底分支）
        floor = max(np.abs(TX).max() * 1e-9, 1e-12)
        TX_safe = np.where(np.abs(TX) < floor, floor, TX)
        H = RX / TX_safe
        sdv[k] = np.median(H.real) + 1j * np.median(H.imag)
        if (k + 1) % 1000 == 0:
            print(f"  {k+1}/{n_fr}")
    print(f"  用时 {time.time()-t0:.1f}s")

    # 共用后端：同一套 EMA 去 DC + 同一套滑动 STFT
    t_caf, freqs, S_caf = stft(ema_dc_remove(caf))
    t_sdv, _, S_sdv = stft(ema_dc_remove(sdv))

    fmask = np.ones(len(freqs), bool) if a.fmax is None else (np.abs(freqs) <= a.fmax)
    f_axis = freqs[fmask]

    def to_db(S):
        return 10.0 * np.log10(S[:, fmask].T + 1e-20)      # (n_freq, n_time)

    Dcaf, Dsdv = to_db(S_caf), to_db(S_sdv)

    # 时间积分的全局多普勒谱（形状对比，各自归一化到峰值）
    g_caf = 10 * np.log10(S_caf[:, fmask].mean(axis=0) + 1e-20)
    g_sdv = 10 * np.log10(S_sdv[:, fmask].mean(axis=0) + 1e-20)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9),
                             gridspec_kw={"width_ratios": [3, 1]})
    for row, (name, D, tt, g) in enumerate([
            ("CAF  dec = sum conj(ch0)*ch1 / (||ch0||*||ch1||)", Dcaf, t_caf, g_caf),
            ("Spectrum division  H = FFT(ch1)/FFT(ch0), median(Re)+j median(Im)",
             Dsdv, t_sdv, g_sdv)]):
        ax = axes[row, 0]
        vmin, vmax = np.percentile(D, 5), np.percentile(D, 95)
        mesh = ax.pcolormesh(tt, f_axis, D, shading="auto", cmap="jet",
                             vmin=vmin, vmax=vmax)
        ax.set_ylabel("Doppler (Hz)")
        ax.set_title(f"{name}   [{base}]")
        fig.colorbar(mesh, ax=ax, label="Relative power (dB)")
        for sgn in (-DC_GUARD_HZ, DC_GUARD_HZ):
            ax.axhline(sgn, color="w", ls="--", lw=0.7, alpha=0.5)

        axg = axes[row, 1]
        axg.plot(g - g.max(), f_axis, lw=1.0, color="tab:blue")
        axg.set_title("time-integrated (norm)")
        axg.set_xlabel("dB")
        axg.grid(alpha=0.3)
        axg.set_ylim(f_axis[0], f_axis[-1])
    axes[1, 0].set_xlabel("Time (s)")

    out = a.out or os.path.join(HERE, base.replace("iq_", "cmp_caf_specdiv_")
                                .replace(".npz", ".png"))
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"🖼  {out}")

    npz_out = out.replace(".png", ".npz")
    np.savez(npz_out, t_caf=t_caf, t_sdv=t_sdv, freqs=freqs,
             S_caf=S_caf.astype(np.float32), S_sdv=S_sdv.astype(np.float32),
             caf_stream=caf.astype(np.complex64), sdv_stream=sdv.astype(np.complex64),
             fs=fs, phase=phase, n_int=n_int, n_period=n_per,
             n_ring=N_RING, hop=HOP, nfft=NFFT, fs_out=FS_OUT, src=base)
    print(f"💾 {npz_out}")

    # 数字对比
    def peak_track(S):
        band = np.abs(freqs) >= DC_GUARD_HZ
        idx = np.where(band, S, -np.inf).argmax(axis=1)
        return freqs[idx]
    pk_caf, pk_sdv = peak_track(S_caf), peak_track(S_sdv)
    print(f"\n峰值多普勒中位数   CAF {np.median(pk_caf):+.1f}Hz   "
          f"频谱除法 {np.median(pk_sdv):+.1f}Hz")
    print(f"两法峰值轨迹相关   {np.corrcoef(pk_caf[:min(len(pk_caf),len(pk_sdv))], pk_sdv[:min(len(pk_caf),len(pk_sdv))])[0,1]:+.3f}")

    if show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
