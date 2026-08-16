#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比「不去 DC 直接 Blackman+FFT」vs「先用历史 DC 均值去掉再 Blackman+FFT」。

用 rt_iqdump.py 存的原始 IQ 现算，不依赖 rt_main --debug 的录制（那份录制里
DC 已经在 rt_dsp.process() 里被扣掉了，没法比较"去之前"）。

DC 估计和 realtime/rt_dsp.py 改过之后的版本完全一致：**严格历史序**——
用上一步留下的 EMA 估计去减本帧，本帧的均值只用来刷新给下一帧用，
不会有本帧值污染自己估计的问题。alpha 预热同款（max(alpha, 1/n)）。

    /usr/bin/python3 debug/dc_removal_compare.py                  # 最新一份 iq_*.npz
    /usr/bin/python3 debug/dc_removal_compare.py iq_xxx.npz --fmax 300
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SC16_SCALE = 32767.0


def newest_iq(d: str) -> str | None:
    files = sorted(glob.glob(os.path.join(d, "iq_*.npz")))
    return files[-1] if files else None


def frame_channel(iq: np.ndarray, phase: int, n_int: int, n_period: int) -> np.ndarray:
    """严格复现 rt_dsp 的逐帧积分，返回**未去 DC**的 complex64 序列。"""
    total = iq.shape[1] // 2
    n_frames = (total - phase - n_int) // n_period
    out = np.empty(max(n_frames, 0), dtype=np.complex64)
    scale = SC16_SCALE * SC16_SCALE * n_int
    for k in range(n_frames):
        s = phase + k * n_period
        a = iq[0, s * 2:(s + n_int) * 2].reshape(n_int, 2)
        b = iq[1, s * 2:(s + n_int) * 2].reshape(n_int, 2)
        ar, ai = a[:, 0], a[:, 1]
        br, bi = b[:, 0], b[:, 1]
        re = (np.einsum("i,i->", ar, br, dtype=np.int64)
              + np.einsum("i,i->", ai, bi, dtype=np.int64))
        im = (np.einsum("i,i->", ar, bi, dtype=np.int64)
              - np.einsum("i,i->", ai, br, dtype=np.int64))
        out[k] = np.complex64(re / scale + 1j * (im / scale))
    return out


def historical_dc_remove(h: np.ndarray, alpha: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """逐帧严格历史 EMA 去 DC，和 rt_dsp.process() 改过之后的算法一致。

    第 k 帧只用第 0..k-1 帧估计出的 DC 去减，估计再拿第 k 帧刷新给第 k+1 帧用。
    返回 (去 DC 后的序列, 每帧的 DC 估计——用来画"DC 到底有多强/在不在动")。
    """
    out = np.empty_like(h)
    dc_track = np.empty(len(h), dtype=np.complex64)
    dc = np.complex64(0)
    for k, v in enumerate(h):
        dc_track[k] = dc
        out[k] = v - dc
        n = k + 1
        a = max(alpha, 1.0 / n)
        dc = np.complex64((1.0 - a) * dc + a * v)
    return out, dc_track


def spectrogram(x: np.ndarray, n_ring: int, n_step: int, nfft: int, fs_out: float):
    """滑窗 Blackman+FFT，和 rt_dsp 的 ring/win/fftbuf 同一套参数。"""
    win = np.blackman(n_ring).astype(np.float32)
    n_frames = len(x)
    starts = list(range(0, n_frames - n_ring + 1, n_step))
    spec = np.empty((len(starts), nfft), dtype=np.float32)
    t = np.empty(len(starts))
    for i, s in enumerate(starts):
        buf = np.zeros(nfft, dtype=np.complex64)
        buf[:n_ring] = x[s:s + n_ring] * win
        S = np.fft.fftshift(np.fft.fft(buf))
        spec[i] = (S.real * S.real + S.imag * S.imag)
        t[i] = (s + n_ring) / fs_out
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs_out))
    return spec, t, freqs


def db(x):
    return 10.0 * np.log10(np.asarray(x, np.float64) + 1e-20)


def line_level_db(spec: np.ndarray, freqs: np.ndarray, target: float, tol_bins: int = 1) -> float:
    """spec 是 (n_time, n_freq) 线性功率；返回 target Hz 附近相对本帧中位数功率的 dB。"""
    i = int(np.argmin(np.abs(freqs - target)))
    j = slice(max(0, i - tol_bins), i + tol_bins + 1)
    peak = spec[:, j].max(axis=1)
    med = np.median(spec, axis=1)
    return float(np.median(db(peak) - db(med)))


def main() -> int:
    p = argparse.ArgumentParser(description="DC 去除前后对比：直接 Blackman+FFT vs 历史DC均值去除后再 Blackman+FFT")
    p.add_argument("npz", nargs="?", default=None)
    p.add_argument("--window-sec", type=float, default=0.2)
    p.add_argument("--step-sec", type=float, default=0.02)
    p.add_argument("--nfft", type=int, default=256)
    p.add_argument("--alpha", type=float, default=0.01, help="DC EMA 系数，默认和 rt_config 一致")
    p.add_argument("--fmax", type=float, default=None, help="只画 ±fmax Hz")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()

    path = a.npz or newest_iq(HERE)
    if not path or not os.path.exists(path):
        print("没有 iq_*.npz，先跑 realtime/rt_iqdump.py 抓一份原始IQ。", file=sys.stderr)
        return 1
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(HERE, path)

    with np.load(path) as d:
        iq, fs = d["iq"], float(d["fs"])
        phase, n_int, n_period = int(d["phase"]), int(d["n_int"]), int(d["n_period"])
        locked = bool(d["locked"]) if "locked" in d.files else None
        contrast = float(d["contrast"]) if "contrast" in d.files else None

    fs_out = fs / n_period
    n_ring = int(round(a.window_sec * fs_out))
    n_step = max(1, int(round(a.step_sec * fs_out)))

    h_raw = frame_channel(iq, phase, n_int, n_period)
    h_dc, dc_track = historical_dc_remove(h_raw, a.alpha)

    base = os.path.basename(path)
    print(f"=== {base} ===")
    print(f"{len(h_raw)} 帧  fs_out={fs_out:.1f}Hz  window={a.window_sec*1e3:.0f}ms(n_ring={n_ring})  "
          f"step={a.step_sec*1e3:.0f}ms(n_step={n_step})  nfft={a.nfft}")
    if locked is not None:
        print(f"TDD锁定={locked}  对比度={contrast:.4f}" if contrast is not None else f"TDD锁定={locked}")
    print(f"原始 |h| 均值(≈DC幅度) = {np.abs(h_raw).mean():.4e}   "
          f"去DC后 |h-dc| 均值(应≈信号本身幅度) = {np.abs(h_dc).mean():.4e}")

    if len(h_raw) < n_ring + n_step:
        print("数据太短，凑不出一整窗", file=sys.stderr)
        return 2

    spec_raw, t, freqs = spectrogram(h_raw, n_ring, n_step, a.nfft, fs_out)
    spec_dc, _, _ = spectrogram(h_dc, n_ring, n_step, a.nfft, fs_out)

    dc_i = int(np.argmin(np.abs(freqs)))
    print(f"\nDC bin 功率：去DC前 {db(np.median(spec_raw[:, dc_i])):.1f}dB  "
          f"去DC后 {db(np.median(spec_dc[:, dc_i])):.1f}dB  "
          f"（下降 {db(np.median(spec_raw[:, dc_i])) - db(np.median(spec_dc[:, dc_i])):.1f}dB）")
    for f in (100.0, 200.0, -100.0, -200.0):
        print(f"{f:+.0f}Hz 相对本帧中位数：去DC前 {line_level_db(spec_raw, freqs, f):+.1f}dB  "
              f"去DC后 {line_level_db(spec_dc, freqs, f):+.1f}dB")

    show = (not a.no_show) and bool(os.environ.get("DISPLAY")) and sys.stdout.isatty()
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fmask = np.ones(len(freqs), dtype=bool) if a.fmax is None else (np.abs(freqs) <= a.fmax)
    f_axis = freqs[fmask]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    for ax, spec, title in ((axes[0], spec_raw, "Before DC removal (raw -> Blackman -> FFT)"),
                            (axes[1], spec_dc, "After historical-EMA DC removal -> Blackman -> FFT")):
        S_db = db(spec[:, fmask]).T
        vmin, vmax = np.percentile(S_db, 5), np.percentile(S_db, 95)
        mesh = ax.pcolormesh(t, f_axis, S_db, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_ylabel("Doppler (Hz)")
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, label="Power (dB)")
        for fl in (100, -100, 200, -200):
            ax.axhline(fl, color="w", ls=":", lw=0.6, alpha=0.5)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"DC removal before Blackman — {base}", fontsize=12)
    fig.tight_layout()

    stamp = os.path.splitext(base)[0].removeprefix("iq_")
    out = a.out or os.path.join(HERE, f"dc_compare_{stamp}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\n🖼  {out}")
    if show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
