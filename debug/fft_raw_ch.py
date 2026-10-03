#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 rt_iqdump 存的 npz 里取一段单通道原始 IQ，直接打一次 FFT。

不切 TDD 窗、不逐帧塌缩、不去 DC —— 就是把连续快时样本原样 FFT，
看这段基带里到底有什么频率成分（LO 偏、TDD 1kHz 门控线、杂散…）。

    /usr/bin/python3 debug/fft_raw_ch.py                    # debug/ 下最新 iq_*.npz，ch0，前 1s
    /usr/bin/python3 debug/fft_raw_ch.py iq_xxx.npz --ch 1 --sec 1 --start 2
    /usr/bin/python3 debug/fft_raw_ch.py --fmax 5000 --window hann --no-show
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def newest_iq(d):
    f = sorted(glob.glob(os.path.join(d, "iq_*.npz")))
    return f[-1] if f else None


def main() -> int:
    p = argparse.ArgumentParser(description="单通道原始 IQ 直接 FFT")
    p.add_argument("npz", nargs="?", default=None, help="iq_*.npz（省略=debug/下最新）")
    p.add_argument("--ch", type=int, default=0, help="通道 0/1（默认 0；ch0=RFA=参考）")
    p.add_argument("--mode", choices=["iq", "abs", "abs2"], default="iq",
                   help="iq=复基带直接FFT(默认) / abs=FFT(|x|) / abs2=FFT(|x|^2) 包络谱")
    p.add_argument("--sec", type=float, default=1.0, help="取多少秒（默认 1.0）")
    p.add_argument("--start", type=float, default=0.0, help="从第几秒开始（默认 0）")
    p.add_argument("--window", choices=["none", "hann", "blackman"], default="none",
                   help="加窗（默认 none，真·直接 FFT）")
    p.add_argument("--rm-dc", action="store_true", help="FFT 前减掉均值")
    p.add_argument("--fmax", type=float, default=None, help="只画 ±fmax Hz")
    p.add_argument("--out", type=str, default=None, help="PNG 输出路径")
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()

    path = a.npz or newest_iq(HERE)
    if not path:
        print("debug/ 下没有 iq_*.npz", file=sys.stderr)
        return 1
    if not os.path.isabs(path):
        path = os.path.join(HERE, path)

    show = (not a.no_show) and bool(os.environ.get("DISPLAY")) and sys.stdout.isatty()
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(path, mmap_mode="r")
    iq = d["iq"]                       # (2, N*2) int16 交织
    fs = float(d["fs"])
    total = iq.shape[1] // 2
    base = os.path.basename(path)

    s0 = int(round(a.start * fs))
    n = int(round(a.sec * fs))
    if s0 < 0 or s0 >= total:
        print(f"start={a.start}s 越界（总长 {total/fs:.3f}s）", file=sys.stderr)
        return 1
    n = min(n, total - s0)

    seg = np.asarray(iq[a.ch, s0 * 2:(s0 + n) * 2], dtype=np.float64)
    x = seg[0::2] + 1j * seg[1::2]     # 复基带
    n = len(x)

    print(f"=== {base}  ch{a.ch} ===")
    print(f"fs={fs/1e6:.3f}MHz  取 [{a.start:.3f}, {a.start + n/fs:.3f}]s  "
          f"= {n} 样本  →  频率分辨率 {fs/n:.3f} Hz")
    dc = x.mean()
    print(f"DC(均值) = {abs(dc):.1f}  ({20*np.log10(abs(dc)/np.abs(x).max() + 1e-20):+.1f} dBFS 相对峰值)")

    # 复基带 / 包络。abs、abs2 会把 TDD 门控的 1kHz 方波暴露成谱线；
    # 复基带直接 FFT 看不到，因为门控是幅度调制（时域乘实方波），
    # 宽带信号卷上 ±n·1kHz 的梳子后各副本重叠，谱形几乎不变。
    if a.mode == "abs":
        x = np.abs(x).astype(np.complex128)
    elif a.mode == "abs2":
        x = (np.abs(x) ** 2).astype(np.complex128)

    if a.rm_dc:
        x = x - x.mean()
    if a.window == "hann":
        x = x * np.hanning(n)
    elif a.window == "blackman":
        x = x * np.blackman(n)

    X = np.fft.fftshift(np.fft.fft(x))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    mag_db = 20.0 * np.log10(np.abs(X) / n + 1e-20)

    # 峰值（排除 ±2 个 bin 的 DC）
    guard = np.abs(freqs) > 2 * (fs / n)
    order = np.argsort(mag_db[guard])[::-1]
    fpos = freqs[guard][order]
    fdb = mag_db[guard][order]
    print("\n非 DC 前 8 个峰:")
    seen = []
    for f_, db_ in zip(fpos, fdb):
        if any(abs(f_ - s) < 5.0 for s in seen):
            continue
        seen.append(f_)
        print(f"  {f_:+12.1f} Hz   {db_:7.1f} dB")
        if len(seen) >= 8:
            break

    # 关注频点：TDD 门控 n×1000Hz + 一直在追的 200Hz 梳子（取邻近 ±3 bin 的最大值）
    df = fs / n
    noise = np.median(mag_db[np.abs(freqs) < 5000])
    print(f"\n关注频点（噪底中位 {noise:.1f} dB）:")
    for tgt in (200, 400, 600, 800, 1000, 2000, 3000):
        sel = np.abs(np.abs(freqs) - tgt) <= 3 * df
        pk = mag_db[sel].max()
        print(f"  {tgt:5d} Hz : {pk:7.1f} dB   ({pk - noise:+.1f} dB 高于噪底)")

    mask = np.ones(n, bool) if a.fmax is None else (np.abs(freqs) <= a.fmax)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(freqs[mask], mag_db[mask], lw=0.5, color="tab:blue")
    ax.axvline(0, color="k", ls="--", lw=0.7, alpha=0.5)
    fm = a.fmax or 3000
    for k in range(1, int(fm // 1000) + 1):
        for s in (-1, 1):
            ax.axvline(s * 1000 * k, color="tab:red", ls=":", lw=0.7, alpha=0.6)
    for s in (-1, 1):
        ax.axvline(s * 200, color="tab:green", ls=":", lw=0.9, alpha=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB, /N)")
    win_txt = "" if a.window == "none" else f" · {a.window}"
    dc_txt = " · dc removed" if a.rm_dc else ""
    ax.set_title(f"{base}  ch{a.ch}   raw FFT   "
                 f"[{a.start:.2f}–{a.start + n/fs:.2f}s, {n} pts]{win_txt}{dc_txt}")
    ax.grid(alpha=0.3)

    out = a.out or os.path.join(
        HERE, base.replace("iq_", f"fft_ch{a.ch}_").replace(".npz", ".png"))
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n🖼  {out}")

    if show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
