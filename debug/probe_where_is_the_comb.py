#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""那把 50/100/200Hz 梳子到底在"幅度"还是"相位"里，在 ch0 还是 ch1 还是两者之差里。

对一份原始 IQ，逐 TDD 帧塌成 6 个标量，每个只暴露一个物理量：

  P0   = Σ|ch0|²                        ch0 幅度（只有幅度，无相位）
  P1   = Σ|ch1|²                        ch1 幅度
  φ0   = angle(Σ ch0)  去趋势            ch0 载波相位
  φ1   = angle(Σ ch1)  去趋势            ch1 载波相位
  Δφ   = angle(Σ conj(ch0)·ch1)         两路相对相位（= cross-CAF 的相位）
  |ρ|  = |Σ conj(ch0)·ch1| / (‖ch0‖‖ch1‖)  相关系数幅度

关键逻辑：
- **自相关 conj(x)·x = |x|² 只有幅度**，任何"相位调制"在自相关里天然看不见。
  所以"ch0 自己 CAF 自己没事" ≠ "ch0 没问题"，只说明 ch0 的**幅度**没问题。
- B210 两路**共用 LO**，`conj(ch0)·ch1` 把共模 LO 相噪减掉了。cross 里还剩的
  50/100Hz 只可能是**某一路特有**的东西（路径/天线/前端级)。
- 梳子出现在 P1 里 = ch1 收到的功率被 100Hz 调幅；出现在 Δφ 里 = ch1 的
  路径相位被 100Hz 调（振动反射体 / 抖动的遮挡物）。

对每条慢时序列做 Welch 平均谱（1s 段 50% 重叠），打印 50/100/150/200Hz
相对本底多少 dB，并画 6 栏谱。

    /usr/bin/python3 debug/probe_where_is_the_comb.py [iq_xxx.npz] [--no-show]
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMB = (50.0, 100.0, 150.0, 200.0, 250.0)


def newest_iq(dirp):
    f = sorted(glob.glob(os.path.join(dirp, "iq_*.npz")))
    return f[-1] if f else None


def welch(x, fs=1000.0, seg=1000, hop=500):
    x = np.asarray(x, np.complex128) - np.mean(x)
    w = np.blackman(seg)
    acc, n = None, 0
    for i in range(0, len(x) - seg + 1, hop):
        X = np.fft.fftshift(np.fft.fft(x[i:i + seg] * w))
        acc = np.abs(X) ** 2 if acc is None else acc + np.abs(X) ** 2
        n += 1
    if acc is None:
        X = np.fft.fftshift(np.fft.fft(x * np.blackman(len(x))))
        acc, n = np.abs(X) ** 2, 1
        fr = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / fs))
    else:
        fr = np.fft.fftshift(np.fft.fftfreq(seg, 1.0 / fs))
    mag = 10 * np.log10(acc / n + 1e-30)
    return fr, mag - np.median(mag)


def comb_row(fr, mag, lines=COMB, hw=1.5):
    base = (np.abs(fr) > 12) & (np.abs(fr) < 245)
    for f0 in lines:
        base &= np.abs(np.abs(fr) - f0) > 6
    fl = np.median(mag[base])
    cells = []
    for f0 in lines:
        m = np.abs(np.abs(fr) - f0) <= hw
        cells.append(mag[m].max() - fl if m.any() else np.nan)
    return fl, cells


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="?", default=None)
    p.add_argument("--fmax", type=float, default=280.0)
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
    base = os.path.basename(path)
    n_fr = (total - phase - n_int) // n_per
    starts = phase + np.arange(n_fr) * n_per
    print(f"=== {base} ===  {n_fr} 帧 @ 1000Hz  n_int={n_int}({n_int/fs*1e6:.0f}µs)")

    def frame(ch, s):
        seg = np.asarray(iq[ch, s * 2:(s + n_int) * 2], np.float64)
        return seg[0::2] + 1j * seg[1::2]

    P0 = np.empty(n_fr); P1 = np.empty(n_fr)
    C0 = np.empty(n_fr, np.complex128); C1 = np.empty(n_fr, np.complex128)
    X01 = np.empty(n_fr, np.complex128)
    print("逐帧提取 …")
    t0 = time.time()
    for k, s in enumerate(starts):
        x = frame(0, s); y = frame(1, s)
        p0 = np.vdot(x, x).real; p1 = np.vdot(y, y).real
        P0[k], P1[k] = p0 / n_int, p1 / n_int
        C0[k], C1[k] = x.sum(), y.sum()
        X01[k] = np.vdot(x, y) / np.sqrt(p0 * p1)          # conj(x)·y 归一
    print(f"  用时 {time.time()-t0:.1f}s")
    print(f"  ch0 {10*np.log10(P0.mean()/32767**2):.1f} dBFS   "
          f"ch1 {10*np.log10(P1.mean()/32767**2):.1f} dBFS   "
          f"|ρ| 均值 {np.abs(X01).mean():.3f}")

    kk = np.arange(n_fr)
    def detrend(ph):
        return ph - np.polyval(np.polyfit(kk, ph, 3), kk)
    ph0 = detrend(np.unwrap(np.angle(C0)))
    ph1 = detrend(np.unwrap(np.angle(C1)))
    dphi = detrend(np.unwrap(np.angle(X01)))

    obs = [
        ("P0  = Sum|ch0|^2   (ch0 amplitude)", P0),
        ("P1  = Sum|ch1|^2   (ch1 amplitude)", P1),
        ("phi0 = angle(Sum ch0)  (ch0 carrier phase)", ph0),
        ("phi1 = angle(Sum ch1)  (ch1 carrier phase)", ph1),
        ("dphi = angle(conj ch0 . ch1)  (relative phase)", dphi),
        ("|rho| = |cross-CAF|  (correlation magnitude)", np.abs(X01)),
    ]

    print(f"\n{'观测量':46s}  " + " ".join(f"{int(f):>5d}" for f in COMB) + "   (dB 高出本底)")
    specs = []
    for name, x in obs:
        fr, mag = welch(x)
        specs.append((name, fr, mag))
        fl, cells = comb_row(fr, mag)
        print(f"  {name:44s}  "
              + " ".join(f"{c:>5.1f}" if np.isfinite(c) else "   - " for c in cells)
              + f"   [floor {fl:.0f}]")

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    for ax, (name, fr, mag) in zip(axes.ravel(), specs):
        ax.plot(fr, mag, lw=0.8)
        for f0 in COMB:
            ax.axvline(f0, color="tab:red", lw=0.5, ls=":")
            ax.axvline(-f0, color="tab:red", lw=0.5, ls=":")
        ax.set_xlim(-a.fmax, a.fmax)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("slow-time freq (Hz)"); ax.set_ylabel("dB rel median")
        ax.grid(alpha=0.3)
    fig.suptitle(f"where is the comb — amplitude vs phase, ch0 vs ch1 vs relative  [{base}]",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, base.replace("iq_", "combprobe_").replace(".npz", ".png"))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n🖼  {out}")
    np.savez(out.replace(".png", ".npz"), P0=P0, P1=P1, ph0=ph0, ph1=ph1,
             dphi=dphi, rho=np.abs(X01), X01=X01.astype(np.complex64),
             fs=fs, src=base)
    if show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
