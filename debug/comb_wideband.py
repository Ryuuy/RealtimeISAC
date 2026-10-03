#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把那把梳子往高频看——ch1 幅度的重复结构到多少 Hz 还在？400Hz 有没有？

两条路子，一份原始 IQ：

1) **逐帧（慢时）**：每个 TDD 帧一个 Σ|ch1|² / Σ|ch0|² / cross-CAF，帧率 1kHz
   → Nyquist ±500Hz。看 50/100/150/200/250/300/350/400/450Hz 有没有梳齿。
   （这是 sensing 链路真正的多普勒量程上限，1kHz 帧率的硬限制。）

2) **连续包络（快时）**：整段 |ch1(t)|² / |ch0(t)|² 抽到 ~8kHz，Welch 谱到 ~3kHz。
   TDD 门控本身是 1kHz 方波 → 必然有 n×1000Hz 的线；梳子（若在快时包络里）
   会在 DC 附近 + 每条 1kHz 线两侧当边带出现。能看到 2000Hz 以内的重复结构。

    /usr/bin/python3 debug/comb_wideband.py [iq_xxx.npz] [--no-show]
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def newest_iq(dirp):
    f = sorted(glob.glob(os.path.join(dirp, "iq_*.npz")))
    return f[-1] if f else None


def welch(x, fs, seg, hop=None):
    hop = hop or seg // 2
    x = np.asarray(x, np.complex128) - np.mean(x)
    w = np.blackman(seg)
    acc, n = None, 0
    for i in range(0, len(x) - seg + 1, hop):
        X = np.fft.rfft(x[i:i + seg].real * w) if not np.iscomplexobj(x) else \
            np.fft.fftshift(np.fft.fft(x[i:i + seg] * w))
        acc = np.abs(X) ** 2 if acc is None else acc + np.abs(X) ** 2
        n += 1
    P = 10 * np.log10(acc / max(n, 1) + 1e-30)
    if np.iscomplexobj(x):
        fr = np.fft.fftshift(np.fft.fftfreq(seg, 1.0 / fs))
    else:
        fr = np.fft.rfftfreq(seg, 1.0 / fs)
    return fr, P - np.median(P)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="?", default=None)
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
    print(f"=== {base} ===  {n_fr} 帧  n_int={n_int}({n_int/fs*1e6:.0f}µs)")

    # --- 1) 逐帧慢时，全 ±500Hz ---
    P0 = np.empty(n_fr); P1 = np.empty(n_fr); X01 = np.empty(n_fr, np.complex128)
    t0 = time.time()
    for k, s in enumerate(starts):
        g0 = np.asarray(iq[0, s * 2:(s + n_int) * 2], np.float64)
        g1 = np.asarray(iq[1, s * 2:(s + n_int) * 2], np.float64)
        x = g0[0::2] + 1j * g0[1::2]
        y = g1[0::2] + 1j * g1[1::2]
        p0 = np.vdot(x, x).real; p1 = np.vdot(y, y).real
        P0[k], P1[k] = p0, p1
        X01[k] = np.vdot(x, y) / np.sqrt(p0 * p1)
    print(f"  逐帧 {time.time()-t0:.1f}s")

    fr_s_p1, S_p1 = welch(P1.astype(complex), 1000.0, min(2000, n_fr // 2 * 2))
    fr_s_p0, S_p0 = welch(P0.astype(complex), 1000.0, min(2000, n_fr // 2 * 2))
    fr_s_x, S_x = welch(X01, 1000.0, min(2000, n_fr // 2 * 2))

    def report(fr, S, name, lines):
        base_m = np.abs(fr) > 12
        for f0 in lines:
            base_m &= np.abs(np.abs(fr) - f0) > 6
        fl = np.median(S[base_m])
        cells = []
        for f0 in lines:
            m = np.abs(np.abs(fr) - f0) <= 2
            cells.append(f"{f0:g}:{S[m].max()-fl:+.1f}" if m.any() else f"{f0:g}:-")
        print(f"  {name:22s} " + "  ".join(cells))

    L = [50, 100, 150, 200, 250, 300, 350, 400, 450]
    print("\n逐帧（慢时，±500Hz 上限）梳齿高出本底 dB:")
    report(fr_s_p1, S_p1, "P1 = Sum|ch1|^2", L)
    report(fr_s_p0, S_p0, "P0 = Sum|ch0|^2", L)
    report(fr_s_x, S_x, "cross-CAF", L)

    # --- 2) 连续包络，抽到 ~8kHz，看到 ~3kHz ---
    dec = int(round(fs / 8000.0))             # -> ~8kHz
    i1 = np.asarray(iq[1, 0:2 * total:2 * dec], np.float64)
    q1 = np.asarray(iq[1, 1:2 * total:2 * dec], np.float64)
    env1 = i1 * i1 + q1 * q1
    i0 = np.asarray(iq[0, 0:2 * total:2 * dec], np.float64)
    q0 = np.asarray(iq[0, 1:2 * total:2 * dec], np.float64)
    env0 = i0 * i0 + q0 * q0
    fs_env = fs / dec
    seg = int(fs_env * 1.0)                   # 1s 段 -> 1Hz 分辨
    fr_w1, W1 = welch(env1.astype(complex), fs_env, seg)
    fr_w0, W0 = welch(env0.astype(complex), fs_env, seg)
    print(f"\n连续包络: 抽到 {fs_env/1e3:.1f}kHz，Welch 1s 段（1Hz 分辨），Nyquist {fs_env/2/1e3:.2f}kHz")

    # --- 图 ---
    fig, ax = plt.subplots(3, 1, figsize=(15, 11))
    ax[0].plot(fr_s_p1, S_p1, lw=0.7, label="P1=Sum|ch1|^2", color="tab:red")
    ax[0].plot(fr_s_p0, S_p0, lw=0.7, label="P0=Sum|ch0|^2", color="tab:blue", alpha=0.6)
    for f0 in range(50, 501, 50):
        ax[0].axvline(f0, color="0.7", lw=0.5, ls=":"); ax[0].axvline(-f0, color="0.7", lw=0.5, ls=":")
    ax[0].set_xlim(-500, 500)
    ax[0].set_title("per-frame (slow-time), full +-500 Hz  — dotted = every 50 Hz")
    ax[0].set_xlabel("Doppler (Hz)"); ax[0].set_ylabel("dB rel median")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    m = (fr_w1 >= 0) & (fr_w1 <= 2500)
    ax[1].plot(fr_w1[m], W1[m], lw=0.6, color="tab:red", label="|ch1(t)|^2 envelope")
    for f0 in range(100, 2500, 100):
        ax[1].axvline(f0, color="0.85", lw=0.4, ls=":")
    for f0 in range(1000, 2500, 1000):
        ax[1].axvline(f0, color="tab:green", lw=0.8, ls="--")
    ax[1].set_xlim(0, 2500)
    ax[1].set_title("ch1 continuous envelope spectrum to 2.5 kHz  "
                    "(green dash = TDD 1 kHz harmonics; light dots = every 100 Hz)")
    ax[1].set_xlabel("Hz"); ax[1].set_ylabel("dB rel median")
    ax[1].grid(alpha=0.3)

    m2 = (fr_w1 >= 0) & (fr_w1 <= 600)
    ax[2].plot(fr_w1[m2], W1[m2], lw=0.8, color="tab:red", label="ch1 env")
    ax[2].plot(fr_w0[m2], W0[m2], lw=0.8, color="tab:blue", alpha=0.6, label="ch0 env")
    for f0 in range(50, 601, 50):
        ax[2].axvline(f0, color="0.7", lw=0.5, ls=":")
    ax[2].set_xlim(0, 600)
    ax[2].set_title("zoom 0-600 Hz of the continuous envelope (independent of TDD framing)")
    ax[2].set_xlabel("Hz"); ax[2].set_ylabel("dB rel median")
    ax[2].legend(); ax[2].grid(alpha=0.3)

    fig.suptitle(f"comb wideband — how far up does the ch1 repeating structure go?  [{base}]")
    fig.tight_layout()
    out = os.path.join(HERE, base.replace("iq_", "combwide_").replace(".npz", ".png"))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n🖼  {out}")

    # 连续包络里 100Hz 整数倍逐个报
    base_m = (fr_w1 > 20) & (fr_w1 < 950)
    for k in range(1, 10):
        base_m &= np.abs(fr_w1 - k * 100) > 8
    fl = np.median(W1[base_m])
    print("ch1 连续包络 100Hz 整数倍相对本底 dB:")
    for k in range(1, 25):
        f0 = k * 100
        mm = np.abs(fr_w1 - f0) <= 3
        if mm.any():
            print(f"   {f0:5d}Hz : {W1[mm].max()-fl:+5.1f}"
                  + ("   <- TDD" if f0 % 1000 == 0 else ""))
    if show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
