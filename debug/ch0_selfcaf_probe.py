#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch0 自己跟自己较劲——±50/100/150/200Hz 那把梳子到底哪来的？

对**一份原始 IQ 的 ch0（参考通道）单独**，把每个 TDD 帧塌成几种不同的标量，
每种都能暴露不同的物理量，看那把梳子出现在哪几种里：

  A) amp   = Σ|ch0|²                     纯幅度（AM）
  B) |coh| = |Σ ch0|²                    相干和的幅度（AM + 帧内相位是否一致）
  C) phase = angle(Σ ch0)                每帧载波相位（慢漂 + 任何相位调制 PM）
  D) cfo   = angle(Σ conj(ch0[:-1])·ch0[1:])   帧内平均频偏（载频在抖吗）
  E) sc(m) = Σ conj(ch0_k)·ch0_{k+m}     帧间自相关（复数，m=1/2/5/10 帧滞后）
                                          —— 这个是"ch0 自己 CAF 自己"最直接的版本
  F) 连续 |ch0(t)|² 抽到 ~2kHz 后的谱      不按 TDD 切，看包络里有没有 50/100/200

对每条慢时序列做 FFT（5s → 0.2Hz 分辨），**把梳齿的精确频率打出来**——
这是判断来源的关键：市电锁 50.00Hz 纹丝不动；通信帧时钟是发射机晶振说了算
（可能 49.98 / 100.02 之类）；机械振动会飘。

    /usr/bin/python3 debug/ch0_selfcaf_probe.py [iq_xxx.npz] [--no-show]
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FS_OUT = 1000.0          # TDD 帧率
COMB = (50.0, 100.0, 150.0, 200.0, 250.0)


def newest_iq(dirp):
    f = sorted(glob.glob(os.path.join(dirp, "iq_*.npz")))
    return f[-1] if f else None


def slow_fft(x, fs=FS_OUT, seg=1000, hop=500):
    """Welch 平均功率谱（1s 段、50% 重叠、Blackman），返回 (freqs, dB rel median)。

    ⚠️ 一定要做分段平均，不能一整段 FFT 再补零——补零只是插值，会把噪声尖峰
    抬成"看起来像谱线"的东西，`comb_peaks` 取 ±2Hz 最大值就会满屏假阳性
    （这个脚本第一版就踩了这个坑，读数全是 +7~9dB 的噪声毛刺）。
    """
    x = np.asarray(x, np.complex128)
    x = x - x.mean()
    w = np.blackman(seg)
    acc = None
    n = 0
    for i in range(0, len(x) - seg + 1, hop):
        X = np.fft.fftshift(np.fft.fft(x[i:i + seg] * w))
        acc = np.abs(X) ** 2 if acc is None else acc + np.abs(X) ** 2
        n += 1
    if acc is None:                       # 数据太短
        X = np.fft.fftshift(np.fft.fft(x * np.blackman(len(x))))
        acc = np.abs(X) ** 2
        n = 1
    fr = np.fft.fftshift(np.fft.fftfreq(seg if acc.size == seg else len(x), 1.0 / fs))
    mag = 10 * np.log10(acc / n + 1e-30)
    return fr, mag - np.median(mag)


def comb_peaks(fr, mag, lines=COMB, halfwidth=2.0):
    """在每条梳齿附近找峰：返回 {标称: (实测频率, 高出本底dB)}。"""
    base_m = np.ones(len(fr), bool)
    for f0 in lines:
        base_m &= np.abs(np.abs(fr) - f0) > 6.0
    base_m &= np.abs(fr) > 6.0
    floor = np.median(mag[base_m])
    out = {}
    for f0 in lines:
        m = np.abs(np.abs(fr) - f0) <= halfwidth
        if not m.any():
            out[f0] = (np.nan, np.nan)
            continue
        idx = np.flatnonzero(m)
        j = idx[np.argmax(mag[idx])]
        out[f0] = (abs(fr[j]), mag[j] - floor)
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
    base = os.path.basename(path)
    n_fr = (total - phase - n_int) // n_per
    starts = phase + np.arange(n_fr) * n_per
    print(f"=== {base} ===  {n_fr} 帧 @ {FS_OUT:.0f}Hz  n_int={n_int}({n_int/fs*1e6:.0f}µs)  "
          f"分辨率 {FS_OUT/n_fr:.2f}Hz")

    def frame(ch, s):
        seg = np.asarray(iq[ch, s * 2:(s + n_int) * 2], np.float64)
        return seg[0::2] + 1j * seg[1::2]

    # ---- 逐帧塌缩：ch0 的几种标量 ----
    print("逐帧提取 ch0 的 amp / |coh| / phase / cfo …")
    t0 = time.time()
    amp = np.empty(n_fr)
    coh = np.empty(n_fr, np.complex128)
    cfo = np.empty(n_fr)
    fsum = np.empty(n_fr, np.complex128)     # 帧信号（留着做帧间自相关）
    # 帧间自相关要原始帧，n_int*n_fr 太大——只留"帧的相干和"够做 m 帧滞后的
    # 相位关系；真正逐样本的帧间自相关另开一小批算（下面 E）。
    for k, s in enumerate(starts):
        x = frame(0, s)
        amp[k] = np.vdot(x, x).real / n_int
        sx = x.sum()
        coh[k] = sx / n_int
        fsum[k] = sx
        cfo[k] = np.angle(np.vdot(x[:-1], x[1:]))       # 帧内平均相位步进
    ph = np.unwrap(np.angle(coh))
    print(f"  用时 {time.time()-t0:.1f}s")

    # ---- E) 帧间自相关 sc(m)：Σ_t conj(ch0_k)·ch0_{k+m}，逐样本 ----
    # 内存：一次只持有两帧。m 帧滞后 -> k 和 k+m。
    print("逐帧算帧间自相关 sc(m), m=1,2,5,10 …")
    t0 = time.time()
    sc = {m: np.zeros(n_fr, np.complex128) for m in (1, 2, 5, 10)}
    prevs = {}
    buf = []
    for k, s in enumerate(starts):
        x = frame(0, s)
        nx = np.sqrt(np.vdot(x, x).real)
        buf.append((x, nx))
        if len(buf) > 11:
            buf.pop(0)
        for m in sc:
            if k - m >= 0:
                xm, nm = buf[-(m + 1)] if len(buf) > m else (None, None)
                if xm is not None:
                    sc[m][k] = np.vdot(xm, x) / (nm * nx)
    print(f"  用时 {time.time()-t0:.1f}s")

    # ---- F) 连续 |ch0(t)|² 抽到 ~2kHz（不按 TDD 切）----
    dec = int(round(fs / 2000.0))            # -> ~2kHz
    i0 = np.asarray(iq[0, 0:2 * total:2 * dec], np.float64)
    q0 = np.asarray(iq[0, 1:2 * total:2 * dec], np.float64)
    env = i0 * i0 + q0 * q0
    fs_env = fs / dec
    we = np.blackman(len(env))
    Ee = 20 * np.log10(np.abs(np.fft.rfft((env - env.mean()) * we)) + 1e-30)
    fre = np.fft.rfftfreq(len(env), 1.0 / fs_env)
    Ee = Ee - np.median(Ee)

    # ---- 汇总每种观测里的梳齿 ----
    series = [
        ("A amp   Sigma|ch0|^2", *slow_fft(amp)),
        ("B |coh|  |Sigma ch0|", *slow_fft(np.abs(coh))),
        ("C phase  angle(Sigma ch0)", *slow_fft(ph - np.polyval(np.polyfit(np.arange(n_fr), ph, 2), np.arange(n_fr)))),
        ("D cfo    in-frame freq", *slow_fft(cfo)),
        ("E sc(1)  frame-lag 1", *slow_fft(sc[1])),
        ("E sc(2)  frame-lag 2", *slow_fft(sc[2])),
        ("E sc(5)  frame-lag 5", *slow_fft(sc[5])),
        ("E sc(10) frame-lag 10", *slow_fft(sc[10])),
    ]
    print(f"\n{'观测量':28s}  " + "  ".join(f"{int(f)}Hz" for f in COMB)
          + "     （实测频率 / 高出本底dB）")
    for name, fr, mag in series:
        pk, fl = comb_peaks(fr, mag)
        cells = []
        for f0 in COMB:
            fm, ex = pk[f0]
            cells.append(f"{fm:5.1f}/{ex:+4.1f}" if np.isfinite(ex) else "  -  ")
        print(f"  {name:26s}  " + "  ".join(cells))
    pkE, flE = comb_peaks(fre, Ee)
    cells = [f"{pkE[f0][0]:5.1f}/{pkE[f0][1]:+4.1f}" if np.isfinite(pkE[f0][1]) else "  -  "
             for f0 in COMB]
    print(f"  {'F envelope 2kHz (no gate)':26s}  " + "  ".join(cells))

    # 精确基频：在 sc(1) 谱里 40–60Hz 找最强峰，抛物线插值
    fr1, m1 = slow_fft(sc[1])
    band = (fr1 > 40) & (fr1 < 60)
    if band.any():
        jj = np.flatnonzero(band)[np.argmax(m1[band])]
        y0, y1, y2 = m1[jj - 1], m1[jj], m1[jj + 1]
        dlt = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-30)
        f_fine = fr1[jj] + dlt * (fr1[1] - fr1[0])
        print(f"\n  >>> sc(1) 里 ~50Hz 峰的精确频率 = {f_fine:.3f} Hz")
        print(f"      （市电应锁在 50.000；偏了就是发射机/晶振时钟）")

    # ---- 出图 ----
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    axes = axes.ravel()
    plot_set = [
        ("A  amp  Sigma|ch0|^2", *slow_fft(amp)),
        ("B  |coh|  |Sigma ch0|", *slow_fft(np.abs(coh))),
        ("C  phase (detrended)", *slow_fft(ph - np.polyval(np.polyfit(np.arange(n_fr), ph, 2), np.arange(n_fr)))),
        ("D  in-frame CFO", *slow_fft(cfo)),
        ("E  sc(1) frame-lag 1", *slow_fft(sc[1])),
        ("E  sc(2) frame-lag 2", *slow_fft(sc[2])),
        ("E  sc(5) frame-lag 5", *slow_fft(sc[5])),
        ("E  sc(10) frame-lag 10", *slow_fft(sc[10])),
        ("F  |ch0(t)|^2 envelope, no gating", fre, Ee),
    ]
    for ax, (name, fr, mag) in zip(axes, plot_set):
        ax.plot(fr, mag, lw=0.7)
        for f0 in COMB:
            ax.axvline(f0, color="tab:red", lw=0.6, ls=":")
            ax.axvline(-f0, color="tab:red", lw=0.6, ls=":")
        ax.set_xlim(-a.fmax if np.any(fr < 0) else 0, a.fmax)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("slow-time freq (Hz)")
        ax.set_ylabel("dB rel median")
        ax.grid(alpha=0.3)
    fig.suptitle(f"ch0 self-probe — where does the 50/100/150/200 Hz comb live?  [{base}]",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, base.replace("iq_", "ch0probe_").replace(".npz", ".png"))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n🖼  {out}")

    np.savez(out.replace(".png", ".npz"), amp=amp, coh=coh.astype(np.complex64),
             ph=ph, cfo=cfo, sc1=sc[1].astype(np.complex64),
             sc2=sc[2].astype(np.complex64), sc5=sc[5].astype(np.complex64),
             sc10=sc[10].astype(np.complex64), env=env, fs_env=fs_env,
             fs=fs, phase=phase, n_int=n_int, src=base)
    if show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
