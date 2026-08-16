#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 rt_iqdump.py 存下的原始 IQ 上，逐帧量突发的**边沿位置和长度**。

起因：一直在找幅度调制，但如果变的是**突发的时间**（起点漂 / 长度伸缩），
而积分窗是固定的，窗边缘就会周期性切进切出突发 —— 同样产生周期性调制，
而且帧结构类的东西天然极稳定，正好解释"为什么那条线这么稳"。

    /usr/bin/python3 debug/analyze_iq.py                  # 最新一份 iq_*.npz
    /usr/bin/python3 debug/analyze_iq.py iq_xxx.npz

做三件事：
1. 逐帧量 ch0 突发的起点/终点/长度，扣掉线性时钟漂移后看抖动有多大、
   有没有 5ms(200Hz) / 10ms(100Hz) 的周期
2. 量真实的 TDD 周期（用起点序列的斜率），看它到底是不是 1.000ms
3. **窗宽实验**：同一份数据，窗只切一个槽 vs 横跨两个槽，
   看 200Hz 会不会被"横跨"造出来 —— 这是对那个假设的直接检验
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def newest(d):
    f = sorted(glob.glob(os.path.join(d, "iq_*.npz")))
    return f[-1] if f else None


def db(x):
    return 10.0 * np.log10(np.asarray(x, float) + 1e-30)


def envelope_1us(iq, ch, fs):
    """(N*2,) int16 -> 每 1µs 一格的平均功率（归一化）。"""
    dec = max(1, int(round(fs * 1e-6)))
    i = iq[ch, 0::2].astype(np.int64)
    q = iq[ch, 1::2].astype(np.int64)
    p = i * i + q * q
    n = (p.size // dec) * dec
    return p[:n].reshape(-1, dec).mean(axis=1) / (32767.0 ** 2)


def line_level(x, fs, targets=(100.0, 200.0)):
    """去均值后各目标频点的幅度，以及相对局部本底高多少 dB。

    输入可以是实数（时间量序列）或复数（CAF 流）。复数必须走**双边**谱：
    强行取实部会把 +f 和 -f 折在一起，多普勒的正负号就丢了。
    """
    y = np.asarray(x)
    y = y - y.mean()
    w = np.blackman(len(y))
    if np.iscomplexobj(y):
        F = np.abs(np.fft.fftshift(np.fft.fft(y * w)))
        fr = np.fft.fftshift(np.fft.fftfreq(len(y), 1.0 / fs))
    else:
        F = np.abs(np.fft.rfft(y * w))
        fr = np.fft.rfftfreq(len(y), 1.0 / fs)
    S = db(F ** 2)
    k = 31
    from numpy.lib.stride_tricks import sliding_window_view
    base = np.median(sliding_window_view(np.pad(S, (k // 2, k // 2), mode="edge"), k),
                     axis=-1)
    out = []
    for tg in targets:
        cand = [tg, -tg] if np.iscomplexobj(y) else [tg]
        best = -np.inf
        for t in cand:                       # 复数谱要 +f/-f 都看
            i = int(np.argmin(np.abs(fr - t)))
            j = slice(max(0, i - 1), i + 2)
            best = max(best, float((S[j] - base[j]).max()))
        out.append(best)
    return out


def caf_stream(iq, phase, n_int, n_per, n_fr, total):
    out = np.empty(n_fr, dtype=np.complex128)
    for k in range(n_fr):
        s = phase + k * n_per
        if s + n_int > total:
            out[k] = out[k - 1] if k else 0
            continue
        a = iq[0, s * 2:(s + n_int) * 2].astype(np.float64).reshape(-1, 2)
        b = iq[1, s * 2:(s + n_int) * 2].astype(np.float64).reshape(-1, 2)
        re = (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum()
        im = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()
        out[k] = (re + 1j * im) / n_int
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="逐帧量突发边沿/长度")
    p.add_argument("npz", nargs="?", default=None)
    a = p.parse_args()
    path = a.npz or newest(HERE)
    if not path:
        print("没有 iq_*.npz，先跑 rt_iqdump.py", file=sys.stderr)
        return 1
    if not os.path.isabs(path):
        path = os.path.join(HERE, path)

    d = np.load(path)
    iq, fs = d["iq"], float(d["fs"])
    phase, n_int, n_per = int(d["phase"]), int(d["n_int"]), int(d["n_period"])
    total = iq.shape[1] // 2
    fs_out = fs / n_per
    print(f"=== {os.path.basename(path)} ===")
    print(f"{total/fs*1e3:.0f}ms @ {fs/1e6:.0f}MHz  录制时的窗=[{phase},{phase+n_int}) "
          f"({n_int/fs*1e6:.0f}µs)  帧率 {fs_out:.1f}Hz")

    e0 = envelope_1us(iq, 0, fs)          # 1µs 一格
    per_us = int(round(n_per / fs * 1e6))
    n_fr = len(e0) // per_us - 1

    # 阈值：全局 p10/p90 的几何中点（和 rt_sync 同一套判据）
    thr = np.sqrt(np.percentile(e0, 10) * np.percentile(e0, 90))
    print(f"边沿判据阈值 = {db(thr):.1f} dBFS  "
          f"(p10={db(np.percentile(e0,10)):.1f}, p90={db(np.percentile(e0,90)):.1f})")

    # ---- 逐帧量 ch0 那个槽的起点/终点 ----
    # 以录制时的窗中心为锚，在 ±per_us/2 内找该帧的上升沿和下降沿
    anchor = (phase + n_int // 2) // int(round(fs / 1e6))    # µs
    starts, ends = [], []
    for k in range(n_fr):
        c = anchor + k * per_us
        lo, hi = c - per_us // 2, c + per_us // 2
        if lo < 0 or hi > len(e0):
            continue
        seg = e0[lo:hi] >= thr
        if not seg.any() or seg.all():
            continue
        # 中心必须在高电平里，否则这一帧没对上
        mid = len(seg) // 2
        if not seg[mid]:
            continue
        s = mid
        while s > 0 and seg[s - 1]:
            s -= 1
        t = mid
        while t < len(seg) - 1 and seg[t + 1]:
            t += 1
        starts.append(lo + s)
        ends.append(lo + t + 1)
    starts = np.array(starts, float)
    ends = np.array(ends, float)
    if len(starts) < 20:
        print("对上的帧太少，测不了", file=sys.stderr)
        return 1
    lens = ends - starts
    idx = np.arange(len(starts))
    print(f"\n--- 逐帧突发边沿（{len(starts)} 帧对上）---")

    # 起点里含"每帧 +per_us"的固定步进 + 时钟漂移，扣掉线性趋势看真实抖动
    k_fit = np.polyfit(idx, starts, 1)
    resid = starts - np.polyval(k_fit, idx)
    real_per_us = k_fit[0]
    ppm = (real_per_us - per_us) / per_us * 1e6
    print(f"  真实 TDD 周期 = {real_per_us:.4f} µs "
          f"(我们假设 {per_us} µs, 差 {ppm:+.1f} ppm)")
    print(f"  起点抖动(扣线性趋势后): std {resid.std():.2f}µs  "
          f"极差 {resid.max()-resid.min():.1f}µs")
    print(f"  突发长度: 均值 {lens.mean():.1f}µs  std {lens.std():.2f}µs  "
          f"极差 {lens.max()-lens.min():.1f}µs  "
          f"[{lens.min():.0f}, {lens.max():.0f}]")

    print(f"\n--- 这些时间量里有没有 5ms/10ms 周期 ---")
    print(f"  {'量':<16} {'100Hz':>8} {'200Hz':>8}  (高出本底 dB)")
    for nm, seq in (("起点抖动", resid), ("突发长度", lens),
                    ("终点抖动", ends - np.polyval(np.polyfit(idx, ends, 1), idx))):
        a100, a200 = line_level(seq, fs_out)
        print(f"  {nm:<16} {a100:>+8.1f} {a200:>+8.1f}")

    # 折叠看形状
    for period in (5, 10):
        m = (len(lens) // period) * period
        g = lens[:m].reshape(-1, period)
        mu, sem = g.mean(axis=0), g.std(axis=0, ddof=1) / np.sqrt(g.shape[0])
        snr = (mu.max() - mu.min()) / max(sem.mean(), 1e-12)
        print(f"  长度按 {period}帧 折叠(µs, 相对均值): "
              + " ".join(f"{x-mu.mean():+.2f}" for x in mu)
              + f"   峰谷={mu.max()-mu.min():.2f}µs = {snr:.1f}x标准误"
              + (" <-有结构" if snr > 3 else " <-噪声"))

    # ---- 窗宽实验：横跨两个槽会不会造出 200Hz ----
    print(f"\n--- 窗宽实验（同一份数据，看'横跨两槽'是不是 200Hz 的来源）---")
    us2s = int(round(fs / 1e6))
    print(f"  {'窗':<28} {'100Hz':>8} {'200Hz':>8}  (CAF 谱里高出本底 dB)")
    base_start = int(np.median(starts)) * us2s
    med_len = int(np.median(lens)) * us2s
    trials = [("只切槽B (实测长度)", base_start + 10 * us2s, med_len - 20 * us2s),
              ("只切槽B 再内缩50µs", base_start + 50 * us2s, med_len - 100 * us2s),
              ("槽B + 前面100µs", base_start - 100 * us2s, med_len + 100 * us2s),
              ("槽B + 前面300µs", base_start - 300 * us2s, med_len + 300 * us2s),
              ("槽B + 前面500µs(横跨)", base_start - 500 * us2s, med_len + 500 * us2s),
              ("整个周期", base_start, n_per)]
    for nm, st, ln in trials:
        st = st % n_per
        ln = max(1, min(ln, n_per))
        nf = (total - st - ln) // n_per
        x = caf_stream(iq, st, ln, n_per, nf, total)
        a100, a200 = line_level(np.abs(x), fs_out)
        c100, c200 = line_level(x - x.mean(), fs_out)
        print(f"  {nm:<28} {c100:>+8.1f} {c200:>+8.1f}   "
              f"(仅幅度: {a100:+.1f} / {a200:+.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
