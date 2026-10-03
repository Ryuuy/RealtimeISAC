#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""20260930 新增，同日两次改版。一份原始 IQ（`rt_iqdump.py` 存的 iq_*.npz）
一次性出**一张**大图：

    行1   |IQ| envelope over N TDD periods + 积分窗（照 rt_iqdump.py 的第一栏）
    行2   窗上升沿/下降沿 ±60µs 放大
    行3   逐帧功率时间序列（按 mark 上色）
    行4   按 mark 帧 / 2*mark 帧折叠的箱线图（左右各一个）
    行5   ch0 self |ch0|^2（自相关，天生对称）      + 该行时间积分谱
    行6   ch1 self |ch1|^2（自相关，天生对称）      + 该行时间积分谱
    行7   ch0 x ch1 cross-CAF（真正能分方向的互相关）+ 该行时间积分谱
    行8   DC 脊宽判据（零延迟、纯多普勒、不碰功率）+ PRESENT 阴影

## 20260930 第二次改版：judge 换成"DC 脊宽"，Energy/CFAR 全部拿掉

用户的论文假设要求判据**完全不依赖 power/振幅**（多普勒要能在功率变化之前
预警），而 `rt_detect.EnergyDetector`（08-15 那版，见 `doppler-only-detection-
design.md`）本质是绝对能量判据，`rt_detect.CfarDetector`（局部对比度）实测
对人体这种弥散多普勒几乎不敏感（真机 present=0%，见同一份 memo）——两个都
不满足要求，已经从这个脚本里彻底删除，不再 import `rt_config`/`rt_detect`。

新判据（`dc_ridge_width()`）：静止时 0Hz 附近只有一条窄的静态泄漏，人一动，
这条脊线两侧因为微多普勒会变宽——直接测"以 0Hz 为中心、往两边扩、还比
(全谱中位数+阈值dB)高就继续算，跌破就停"这段的宽度（Hz）。零延迟、单帧内
算完、只用多普勒谱自己的形状，不碰任何 RX 绝对功率。已用真实数据验证过
（见 memo）：安静时 p90≈35-47Hz（含空调/灯干扰），走动时 p90≈150-170Hz，
时间分辨率上跟人开始走动的时刻对得上。**阈值默认 90Hz**——20260930用4份
零人(含空调/灯干扰)+4份走动数据扫过阈值敏感性：90Hz是零人组精确降到0.0%
误触发的分界点，同时走动组仍保留19.7%-60.8%检出率（完整扫描表见memo）。

⚠️ 这是纯离线验证用的新方法，还没有移植进 `realtime/rt_detect.py` 的生产
判决链路，`rt_main.py` 目前跑的还是老的 Energy+CFAR。要不要移植等这个方法
在更多数据上验证过再说。

## 20260930 第三次改版：周期性重新同步 TDD 窗口位置（--resync-sec）

`rt_iqdump.py` 对整份录制只估一次 TDD phase，长录制（15s）里发现：①链路
中途真的断连重连过一次，重连后 phase 漂移到完全不同的值；②即使没断连，
**人在走动、反射信号扰动了 ch0 的功率包络时，盲同步算法本身会被带偏**，
同一份 15s 文件里"没人时"和"人在走"这两段单独重估出来的 phase 能差出
近半个周期。用整份文件的一个全局 phase 去切，中间某一段必然被切歪
（互相关从正确窗口的 0.66 掉到错误窗口的 0.017，纯噪声水平，肉眼看像
"信号消失"，其实是窗口错了）。

**已修**：不再用 npz 里存的单一全局 phase，改成每 `--resync-sec`（默认1.0秒）
重新跑一次 `TddSync`，每一段用它自己重新估出来的 phase/n_int 切帧。实测
开销可忽略（15次独立重同步共0.074s，逐帧算互相关那步本身就要0.5s）。
某段重同步失败（未锁定）时回退到 npz 里存的全局值，并打印告警。

    /usr/bin/python3 debug/full_analysis.py iq_20260930_153336.npz [--fmax 500] [--ridge-thresh-hz 70]

输出：
    <stem>_full.png     一张大图
    <stem>_analysis.npz 中间量（s00/s11/s01/ridge_width/present），
                        以后重新画图不用重跑逐帧计算
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "realtime"))
from rt_config import RtConfig    # noqa: E402
from rt_sync import TddSync       # noqa: E402

N_RING, HOP, NFFT, FS_OUT = 200, 20, 256, 1000.0
DC_GUARD_HZ = 20.0
LINE_HZ = (50.0, 100.0, 150.0, 200.0, 300.0)


def to_db(x):
    return 10.0 * np.log10(np.asarray(x, dtype=np.float64) + 1e-20)


def stft_short(z):
    """跟 selfcaf_doppler.py 完全一样的滑动 STFT 后端（ring=200/hop=20/nfft=256）。
    20260930：不做时域去DC——那是 rt_dsp.py commit 1833792 修掉的 ±50Hz·k 梳子
    根源（每 hop 块状相减=零阶保持阶梯），只靠频域 DC_GUARD_HZ 排除零多普勒。"""
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


def dc_ridge_width(power, freqs, thresh_db=6.0, search_hz=100.0):
    """20260930 新判据，零延迟、纯多普勒、不碰功率。

    从 f=0 所在的 bin 往两边扩：只要下一个 bin 的功率还比
    (全谱中位数 * 10^(thresh_db/10)) 高，就继续往外扩，第一次跌破就停。
    返回这段"以0Hz为中心、连续超阈值"的总宽度（Hz）。

    物理意义：静止时0Hz附近只有一条窄的静态泄漏（几个bin宽），人体多普勒
    的微动会让这条脊线两侧展宽。用全谱中位数当阈值基准，不需要任何历史
    状态、不需要标定噪声本底，单帧内直接算完。
    """
    power = power.astype(np.float64)
    med = np.median(power)
    thresh = med * (10.0 ** (thresh_db / 10.0))
    center = int(np.argmin(np.abs(freqs)))
    n = len(freqs)
    bin_hz = float(freqs[1] - freqs[0])
    j = center
    while j + 1 < n and freqs[j + 1] <= search_hz and power[j + 1] > thresh:
        j += 1
    i = center
    while i - 1 >= 0 and freqs[i - 1] >= -search_hz and power[i - 1] > thresh:
        i -= 1
    return (j - i) * bin_hz


def env(iq, ch, i0, i1, dec=1):
    s = iq[ch, i0 * 2:i1 * 2].astype(np.float64).reshape(-1, 2)
    mag = np.hypot(s[:, 0], s[:, 1]) / 32767.0
    if dec > 1:
        m = (len(mag) // dec) * dec
        mag = mag[:m].reshape(-1, dec).max(axis=1)
    return mag


def main() -> int:
    p = argparse.ArgumentParser(description="一份 iq_*.npz 出一张大图（窗诊断+自/互相关+DC脊宽判据）")
    p.add_argument("npz")
    p.add_argument("--fmax", type=float, default=500.0)
    p.add_argument("--periods", type=int, default=20)
    p.add_argument("--mark", type=int, default=5)
    p.add_argument("--win-ch", type=int, default=1, help="窗诊断第三栏画哪个通道")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--ridge-thresh-db", type=float, default=6.0,
                   help="DC脊宽判据：bin功率超过(全谱中位数+这么多dB)才算进脊线，默认6dB")
    p.add_argument("--ridge-thresh-hz", type=float, default=90.0,
                   help="DC脊宽超过这个值就判PRESENT，默认90Hz——20260930用4份零人(含AC/灯干扰)+"
                        "4份走动数据扫过：90Hz是零人组降到精确0.0%%误触发的分界点，"
                        "同时走动组仍保留19.7%%-60.8%%检出率，见memo里的完整扫描表")
    p.add_argument("--resync-sec", type=float, default=1.0,
                   help="每隔多少秒重新跑一次TddSync（默认1.0s）。长录制/人在走动时TDD窗口位置会漂移，"
                        "全程用npz里存的单一phase会把某些段切歪，见文件头20260930第三次改版说明")
    a = p.parse_args()

    path = a.npz if os.path.isabs(a.npz) else os.path.join(HERE, a.npz)
    d = np.load(path, mmap_mode="r")
    iq = d["iq"]
    fs = float(d["fs"])
    n_per = int(d["n_period"])
    phase0, n_int0 = int(d["phase"]), int(d["n_int"])       # npz全局估计，当分段失锁时的兜底
    locked0, contrast0 = bool(d["locked"]), float(d["contrast"])
    total = iq.shape[1] // 2
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = a.out_dir or HERE

    print(f"=== {base} ===  {total/fs*1e3:.0f}ms x2ch  fs={fs/1e6:.0f}MHz  "
          f"全局估计(npz原始,当兜底用): phase={phase0} n_int={n_int0}({n_int0/fs*1e6:.0f}µs)  "
          f"{'锁定' if locked0 else '未锁定'} contrast={contrast0:.2f}")
    if not locked0:
        print("⚠️ 全局TDD未锁定，分段重同步若也失败将退化到无意义的默认窗位置", file=sys.stderr)

    # ---- 分段重新同步 TDD 窗口位置（20260930第三次改版）----
    # 长录制/人走动时盲同步会被带偏，全程用一个phase会把某段切歪，见文件头说明。
    resync_n = max(1, int(round(a.resync_sec * fs)))
    bounds = list(range(0, total, resync_n))
    if bounds[-1] != total:
        bounds.append(total)
    segments = []  # (seg_start, seg_end, phase, n_int, contrast, locked)
    for i in range(len(bounds) - 1):
        s0, s1 = bounds[i], bounds[i + 1]
        seg_len_s = (s1 - s0) / fs
        if seg_len_s < 0.05 and segments:  # 太短的尾巴段并入上一段，不单独估
            ps0, ps1, ph, ni, ct, lk = segments[-1]
            segments[-1] = (ps0, s1, ph, ni, ct, lk)
            continue
        raw = np.asarray(iq[:, s0 * 2:s1 * 2])
        cfg = RtConfig(sample_rate=fs, step_sec=max(seg_len_s - 0.002, 0.01))
        sync = TddSync(cfg)
        sync.update(raw, force=True)
        if sync.locked:
            segments.append((s0, s1, sync.phase, sync.n_integrate, sync.contrast, True))
        else:
            segments.append((s0, s1, phase0, n_int0, contrast0, False))

    n_ok = sum(1 for seg in segments if seg[5])
    phases_seen = [seg[2] for seg in segments]
    print(f"\n分段重新同步：共{len(segments)}段(每段~{a.resync_sec:.1f}s)，"
          f"{n_ok}/{len(segments)}段成功锁定，phase范围=[{min(phases_seen)},{max(phases_seen)}]"
          f"（跨度大说明链路/走动确实让窗口漂移了，跟npz里全局phase={phase0}多半对不上）")
    for s0, s1, ph, ni, ct, lk in segments:
        print(f"  t={s0/fs:5.2f}-{s1/fs:5.2f}s  phase={ph:5d}  n_int={ni:5d}({ni/fs*1e6:.0f}µs)  "
              f"contrast={ct:.2f}  {'✓' if lk else '⚠回退到全局值'}")

    # 第一段的估计值用于窗诊断图（面板1画的就是文件开头那一小段）
    phase, n_int, contrast, locked = segments[0][2], segments[0][3], segments[0][4], segments[0][5]

    def frame_power(ch):
        out = []
        for s0, s1, ph, ni, _, _ in segments:
            n_fr_i = (s1 - s0 - ph - ni) // n_per
            for k in range(n_fr_i):
                s = s0 + ph + k * n_per
                seg = iq[ch, s * 2:(s + ni) * 2].astype(np.float64)
                out.append((seg * seg).sum() / ni)
        return np.array(out) / (32767.0 ** 2)

    p0 = frame_power(0)
    p1 = frame_power(1)
    print("\n--- 每帧积分功率（窗内）---")
    for ch, pp in ((0, p0), (1, p1)):
        v = pp[np.isfinite(pp)]
        print(f"  ch{ch}: 均值 {to_db(v.mean()):.2f}dB  "
              f"逐帧起伏 std/mean = {100*v.std()/v.mean():.2f}%  "
              f"极差 {to_db(v.max())-to_db(v.min()):.2f}dB")

    # ---- 逐帧自相关 s00/s11 + 互相关 s01（归一化，跟 rt_dsp.py 同一套数学）----
    # 每帧用它所在那一段自己重新估出来的 (phase, n_int)，不是全程一个固定值。
    frame_specs = []  # (绝对起点样本, 该段的n_int)
    for s0, s1, ph, ni, _, _ in segments:
        n_fr_i = (s1 - s0 - ph - ni) // n_per
        for k in range(n_fr_i):
            frame_specs.append((s0 + ph + k * n_per, ni))
    n_fr = len(frame_specs)

    def load_frame(ch, s, ni):
        seg = np.asarray(iq[ch, s * 2:(s + ni) * 2], dtype=np.float64)
        return seg[0::2] + 1j * seg[1::2]

    print(f"\n逐帧算自相关 s00/s11 + 互相关 s01（{n_fr} 帧，跨{len(segments)}段独立窗口）…")
    t0 = time.time()
    s00 = np.empty(n_fr, np.float64)
    s11 = np.empty(n_fr, np.float64)
    s01 = np.empty(n_fr, np.complex128)
    for k, (s, ni) in enumerate(frame_specs):
        a0 = load_frame(0, s, ni)
        b0 = load_frame(1, s, ni)
        p0v = np.vdot(a0, a0).real
        p1v = np.vdot(b0, b0).real
        s00[k] = p0v / ni
        s11[k] = p1v / ni
        s01[k] = np.vdot(a0, b0) / (np.sqrt(p0v) * np.sqrt(p1v))
    print(f"  用时 {time.time()-t0:.1f}s  ch0功率 {10*np.log10(s00.mean()/32767**2):.1f}dBFS  "
          f"ch1功率 {10*np.log10(s11.mean()/32767**2):.1f}dBFS")

    streams = [
        ("ch0 self  |ch0|^2  (reference x itself)", s00.astype(np.complex128)),
        ("ch1 self  |ch1|^2  (sensing x itself)", s11.astype(np.complex128)),
        ("ch0 x ch1  cross-CAF  (normal Doppler)", s01),
    ]
    res = []
    for name, z in streams:
        tt, freqs, S = stft_short(z)
        res.append((name, tt, freqs, S))
    tt_cross, freqs_cross, S_cross = res[2][1], res[2][2], res[2][3]
    n_steps = S_cross.shape[0]

    # ---- DC 脊宽判据（零延迟，逐帧算，不碰功率）----
    ridge_width = np.array([
        dc_ridge_width(S_cross[i], freqs_cross, thresh_db=a.ridge_thresh_db)
        for i in range(n_steps)
    ])
    present = ridge_width >= a.ridge_thresh_hz
    print(f"\nDC脊宽判据（thresh_db={a.ridge_thresh_db} thresh_hz={a.ridge_thresh_hz}）："
          f"present占比 {100*present.mean():.1f}%  "
          f"脊宽 p10={np.percentile(ridge_width,10):.1f}Hz p50={np.percentile(ridge_width,50):.1f}Hz "
          f"p90={np.percentile(ridge_width,90):.1f}Hz")

    # ================================================================ 出图
    import matplotlib
    show = bool(os.environ.get("DISPLAY")) and sys.stdout.isatty()
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(17, 30))
    gs = fig.add_gridspec(
        8, 2, width_ratios=[3, 1],
        height_ratios=[1.1, 0.85, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9],
        hspace=0.4, wspace=0.22)
    us = 1e6 / fs

    # ---- 行1：N个TDD周期包络 + 积分窗 ----
    ax = fig.add_subplot(gs[0, :])
    npd = max(1, a.periods)
    i0 = max(0, phase - n_per // 2)
    i1 = min(total, i0 + npd * n_per)
    dec = max(1, (i1 - i0) // 6000)
    for c, col in ((0, "tab:blue"), (1, "tab:orange")):
        y = to_db(env(iq, c, i0, i1, dec) ** 2)
        ax.plot((i0 + np.arange(len(y)) * dec) * us / 1000.0, y, lw=0.6, color=col, label=f"ch{c}")
    for k in range(-1, npd + 1):
        s = phase + k * n_per
        if i0 <= s < i1:
            ax.axvspan(s * us / 1000.0, (s + n_int) * us / 1000.0, color="tab:green", alpha=0.13)
            if k >= 0 and k % a.mark == 0:
                ax.axvline(s * us / 1000.0, color="tab:red", lw=1.1, alpha=0.8)
    ax.set_title(f"[1] |IQ| envelope over {npd} TDD periods — green = integration window "
                 f"({n_int*us:.0f}µs of each 1ms) — red line every {a.mark} frames "
                 f"({a.mark} ms = {1000/a.mark:.0f} Hz)   |   TDD {'LOCKED' if locked else 'NOT LOCKED'} "
                 f"contrast={contrast:.2f}")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("dBFS"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    # ---- 行2：上升沿/下降沿放大 ----
    ax = fig.add_subplot(gs[1, :])
    w = int(60 / us)
    for tag, edge, c in (("rising", phase, "tab:green"), ("falling", phase + n_int, "tab:red")):
        lo, hi = max(0, edge - w), min(total, edge + w)
        y = to_db(env(iq, 0, lo, hi) ** 2)
        ax.plot((np.arange(len(y)) + lo - edge) * us, y, lw=0.8, color=c, label=f"ch0 {tag} edge")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_title("[2] Zoom on the two window boundaries (ch0), ±60µs")
    ax.set_xlabel("Offset from window boundary (µs)"); ax.set_ylabel("dBFS")
    ax.legend(); ax.grid(alpha=0.3)

    # ---- 行3：逐帧功率时间序列 ----
    ax = fig.add_subplot(gs[2, :])
    nshow = min(n_fr, 120)
    tt120 = np.arange(nshow)
    psel = p1 if a.win_ch == 1 else p0
    v0 = 100 * (psel[:nshow] / np.nanmean(psel) - 1)
    vall = 100 * (psel / np.nanmean(psel) - 1)
    ax.plot(tt120, v0, lw=0.6, color="0.6", zorder=1)
    cmap = plt.get_cmap("tab10")
    for r in range(a.mark):
        sel = tt120 % a.mark == r
        ax.scatter(tt120[sel], v0[sel], s=26, color=cmap(r), zorder=3, label=f"frame%{a.mark}=={r}")
        ax.axhline(np.nanmean(vall[np.arange(len(vall)) % a.mark == r]), color=cmap(r), lw=1.4, ls="--", alpha=0.9)
    ax.set_title(f"[3] ch{a.win_ch} per-frame power (deviation from mean), colored mod {a.mark}")
    ax.set_xlabel("TDD frame index (1 frame = 1 ms)")
    ax.set_ylabel("Deviation (%)"); ax.legend(fontsize=7, ncol=a.mark); ax.grid(alpha=0.3)

    # ---- 行4：mark / 2*mark 折叠箱线图 ----
    for col, period in ((0, a.mark), (1, 2 * a.mark)):
        ax = fig.add_subplot(gs[3, col])
        for c, pp, colr, off in ((0, p0, "tab:blue", -0.16), (1, p1, "tab:orange", 0.16)):
            v = pp[np.isfinite(pp)]
            m = (len(v) // period) * period
            if m < period * 3:
                continue
            g = 100 * (v[:m].reshape(-1, period) / v[:m].mean() - 1)
            bp = ax.boxplot([g[:, i] for i in range(period)], positions=np.arange(period) + off,
                            widths=0.28, showfliers=False, patch_artist=True, medianprops=dict(color="k", lw=1.2))
            for b in bp["boxes"]:
                b.set(facecolor=colr, alpha=0.45)
            mu = g.mean(axis=0)
            sem = g.std(axis=0, ddof=1) / np.sqrt(g.shape[0])
            ax.errorbar(np.arange(period) + off, mu, yerr=sem, fmt="D", ms=4, color=colr,
                        ecolor="k", capsize=3, zorder=5, label=f"ch{c} mean±SEM")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(np.arange(period)); ax.set_xticklabels([str(i) for i in range(period)])
        ax.set_title(f"[4] Grouped mod {period} ({period}ms = {1000/period:.0f}Hz)")
        ax.set_xlabel(f"Position within {period}-frame group")
        ax.set_ylabel("Deviation (%)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ---- 行5-7：ch0 self / ch1 self / cross-CAF ----
    print("\n时间积分谱里 TX 线相对本底 (dB):")
    for i, (name, tt, freqs, S) in enumerate(res):
        row = 4 + i
        fmask = np.abs(freqs) <= a.fmax
        fa = freqs[fmask]
        D = 10 * np.log10(S[:, fmask].T + 1e-20)
        vmin, vmax = np.percentile(D, 5), np.percentile(D, 95)
        ax = fig.add_subplot(gs[row, 0])
        ax.pcolormesh(tt, fa, D, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_ylabel("Doppler (Hz)"); ax.set_title(f"[{5+i}] {name}"); ax.set_ylim(-a.fmax, a.fmax)
        for sgn in (-DC_GUARD_HZ, DC_GUARD_HZ):
            ax.axhline(sgn, color="w", ls="--", lw=0.6, alpha=0.5)
        if i == 2:
            ax.set_xlabel("Time (s)")

        g = 10 * np.log10(S[:, fmask].mean(axis=0) + 1e-20)
        ex, floor = line_excess(g, fa)
        axg = fig.add_subplot(gs[row, 1])
        axg.plot(g - g.max(), fa, lw=1.0, color="tab:blue")
        for f0 in LINE_HZ:
            for sgn in (-1, 1):
                axg.axhline(sgn * f0, color="0.7", lw=0.5, ls=":")
        axg.set_title("time-integrated"); axg.set_xlabel("dB (norm)")
        axg.set_ylim(-a.fmax, a.fmax); axg.grid(alpha=0.3)
        print(f"  {name[:40]:40s} floor={floor:6.1f}  " + " ".join(f"{int(f)}:{ex[f]:+5.1f}" for f in LINE_HZ))

    # ---- 行8：DC 脊宽判据 ----
    axR = fig.add_subplot(gs[7, 0])
    idx = np.flatnonzero(np.diff(np.concatenate(([0], present.astype(int), [0]))))
    for s0, e0 in zip(idx[0::2], idx[1::2]):
        axR.axvspan(tt_cross[s0], tt_cross[min(e0, n_steps - 1)], color="pink", alpha=0.35)
    axR.plot(tt_cross, ridge_width, color="tab:purple", lw=1.0, label="DC ridge width (Hz)")
    axR.axhline(a.ridge_thresh_hz, color="tab:red", lw=0.8, ls=":",
                label=f"threshold ({a.ridge_thresh_hz:.0f}Hz)")
    axR.set_xlim(tt_cross[0], tt_cross[-1])  # 跟谱图(pcolormesh天然贴边)对齐，line plot默认会留白边
    axR.set_ylabel("Ridge width (Hz)"); axR.set_xlabel("Time (s)")
    axR.legend(fontsize=8, loc="upper right"); axR.grid(alpha=0.3)
    axR.set_title("[8] DC ridge width (zero-delay, Doppler-only, no power) — pink = PRESENT")

    fig.suptitle(f"{base}  —  full analysis  (fs={fs/1e6:.0f}MHz, TDD {'locked' if locked else 'NOT locked'}, "
                 f"contrast={contrast:.2f}, present={100*present.mean():.0f}%)", fontsize=13, y=0.9995)
    fig.subplots_adjust(top=0.985)

    out = os.path.join(out_dir, f"{base}_full.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n🖼  {out}")

    npz_out = os.path.join(out_dir, f"{base}_analysis.npz")
    np.savez(npz_out, t=tt_cross, freqs=freqs_cross, S_cross=S_cross.astype(np.float32),
             s00=s00, s11=s11, s01=s01.astype(np.complex64),
             ridge_width=ridge_width, present=present,
             p0=p0, p1=p1, src=base)
    print(f"💾 {npz_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
