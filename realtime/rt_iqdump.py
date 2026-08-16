#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓一小段原始 IQ 存下来 + 出图，专门用来**肉眼看门控切得干不干净**。

这是唯一会存原始 IQ 的脚本（实时链路的硬约束是不落盘，这个是临时诊断工具）。

    cd RealtimeISAC/realtime
    /usr/bin/python3 rt_iqdump.py                  # 抓 200ms，存 npz + 出图
    /usr/bin/python3 rt_iqdump.py --ms 500         # 抓久一点，超帧看得更清楚
    /usr/bin/python3 rt_iqdump.py --no-save        # 只出图不存 npz

出四张图：

1. **三个 TDD 周期的 |IQ| 包络 + 积分窗**（阴影）—— 直接看窗有没有切到沿上
2. **上升沿 / 下降沿放大**（±60µs）—— 看暂态有多长、窗内缩够不够
3. **每帧积分功率的时间序列**（1kHz）—— 看是不是真的每 10ms 涨一次
4. **按 5帧/10帧折叠**（5ms=200Hz / 10ms=100Hz 超帧）—— 如果真有周期性，
   折叠后会站出一个稳定的形状；纯噪声折出来是平的

存的 npz 里有原始 int16 交织数据（`iq` 字段，shape=(2, N*2)），
自己想怎么切都行：`ch0 = iq[0,0::2] + 1j*iq[0,1::2]`
"""

import argparse
import os
import sys
import time

import numpy as np

from rt_config import RtConfig
from rt_dsp import to_db
from rt_sync import TddSync

DEBUG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "debug")


def frame_power(iq, ch, start, n_int, n_per, n_fr, total):
    """每个 TDD 帧在 [start, start+n_int) 窗内的平均功率。"""
    p = np.empty(n_fr)
    for k in range(n_fr):
        s = start + k * n_per
        if s + n_int > total:
            p[k] = np.nan
            continue
        seg = iq[ch, s * 2:(s + n_int) * 2].astype(np.float64)
        p[k] = (seg * seg).sum() / n_int
    return p / (32767.0 ** 2)


def main() -> int:
    p = argparse.ArgumentParser(description="抓原始 IQ 看门控切得干不干净")
    p.add_argument("--ms", type=float, default=200.0, help="抓多少毫秒")
    p.add_argument("--fs", type=float, default=10.0, help="采样率 MHz")
    p.add_argument("--gain", type=float, default=30.0)
    p.add_argument("--serial", type=str, default="32392D3")
    p.add_argument("--no-save", action="store_true", help="不存 npz，只出图")
    p.add_argument("--out-dir", type=str, default=DEBUG_DIR)
    p.add_argument("--periods", type=int, default=20,
                   help="第一栏画多少个 TDD 周期（默认 20）")
    p.add_argument("--mark", type=int, default=5,
                   help="按这个周期给帧上色/分组（默认 5，即 200Hz）")
    p.add_argument("--ch", type=int, default=1,
                   help="第三栏画哪个通道的逐帧功率（默认 1，超帧调制主要在 ch1）")
    p.add_argument("--from-npz", type=str, default=None,
                   help="不碰硬件，直接从存好的 iq_*.npz 重画（换 --mark/--ch 用）")
    a = p.parse_args()

    cfg = RtConfig(serial=a.serial, sample_rate=a.fs * 1e6, gain=a.gain)
    n_step = cfg.n_step
    n_cap = max(1, int(np.ceil(a.ms * 1e-3 / cfg.step_sec)))

    if a.from_npz:
        # 重画模式：窗位置直接沿用采集时存下来的，保证和当时看到的是同一个窗
        src_path = (a.from_npz if os.path.isabs(a.from_npz)
                    else os.path.join(a.out_dir, a.from_npz))
        d = np.load(src_path)
        iq = d["iq"]
        cfg = RtConfig(serial=a.serial, sample_rate=float(d["fs"]),
                       gain=float(d["gain"]))
        total = iq.shape[1] // 2
        n_per, n_int, phase = int(d["n_period"]), int(d["n_int"]), int(d["phase"])
        locked, contrast = bool(d["locked"]), float(d["contrast"])
        errors, dropped = "(重画，无采集)", 0
        a.no_save = True
        print(f"从 {os.path.basename(src_path)} 重画（不碰硬件）")

        class _S:                     # 只为了下面统一走 sync.status()/locked
            pass
        sync = _S()
        sync.locked, sync.contrast = locked, contrast
        sync.status = lambda: (f"TDD锁定 phase={phase} 对比度={contrast:.2f} "
                               f"积分={n_int}样本({n_int/cfg.sample_rate*1e6:.0f}µs)"
                               if locked else "TDD未锁定")
    else:
        from rt_source import UsrpSource
        sync = TddSync(cfg)
        src = UsrpSource(cfg).open()
        try:
            chunks = []
            for k, raw in enumerate(src.steps()):
                if k >= n_cap:
                    break
                chunks.append(raw.copy())
                sync.update(raw, force=True)
            errors, dropped = str(src.errors), src.dropped
        finally:
            src.close()

        iq = np.concatenate(chunks, axis=1)      # (2, N*2) int16
        total = iq.shape[1] // 2
        n_per, n_int = cfg.n_period, sync.n_integrate
        phase = sync.phase
    n_fr = (total - phase - n_int) // n_per

    print("=" * 72)
    print(f"抓到 {total/cfg.sample_rate*1e3:.0f} ms x2ch @ {cfg.sample_rate/1e6:.0f}MHz "
          f"gain={cfg.gain:.0f}dB   UHD: {errors} 丢步={dropped}")
    print(sync.status())
    if not sync.locked:
        print("⚠️ TDD 未锁定（TX 没开？），下面的窗位置是无意义的默认值")
    print(f"积分窗 = 每个 1ms 周期的 [{phase}, {phase+n_int}) 样本 "
          f"= [{phase/cfg.sample_rate*1e6:.0f}, {(phase+n_int)/cfg.sample_rate*1e6:.0f}) µs"
          f"，共 {n_fr} 帧")

    path = None
    if not a.no_save:
        os.makedirs(a.out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(a.out_dir, f"iq_{stamp}.npz")
        np.savez(path, iq=iq, fs=cfg.sample_rate, phase=phase, n_int=n_int,
                 n_period=n_per, locked=sync.locked, contrast=sync.contrast,
                 gain=cfg.gain, center_freq=cfg.center_freq)
        print(f"💾 原始 IQ 已存 {path}  ({os.path.getsize(path)/1e6:.0f} MB)")
        print("   自己切: d=np.load(路径); ch0=d['iq'][0,0::2]+1j*d['iq'][0,1::2]")

    # ---- 每帧积分功率（这就是喂给 FFT 的那个 1kHz 流的幅度）----
    p0 = frame_power(iq, 0, phase, n_int, n_per, n_fr, total)
    p1 = frame_power(iq, 1, phase, n_int, n_per, n_fr, total)
    print(f"\n--- 每帧积分功率（窗内）---")
    for ch, pp in ((0, p0), (1, p1)):
        v = pp[np.isfinite(pp)]
        print(f"  ch{ch}: 均值 {to_db(v.mean()):.2f}dB  "
              f"逐帧起伏 std/mean = {100*v.std()/v.mean():.2f}%  "
              f"极差 {to_db(v.max())-to_db(v.min()):.2f}dB")

    print(f"\n--- 按超帧折叠（看是不是真有 5ms / 10ms 周期）---")
    for period in (5, 10):
        print(f"  周期 {period} 帧 ({period}ms, {1000/period:.0f}Hz):")
        for ch, pp in ((0, p0), (1, p1)):
            v = pp[np.isfinite(pp)]
            m = (len(v) // period) * period
            if m < period * 3:
                continue
            g = v[:m].reshape(-1, period)
            mu = g.mean(axis=0)
            # 组间差异 vs 组内噪声：>3 才算真有结构
            sem = g.std(axis=0, ddof=1) / np.sqrt(g.shape[0])
            snr = (mu.max() - mu.min()) / max(sem.mean(), 1e-30)
            print(f"    ch{ch} 各相位均值(相对总均值,%): "
                  + " ".join(f"{100*(x/mu.mean()-1):+5.2f}" for x in mu))
            print(f"         峰谷差 {100*(mu.max()-mu.min())/mu.mean():.2f}% "
                  f"= {snr:.1f}x 组内标准误 "
                  f"{'<- 有结构' if snr > 3 else '<- 和噪声分不开'}")

    # ---- 出图 ----
    import matplotlib
    show = bool(os.environ.get("DISPLAY")) and sys.stdout.isatty()
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.15, 0.85, 1.0, 1.0], hspace=0.32)
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :]),
            fig.add_subplot(gs[2, :]), fig.add_subplot(gs[3, 0]),
            fig.add_subplot(gs[3, 1])]
    us = 1e6 / cfg.sample_rate

    def env(ch, i0, i1, dec=1):
        s = iq[ch, i0 * 2:i1 * 2].astype(np.float64).reshape(-1, 2)
        mag = np.hypot(s[:, 0], s[:, 1]) / 32767.0
        if dec > 1:
            m = (len(mag) // dec) * dec
            mag = mag[:m].reshape(-1, dec).max(axis=1)
        return mag

    # 1) N 个周期 + 积分窗（每 mark 帧画一条红色分隔，肉眼数周期用）
    ax = axes[0]
    npd = max(1, a.periods)
    i0 = max(0, phase - n_per // 2)
    i1 = min(total, i0 + npd * n_per)
    dec = max(1, (i1 - i0) // 6000)          # 控制在 ~6000 点，太密看不清
    for ch, c in ((0, "tab:blue"), (1, "tab:orange")):
        y = to_db(env(ch, i0, i1, dec) ** 2 + 1e-20)
        ax.plot((i0 + np.arange(len(y)) * dec) * us / 1000.0, y, lw=0.6, color=c,
                label=f"ch{ch}")
    for k in range(-1, npd + 1):
        s = phase + k * n_per
        if i0 <= s < i1:
            ax.axvspan(s * us / 1000.0, (s + n_int) * us / 1000.0,
                       color="tab:green", alpha=0.13)
            if k >= 0 and k % a.mark == 0:   # 每 mark 帧一条红线
                ax.axvline(s * us / 1000.0, color="tab:red", lw=1.1, alpha=0.8)
    ax.set_title(f"|IQ| envelope over {npd} TDD periods — green = integration window "
                 f"({n_int*us:.0f}µs of each 1ms) — red line every {a.mark} frames "
                 f"({a.mark} ms = {1000/a.mark:.0f} Hz)")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("dBFS")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    # 2) 上升沿/下降沿放大
    ax = axes[1]
    w = int(60 / us)
    for tag, edge, c in (("rising", phase, "tab:green"),
                         ("falling", phase + n_int, "tab:red")):
        lo, hi = max(0, edge - w), min(total, edge + w)
        y = to_db(env(0, lo, hi) ** 2 + 1e-20)
        ax.plot((np.arange(len(y)) + lo - edge) * us, y, lw=0.8, color=c,
                label=f"ch0 {tag} edge (window boundary at 0)")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_title("Zoom on the two window boundaries (ch0), ±60µs")
    ax.set_xlabel("Offset from window boundary (µs)"); ax.set_ylabel("dBFS")
    ax.legend(); ax.grid(alpha=0.3)

    # 3) 每帧积分功率序列，按 mark 上色 —— 若真有"每 mark 帧高一个"，
    #    同一种颜色会系统性地偏上。只画前 120 帧，太密就看不出颜色了。
    ax = axes[2]
    nshow = min(n_fr, 120)
    tt = np.arange(nshow)
    psel = p1 if a.ch == 1 else p0
    v0 = 100 * (psel[:nshow] / np.nanmean(psel) - 1)
    vall = 100 * (psel / np.nanmean(psel) - 1)      # 虚线用全部帧算，不只前 120
    ax.plot(tt, v0, lw=0.6, color="0.6", zorder=1)
    cmap = plt.get_cmap("tab10")
    for r in range(a.mark):
        sel = tt % a.mark == r
        ax.scatter(tt[sel], v0[sel], s=26, color=cmap(r), zorder=3,
                   label=f"frame%{a.mark}=={r}")
        ax.axhline(np.nanmean(vall[np.arange(len(vall)) % a.mark == r]),
                   color=cmap(r), lw=1.4, ls="--", alpha=0.9)
    ax.set_title(f"ch{a.ch} per-frame power (deviation from mean), colored by "
                 f"frame index mod {a.mark}. Dashed = mean of each colour over ALL "
                 f"{n_fr} frames. Separated dashed lines = real "
                 f"'every {a.mark}th frame differs'.")
    ax.set_xlabel("TDD frame index (1 frame = 1 ms)")
    ax.set_ylabel("Deviation (%)"); ax.legend(fontsize=7, ncol=a.mark); ax.grid(alpha=0.3)

    # 4/5) 按 5 帧 和 10 帧 分组的箱线图 —— 1% 的系统性偏差在这里看得见
    for ax, period in ((axes[3], a.mark), (axes[4], 2 * a.mark)):
        for ch, pp, c, off in ((0, p0, "tab:blue", -0.16), (1, p1, "tab:orange", 0.16)):
            v = pp[np.isfinite(pp)]
            m = (len(v) // period) * period
            if m < period * 3:
                continue
            g = 100 * (v[:m].reshape(-1, period) / v[:m].mean() - 1)
            bp = ax.boxplot([g[:, i] for i in range(period)],
                            positions=np.arange(period) + off, widths=0.28,
                            showfliers=False, patch_artist=True,
                            medianprops=dict(color="k", lw=1.2))
            for b in bp["boxes"]:
                b.set(facecolor=c, alpha=0.45)
            mu = g.mean(axis=0)
            sem = g.std(axis=0, ddof=1) / np.sqrt(g.shape[0])
            ax.errorbar(np.arange(period) + off, mu, yerr=sem, fmt="D", ms=4,
                        color=c, ecolor="k", capsize=3, zorder=5,
                        label=f"ch{ch} mean±SEM")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(np.arange(period))
        ax.set_xticklabels([str(i) for i in range(period)])
        ax.set_title(f"Grouped by frame index mod {period} "
                     f"({period} ms = {1000/period:.0f} Hz)\n"
                     f"box = spread of individual frames, diamond = mean ± SEM")
        ax.set_xlabel(f"Position within {period}-frame group")
        ax.set_ylabel("Deviation (%)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 不能用 tight_layout：上面是手工 gridspec（含 hspace），它会警告并可能画歪
    png = (path.replace(".npz", ".png") if path
           else os.path.join(a.out_dir, "iq_latest.png"))
    os.makedirs(os.path.dirname(png), exist_ok=True)
    fig.savefig(png, dpi=110, bbox_inches="tight")
    print(f"\n🖼  {png}")
    if show:
        plt.show()
    plt.close(fig)
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
