#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查看 FFT/Blackman 之前的逐 TDD 帧复数 channel。

计算式与实时内核一致：

    h[k] = mean(conj(ch0[k]) * ch1[k])

其中代码 ch0 是 reference。脚本只做逐帧积分，不做 DC EMA、Blackman、FFT 或
CFAR，专门用于判断原始 channel 本身是否存在每 5 帧重复的幅度/相位结构。

    /usr/bin/python3 debug/plot_frame_channel.py
    /usr/bin/python3 debug/plot_frame_channel.py debug/iq_xxx.npz --no-show
"""

import argparse
import glob
import os
import sys

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SC16_SCALE = 32767.0


def newest_iq(directory: str) -> str | None:
    files = sorted(glob.glob(os.path.join(directory, "iq_*.npz")))
    return files[-1] if files else None


def resolve_path(path: str | None) -> str | None:
    path = path or newest_iq(HERE)
    if path is None or os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(HERE, path)


def frame_channel(iq: np.ndarray, phase: int, n_int: int,
                  n_period: int) -> np.ndarray:
    """严格复现 rt_dsp 的逐帧 int16/int64 相关积分，返回归一化 complex64。"""
    total = iq.shape[1] // 2
    n_frames = (total - phase - n_int) // n_period
    if n_frames < 1:
        return np.empty(0, dtype=np.complex64)

    out = np.empty(n_frames, dtype=np.complex64)
    scale = SC16_SCALE * SC16_SCALE * n_int
    for k in range(n_frames):
        start = phase + k * n_period
        a = iq[0, start * 2:(start + n_int) * 2].reshape(n_int, 2)
        b = iq[1, start * 2:(start + n_int) * 2].reshape(n_int, 2)
        ar, ai = a[:, 0], a[:, 1]
        br, bi = b[:, 0], b[:, 1]
        re = (np.einsum("i,i->", ar, br, dtype=np.int64)
              + np.einsum("i,i->", ai, bi, dtype=np.int64))
        im = (np.einsum("i,i->", ar, bi, dtype=np.int64)
              - np.einsum("i,i->", ai, br, dtype=np.int64))
        out[k] = np.complex64(re / scale + 1j * (im / scale))
    return out


def detrended_phase_deg(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回展相相位及扣除线性趋势后的相位，单位均为度。"""
    idx = np.arange(len(h), dtype=np.float64)
    unwrapped = np.unwrap(np.angle(h))
    if len(h) >= 2:
        trend = np.polyval(np.polyfit(idx, unwrapped, 1), idx)
    else:
        trend = np.zeros_like(unwrapped)
    return np.rad2deg(unwrapped), np.rad2deg(unwrapped - trend)


def folded_stats(h: np.ndarray, phase_resid_deg: np.ndarray,
                 period: int = 5):
    n = len(h) // period * period
    if n < period:
        raise ValueError(f"至少需要 {period} 帧，当前只有 {len(h)} 帧")
    hg = h[:n].reshape(-1, period)
    ag = np.abs(hg)
    pg = phase_resid_deg[:n].reshape(-1, period)
    amp_mean = ag.mean(axis=0)
    amp_sem = ag.std(axis=0, ddof=1) / np.sqrt(ag.shape[0]) \
        if ag.shape[0] > 1 else np.zeros(period)
    phase_mean = pg.mean(axis=0)
    phase_sem = pg.std(axis=0, ddof=1) / np.sqrt(pg.shape[0]) \
        if pg.shape[0] > 1 else np.zeros(period)
    return hg, ag, pg, amp_mean, amp_sem, phase_mean, phase_sem


def projection_dbc(h: np.ndarray, fs_out: float, freq: float) -> float:
    """未加窗复投影相对 channel 复均值的 dBc。"""
    idx = np.arange(len(h), dtype=np.float64)
    carrier = abs(h.mean())
    tone = abs(np.mean((h - h.mean())
                       * np.exp(-2j * np.pi * freq * idx / fs_out)))
    return float(20.0 * np.log10((tone + 1e-30) / (carrier + 1e-30)))


def main() -> int:
    p = argparse.ArgumentParser(description="画 FFT 前的逐 TDD 帧复数 channel")
    p.add_argument("npz", nargs="?", default=None,
                   help="iq_*.npz 路径（省略 = debug/ 下最新一份）")
    p.add_argument("--period", type=int, default=5,
                   help="折叠/着色周期，默认5帧（200Hz）")
    p.add_argument("--out", type=str, default=None, help="PNG 输出路径")
    p.add_argument("--no-show", action="store_true", help="只保存PNG，不弹窗")
    p.add_argument("--print-limit", type=int, default=0,
                   help="逐帧表最多打印多少行；0=全部")
    a = p.parse_args()

    path = resolve_path(a.npz)
    if not path or not os.path.exists(path):
        print("没有 iq_*.npz；先用 realtime/rt_iqdump.py 抓一份原始IQ。",
              file=sys.stderr)
        return 1
    if a.period < 2:
        print("--period 必须 >= 2", file=sys.stderr)
        return 2

    with np.load(path) as data:
        required = {"iq", "fs", "phase", "n_int", "n_period"}
        missing = sorted(required.difference(data.files))
        if missing:
            print(f"NPZ 缺字段: {', '.join(missing)}", file=sys.stderr)
            return 2
        iq = data["iq"]
        fs = float(data["fs"])
        phase = int(data["phase"])
        n_int = int(data["n_int"])
        n_period = int(data["n_period"])

    if iq.ndim != 2 or iq.shape[0] < 2 or iq.dtype != np.int16:
        print(f"iq 格式不对：期望 (>=2, N*2) int16，实际 {iq.shape} {iq.dtype}",
              file=sys.stderr)
        return 2

    h = frame_channel(iq, phase, n_int, n_period)
    if len(h) < a.period:
        print(f"有效帧不足：{len(h)}", file=sys.stderr)
        return 2

    fs_out = fs / n_period
    idx = np.arange(len(h))
    magnitude = np.abs(h)
    phase_deg, phase_resid = detrended_phase_deg(h)
    centered = h - h.mean()
    (hg, ag, pg, amp_mean, amp_sem,
     phase_mean, phase_sem) = folded_stats(h, phase_resid, a.period)

    base = os.path.basename(path)
    print(f"=== {base} ===")
    print(f"代码 ch0=reference；h[k]=mean(conj(ch0)*ch1)")
    print(f"{len(h)} 帧  fs_out={fs_out:.3f}Hz  phase={phase}样本  "
          f"积分={n_int}样本({n_int/fs*1e6:.1f}µs)")
    print("本脚本未做 DC EMA / Blackman / FFT / CFAR")
    print(f"未加窗复投影: +200Hz={projection_dbc(h, fs_out, 200.0):+.2f}dBc  "
          f"-200Hz={projection_dbc(h, fs_out, -200.0):+.2f}dBc")

    amp_ref = magnitude.mean()
    print(f"\n--- frame mod {a.period} 折叠（均值 ± SEM）---")
    print(f"{'位置':>4} {'幅度偏差%':>12} {'幅度SEM%':>12} "
          f"{'去趋势相位°':>14} {'相位SEM°':>12} {'复均值(re, im)':>27}")
    for r in range(a.period):
        cm = hg[:, r].mean()
        print(f"{r:>4d} {100*(amp_mean[r]/amp_ref-1):>+12.4f} "
              f"{100*amp_sem[r]/amp_ref:>12.4f} "
              f"{phase_mean[r]:>+14.4f} {phase_sem[r]:>12.4f} "
              f"({cm.real:+.6e}, {cm.imag:+.6e})")
    amp_p2p = 100.0 * np.ptp(amp_mean) / amp_ref
    phase_p2p = float(np.ptp(phase_mean))
    print(f"幅度折叠峰谷={amp_p2p:.4f}%  相位折叠峰谷={phase_p2p:.4f}°")

    limit = len(h) if a.print_limit <= 0 else min(len(h), a.print_limit)
    print(f"\n--- 逐帧 channel（打印 {limit}/{len(h)}）---")
    print("frame,real,imag,magnitude,phase_deg,phase_detrended_deg,frame_mod")
    for k in range(limit):
        print(f"{k},{h[k].real:.9e},{h[k].imag:.9e},{magnitude[k]:.9e},"
              f"{phase_deg[k]:.6f},{phase_resid[k]:.6f},{k % a.period}")

    show = (not a.no_show) and bool(os.environ.get("DISPLAY")) and sys.stdout.isatty()
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    colors = [cmap(k % a.period) for k in idx]
    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 1.1],
                          hspace=0.34, wspace=0.25)
    ax_amp = fig.add_subplot(gs[0, :])
    ax_phase = fig.add_subplot(gs[1, :])
    ax_ri = fig.add_subplot(gs[2, :])
    ax_complex = fig.add_subplot(gs[3, 0])
    ax_fold = fig.add_subplot(gs[3, 1])

    amp_dev = 100.0 * (magnitude / amp_ref - 1.0)
    ax_amp.plot(idx, amp_dev, color="0.72", lw=0.55, zorder=1)
    ax_amp.scatter(idx, amp_dev, c=colors, s=13, zorder=2)
    for k in range(0, len(h), a.period):
        ax_amp.axvline(k, color="tab:red", alpha=0.10, lw=0.6)
    for r in range(a.period):
        ax_amp.axhline(100.0 * (amp_mean[r] / amp_ref - 1.0),
                       color=cmap(r), ls="--", lw=1.0,
                       label=f"frame%{a.period}={r}")
    ax_amp.set_title("Per-frame |h[k]| before DC/Blackman/FFT")
    ax_amp.set_ylabel("Deviation from mean (%)")
    ax_amp.legend(ncol=a.period, fontsize=8)
    ax_amp.grid(alpha=0.25)

    ax_phase.plot(idx, phase_resid, color="0.72", lw=0.55, zorder=1)
    ax_phase.scatter(idx, phase_resid, c=colors, s=13, zorder=2)
    for r in range(a.period):
        ax_phase.axhline(phase_mean[r], color=cmap(r), ls="--", lw=1.0)
    ax_phase.set_title("Unwrapped phase after removing its linear trend")
    ax_phase.set_ylabel("Residual phase (deg)")
    ax_phase.grid(alpha=0.25)

    ax_ri.plot(idx, centered.real, lw=0.75, label="Re(h - mean(h))")
    ax_ri.plot(idx, centered.imag, lw=0.75, label="Im(h - mean(h))")
    ax_ri.set_title("Centered complex channel components")
    ax_ri.set_xlabel("TDD frame index (1 frame = 1 ms)")
    ax_ri.set_ylabel("Normalized channel")
    ax_ri.legend()
    ax_ri.grid(alpha=0.25)

    for r in range(a.period):
        sel = idx % a.period == r
        ax_complex.scatter(h[sel].real, h[sel].imag, s=17, alpha=0.65,
                           color=cmap(r), label=str(r))
    ax_complex.set_title("Complex plane, colored by frame modulo")
    ax_complex.set_xlabel("Real(h)")
    ax_complex.set_ylabel("Imag(h)")
    ax_complex.axis("equal")
    ax_complex.legend(title=f"mod {a.period}", ncol=min(a.period, 5), fontsize=8)
    ax_complex.grid(alpha=0.25)

    amp_groups = [100.0 * (ag[:, r] / amp_ref - 1.0) for r in range(a.period)]
    bp = ax_fold.boxplot(amp_groups, positions=np.arange(a.period), widths=0.55,
                         showfliers=False, patch_artist=True)
    for r, box in enumerate(bp["boxes"]):
        box.set(facecolor=cmap(r), alpha=0.45)
    ax_fold.errorbar(np.arange(a.period), 100.0 * (amp_mean / amp_ref - 1.0),
                     yerr=100.0 * amp_sem / amp_ref, fmt="D", color="k",
                     capsize=4, ms=4, label="mean ± SEM")
    ax_fold.axhline(0, color="k", lw=0.7)
    ax_fold.set_title(f"Folded by {a.period} frames ({fs_out/a.period:.1f} Hz)")
    ax_fold.set_xlabel(f"frame index mod {a.period}")
    ax_fold.set_ylabel("|h| deviation (%)")
    ax_fold.set_xticks(np.arange(a.period))
    ax_fold.legend(fontsize=8)
    ax_fold.grid(alpha=0.25)

    fig.suptitle(f"Raw per-frame CAF channel — {base}\n"
                 "h[k] = mean(conj(ch0 reference) · ch1), no Blackman/FFT",
                 fontsize=13)

    if a.out:
        out = a.out
    else:
        stamp = os.path.splitext(base)[0].removeprefix("iq_")
        out = os.path.join(HERE, f"frame_channel_{stamp}.png")
    if not os.path.isabs(out) and os.path.dirname(out) == "":
        out = os.path.join(HERE, out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n图已保存: {out}")
    if show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
