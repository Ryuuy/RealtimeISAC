#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""realtime 对比 Rx1(参考/ch0) 和 Rx2(感知/ch1) 各自的 FFT，找 200Hz 伪影从哪来。

不切 TDD 帧、不逐帧塌缩 —— 直接从 rt_iqdump 存的 npz 里各取一段（默认 0.2s）
连续原始 IQ，reference(ch0) 和 sensing(ch1) 分别打两种谱：

  1. 复基带直接 FFT（±fmax，默认 ±3000Hz）—— 看频率成分本身
  2. |IQ| 包络的 FFT（同样 ±fmax）—— 幅度调制会在这里暴露成谱线，
     200Hz 这种"调幅"类伪影通常在这张图里比复基带 FFT 更干净地跳出来

两路各出一张图（同一张 figure 里上下对比），方便一眼看出 200Hz 梳子
是两路都有（→ 共模，LO/参考钟一类）还是只在某一路（→ 该路自己的
前端/路径问题）。

    /usr/bin/python3 debug/compare_ref_sensing_fft.py                    # 最新 iq_*.npz，前 0.2s
    /usr/bin/python3 debug/compare_ref_sensing_fft.py iq_xxx.npz --sec 0.2 --start 1.0
    /usr/bin/python3 debug/compare_ref_sensing_fft.py --fmax 3000 --window hann --no-show
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMB = (200.0, 400.0, 600.0, 800.0, 1000.0, 2000.0, 3000.0)


def newest_iq(d):
    f = sorted(glob.glob(os.path.join(d, "iq_*.npz")))
    return f[-1] if f else None


def make_window(name, n):
    if name == "hann":
        return np.hanning(n)
    if name == "blackman":
        return np.blackman(n)
    return np.ones(n)


def fft_db(x, fs, win):
    n = len(x)
    # complex64/float32 全程：5s@10MHz 是 5000万点，complex128 一份就是 800MB，
    # 这函数里一度同时活着好几份同尺寸数组，翻倍很容易把这台机器（7.4GB内存）
    # 撑爆 OOM-killed。complex64 对这种"看谱形/找峰"的用途精度绰绰有余。
    x = ((x - x.mean()) * win).astype(np.complex64)
    X = np.fft.fftshift(np.fft.fft(x)).astype(np.complex64)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs)).astype(np.float32)
    mag_db = (20.0 * np.log10(np.abs(X) / n + 1e-20)).astype(np.float32)
    return freqs, mag_db


def report(name, freqs, mag_db, fs, n):
    df = fs / n
    noise = np.median(mag_db[np.abs(freqs) < min(5000.0, freqs.max())])
    print(f"  [{name}]  噪底中位 {noise:.1f} dB")
    for tgt in COMB:
        if tgt > freqs.max():
            continue
        sel = np.abs(np.abs(freqs) - tgt) <= 3 * df
        pk = mag_db[sel].max()
        print(f"    {tgt:6.0f} Hz : {pk:7.1f} dB   ({pk - noise:+.1f} dB 高于噪底)")


def analyze_and_plot(iq, fs, tag, out, fmax=3000.0, window="none", show=False,
                      extra_title="", env_ms=50.0, abs_only=False):
    """核心分析：iq=(2, n*2) int16 交织，ch0=reference / ch1=sensing。

    打印两路各自的 IQ 复基带 FFT / |IQ| 包络 FFT 在 COMB 频点的峰值，
    再出一张 2x2 对比图存到 out。供 npz 回放（本文件 main）和现采
    （capture_and_compare_fft.py）共用，避免两处各写一份分析逻辑。

    env_ms: 顶上时间轴能量图只画开头这么多毫秒（默认 50ms），跟 FFT 用的总时长
    （n/fs，可以是几秒）分开算——时间轴图固定抽到 ~6000 点，如果拿整个分析时长
    去抽，时长一长（比如 5s）每点就摊到几百微秒，比 1ms 的 TDD 周期还宽，
    方波直接被抽没；固定只看开头一小段、密抽，才能看清 TDD 突发本身。

    abs_only: 只算 |IQ| 包络 FFT，跳过复基带 FFT——省一半 FFT + 一份大数组，
    5s@10MHz 全量跑的时候明显快（复基带 FFT 对判断 200Hz 这类幅度调制没有
    额外信息，本来就用不上）。
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = iq.shape[1] // 2
    win = make_window(window, n).astype(np.float32)
    print(f"fs={fs/1e6:.3f}MHz  {n} 样本 = {n/fs*1e3:.1f}ms  →  频率分辨率 {fs/n:.3f} Hz")

    # 时域包络图只取开头 env_ms 那一小段，独立于 FFT 用的整段长度 n——
    # 5s@10MHz 一份 complex64 x 就有 400MB，两路 + FFT 中间量同时活着很容易 OOM，
    # 每路算完就地丢掉 x/env，只留画图需要的小数组和 FFT 结果。
    env_n = min(n, max(1, int(round(env_ms * 1e-3 * fs))))
    env_dec = max(1, env_n // 6000)   # 控制在 ~6000 点，太密看不清

    chans = [(0, "ch0 reference (RFA)"), (1, "ch1 sensing (RX2)")]
    data = {}
    for ch, label in chans:
        seg = np.asarray(iq[ch], dtype=np.float32)
        x = (seg[0::2] + 1j * seg[1::2]).astype(np.complex64)        # 复基带
        del seg
        m = (env_n // env_dec) * env_dec
        env_trace = np.abs(x[:m]).reshape(-1, env_dec).max(axis=1).astype(np.float32)
        freqs_iq = mag_iq = None
        if not abs_only:
            freqs_iq, mag_iq = fft_db(x, fs, win)
        env = np.abs(x).astype(np.complex64)
        del x
        freqs_env, mag_env = fft_db(env, fs, win)
        del env
        data[ch] = dict(label=label, env_trace=env_trace, freqs_iq=freqs_iq, mag_iq=mag_iq,
                         freqs_env=freqs_env, mag_env=mag_env)
        print(f"\n--- {label} ---")
        if not abs_only:
            report("IQ 复基带 FFT", freqs_iq, mag_iq, fs, n)
        report("|IQ| 包络 FFT", freqs_env, mag_env, fs, n)

    n_rows = 2 if abs_only else 3
    fig = plt.figure(figsize=(15, 8 if abs_only else 12))
    hr = [0.7, 1.0] if abs_only else [0.7, 1.0, 1.0]
    gs = fig.add_gridspec(n_rows, 2, height_ratios=hr, hspace=0.4, wspace=0.25,
                          left=0.06, right=0.98, top=0.88 if abs_only else 0.90, bottom=0.08)
    ax_env = fig.add_subplot(gs[0, :])

    # 时域包络（dB），两路叠在一起：直接肉眼确认 TDD 突发在不在、锁得干不干净——
    # 下面两行是频域图，天生看不出"这段时间功率高不高"这种时序结构。
    # env_trace 已经在上面每路各自算完就抽取好了（只取开头 env_ms，见上面注释）。
    tt = np.arange(0, env_n - env_dec + 1, env_dec) / fs * 1e3
    for ch, c in ((0, "tab:blue"), (1, "tab:orange")):
        env_trace = data[ch]["env_trace"]
        ax_env.plot(tt[:len(env_trace)], 20 * np.log10(env_trace + 1e-9), lw=0.6, color=c,
                    label=data[ch]["label"])
    ax_env.set_title(f"time-domain |IQ| magnitude, first {env_ms:.0f}ms of the analyzed segment, "
                      f"max per bin (20log10|x| = 10log10 power) "
                      "— is TDD actually bursting here?", fontsize=10)
    ax_env.set_xlabel("Time (ms), relative to start of analyzed segment")
    ax_env.set_ylabel("dB re raw ADC counts")
    ax_env.legend(loc="upper right", fontsize=8); ax_env.grid(alpha=0.3)

    win_txt = "" if window == "none" else f" · {window}"
    env_row = 1 if abs_only else 2
    for col, (ch, label) in enumerate(chans):
        dd = data[ch]
        mask_env = np.abs(dd["freqs_env"]) <= fmax

        if not abs_only:
            mask_iq = np.abs(dd["freqs_iq"]) <= fmax
            ax = fig.add_subplot(gs[1, col])
            ax.plot(dd["freqs_iq"][mask_iq], dd["mag_iq"][mask_iq], lw=0.6, color="tab:blue")
            ax.set_title(f"{label}\nIQ complex FFT{win_txt}", fontsize=10)
            ax.set_ylabel("Magnitude (dB, /N)")
            ax.axvline(0, color="k", ls="--", lw=0.7, alpha=0.5)
            for s in (-1, 1):
                ax.axvline(s * 200.0, color="tab:green", ls=":", lw=0.9, alpha=0.8)
            for k in range(1, int(fmax // 1000) + 1):
                for s in (-1, 1):
                    ax.axvline(s * 1000 * k, color="tab:red", ls=":", lw=0.6, alpha=0.5)
            ax.set_xlim(-fmax, fmax)
            ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[env_row, col])
        ax.plot(dd["freqs_env"][mask_env], dd["mag_env"][mask_env], lw=0.6, color="tab:orange")
        ax.set_title(f"{label}\n|IQ| envelope FFT{win_txt}", fontsize=10)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB, /N)")
        ax.axvline(0, color="k", ls="--", lw=0.7, alpha=0.5)
        for s in (-1, 1):
            ax.axvline(s * 200.0, color="tab:green", ls=":", lw=0.9, alpha=0.8)
        for k in range(1, int(fmax // 1000) + 1):
            for s in (-1, 1):
                ax.axvline(s * 1000 * k, color="tab:red", ls=":", lw=0.6, alpha=0.5)
        ax.set_xlim(-fmax, fmax)
        ax.grid(alpha=0.3)

    fig.suptitle(f"{tag}   reference vs sensing raw FFT   "
                 f"[{n/fs*1e3:.0f}ms, {n} pts, +-{fmax:.0f}Hz]{extra_title}"
                 f"   green=200Hz  red=1kHz harmonics", fontsize=11)
    # 不用 tight_layout：上面是手工 gridspec（含 hspace/margins），tight_layout 会警告
    # 且可能画歪，margins 已经在 add_gridspec 里手工留好了。

    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n\U0001f5bc  {out}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="对比 reference(ch0) / sensing(ch1) 的原始 FFT")
    p.add_argument("npz", nargs="?", default=None, help="iq_*.npz（省略=debug/下最新）")
    p.add_argument("--sec", type=float, default=0.2, help="取多少秒（默认 0.2）")
    p.add_argument("--start", type=float, default=0.0, help="从第几秒开始（默认 0）")
    p.add_argument("--fmax", type=float, default=3000.0, help="只画 ±fmax Hz（默认 3000）")
    p.add_argument("--window", choices=["none", "hann", "blackman"], default="none",
                   help="加窗（默认 none，真·直接 FFT）")
    p.add_argument("--env-ms", type=float, default=50.0,
                   help="顶上时间轴能量图只画开头多少毫秒（默认 50ms，跟 --sec 分开控制，"
                        "--sec 拉长不会把这张图的方波抽没）")
    p.add_argument("--abs-only", action="store_true",
                   help="只算 |IQ| 包络 FFT，跳过复基带 FFT（省一半算力/内存，全量5秒时明显更快）")
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
    seg = iq[:, s0 * 2:(s0 + n) * 2]

    print(f"=== {base} ===")
    out = a.out or os.path.join(HERE, base.replace("iq_", "cmpfft_ref_sensing_").replace(".npz", ".png"))
    analyze_and_plot(seg, fs, base, out, fmax=a.fmax, window=a.window, show=show,
                      extra_title=f"  [{a.start:.2f}-{a.start + n/fs:.2f}s]", env_ms=a.env_ms,
                      abs_only=a.abs_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
