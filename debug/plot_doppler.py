#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 `rt_main --debug` 录下的多普勒谱画出来。默认拿**最近一次**录制。

    /usr/bin/python3 debug/plot_doppler.py                # 最新一次
    /usr/bin/python3 debug/plot_doppler.py --phase        # 加相位面板
    /usr/bin/python3 debug/plot_doppler.py --fmax 200     # 只看 ±200Hz
    /usr/bin/python3 debug/plot_doppler.py doppler_20260815_143012.npz

画法照抄 `current/validate_capture_and_caf.py` 里的 CAF spectrogram
（pcolormesh + jet + 5/95 分位色标 + 相位去旋转），去掉 delay 维——实时链路
每帧只在一个 delay 上算 CAF，没有 delay 轴可扫。

## 两处和老 CAF 图**数值不同**，别拿去对着读

1. 老脚本画的是 `10*log10(|S|)`（幅度的 dB），这里是 `10*log10(|S|²)`（功率的
   dB，也就是 `20*log10|S|`）。同一份数据，这里的 dB 读数是老图的 2 倍，
   **形状和对比度完全一样**（色标是分位数自适应的）。
2. 老脚本对整段连续流做 STFT；实时链路是 TDD 门控后每帧一个点的 1kHz 流，
   频率轴天然只到 ±500Hz，且不含 OFF 段引入的 1kHz 方波调制。
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def newest_npz(d: str) -> str | None:
    files = sorted(glob.glob(os.path.join(d, "doppler_*.npz")))
    return files[-1] if files else None


def main() -> int:
    p = argparse.ArgumentParser(description="画 debug 录下的多普勒图")
    p.add_argument("npz", nargs="?", default=None,
                   help="npz 路径（省略 = debug/ 下最新的一个）")
    p.add_argument("--fmax", type=float, default=None,
                   help="只画 ±fmax Hz（默认全量程）")
    p.add_argument("--phase", action="store_true", help="加一栏相位（去旋转后）")
    p.add_argument("--keep-priming", action="store_true",
                   help="保留环形缓冲灌注期的帧（默认丢掉，那几帧的谱不可信）")
    p.add_argument("--vrange", type=float, nargs=2, default=None,
                   metavar=("VMIN", "VMAX"), help="手动指定色标 dB 范围")
    p.add_argument("--out", type=str, default=None, help="PNG 输出路径")
    p.add_argument("--no-show", action="store_true", help="只存图不弹窗")
    a = p.parse_args()

    path = a.npz or newest_npz(HERE)
    if not path:
        print(f"debug/ 下没有 doppler_*.npz。先跑一次：\n"
              f"  cd RealtimeISAC/realtime && "
              f"/usr/bin/python3 rt_main.py --duration 30 --debug", file=sys.stderr)
        return 1
    if not os.path.isabs(path):
        path = os.path.join(HERE, path) if not os.path.exists(path) else path

    # 弹窗只在「有 DISPLAY 且 stdout 是终端」时开——被管道接走时弹窗会卡死脚本
    show = (not a.no_show) and bool(os.environ.get("DISPLAY")) and sys.stdout.isatty()
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(path)
    meta = json.loads(str(d["meta_json"]))
    spec, t, freqs = d["spec"], d["t"], d["freqs"]
    ready, present, n_hits, peak_hz = d["ready"], d["present"], d["n_hits"], d["peak_hz"]
    max_run = d["max_run"] if "max_run" in d.files else None   # 老录制没这一项
    pw = d["pw"] if "pw" in d.files else None      # 老录制没这一项
    e_db = d["e_db"] if "e_db" in d.files else None
    floor_db = d["floor_db"] if "floor_db" in d.files else None

    base = os.path.basename(path)
    print(f"=== {base} ===")
    print(f"采集 {meta['t_start_str']}   {len(t)} 帧 / {t[-1]:.1f}s   "
          f"fs={meta['sample_rate']/1e6:.0f}MHz  fc={meta['center_freq']/1e9:.3f}GHz  "
          f"gain={meta['gain']:.0f}dB")
    print(f"窗={meta['window_sec']*1e3:.0f}ms  步进={meta['step_sec']*1e3:.0f}ms  "
          f"FFT={meta['nfft']}  bin={meta['bin_hz']:.2f}Hz  "
          f"量程=±{meta['fs_out']/2:.0f}Hz")
    if meta.get("sync_locked"):
        print(f"TDD锁定 contrast={meta['sync_contrast']:.2f} "
              f"占空={meta['sync_duty']:.1%} phase={meta['sync_phase']} "
              f"积分={meta['sync_n_integrate']}样本")
    else:
        print("⚠️ TDD 未锁定（自由运行），这份数据没有门控")
    print(f"Rx功率 {meta.get('power_dbfs')} dBFS   UHD {meta.get('uhd_errors')}")
    if meta.get("truncated"):
        print("⚠️ 录制缓冲曾满，尾部帧丢失")

    if not a.keep_priming:
        keep = ready.astype(bool)
        if keep.sum() < 2:
            print("⚠️ 可用帧不足，改为保留灌注期", file=sys.stderr)
        else:
            spec, t, present = spec[keep], t[keep], present[keep]
            n_hits, peak_hz = n_hits[keep], peak_hz[keep]
            if max_run is not None:
                max_run = max_run[keep]
            if pw is not None:
                pw = pw[keep]
            if e_db is not None:
                e_db, floor_db = e_db[keep], floor_db[keep]
            print(f"丢掉灌注期 {int((~keep).sum())} 帧，剩 {len(t)} 帧")

    fmask = np.ones(len(freqs), dtype=bool) if a.fmax is None \
        else (np.abs(freqs) <= a.fmax)
    f_axis = freqs[fmask]
    S = spec[:, fmask].T                       # (n_freq, n_time)，和 CAF 一致
    Sxx_db = 10.0 * np.log10(np.abs(S) ** 2 + 1e-20)

    if a.vrange:
        vmin, vmax = a.vrange
    else:
        vmin, vmax = np.percentile(Sxx_db, 5), np.percentile(Sxx_db, 95)

    has_pw = pw is not None and np.any(pw != 0)
    has_e = e_db is not None and np.any(e_db != 0)
    heights = ([3.0] + ([3.0] if a.phase else []) + ([1.6] if has_e else [])
               + [1.4] + ([1.0] if has_pw else []))
    n_rows = len(heights)
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, sum(heights) * 1.3), sharex=True,
                             gridspec_kw={"height_ratios": heights})
    ax_amp = axes[0]
    mesh = ax_amp.pcolormesh(t, f_axis, Sxx_db, shading="auto", cmap="jet",
                             vmin=vmin, vmax=vmax)
    ax_amp.set_ylabel("Doppler Frequency (Hz)")
    ax_amp.set_title(f"Doppler amplitude — {base}  "
                     f"(Tw={meta['window_sec']}s, step={meta['step_sec']}s, "
                     f"bin={meta['bin_hz']:.1f}Hz)")
    fig.colorbar(mesh, ax=ax_amp, label="Relative Power (dB)")
    g = meta.get("dc_guard_hz", 0.0)
    for s in (-g, g):
        ax_amp.axhline(s, color="w", ls="--", lw=0.8, alpha=0.6)
    # 图上一律用英文：默认字体没有 CJK 字形，中文会画成一堆豆腐块还刷警告
    ax_amp.text(t[0], g, " DC guard", color="w", va="bottom", fontsize=8)

    if a.phase:
        ax_ph = axes[1]
        # 去旋转：和老 CAF 脚本同一处理——扣掉 exp(j2πft) 这个纯几何相位，
        # 剩下的才是目标本身的相位演化。
        derot = np.exp(-1j * 2 * np.pi * f_axis.reshape(-1, 1) * t.reshape(1, -1))
        ph = np.rad2deg(np.angle(S * derot))
        mesh_p = ax_ph.pcolormesh(t, f_axis, ph, shading="auto", cmap="hsv",
                                  vmin=-180, vmax=180)
        ax_ph.set_ylabel("Doppler Frequency (Hz)")
        ax_ph.set_title("Doppler phase (derotated)")
        fig.colorbar(mesh_p, ax=ax_ph, label="Phase (deg)")

    # 末栏：峰值多普勒轨迹 + 判决。验证「多普勒对不对」时这栏最直接——
    # 人朝天线走 = 正频率，走开 = 负频率，静止 = 只剩 DC 附近。
    # 栏序: 幅度 / (相位) / (能量判据) / 判决 / (Rx功率)
    row = 2 if a.phase else 1
    if has_e:
        # 这一栏是**主判据**：E 超过 floor+margin 就算这帧有动静。
        # 判决为什么在某一刻翻转，看这里最直接。
        ax_e = axes[row]
        row += 1
        mg = meta.get("energy_margin_db", 10.0)
        ax_e.plot(t, e_db, lw=1.0, color="tab:green", label="E (non-DC energy)")
        ax_e.plot(t, floor_db, lw=1.0, ls="--", color="gray", label="baseline (p10+ratchet)")
        ax_e.plot(t, floor_db + mg, lw=1.2, ls=":", color="tab:red",
                  label=f"threshold (+{mg:.0f}dB)")
        ax_e.fill_between(t, floor_db + mg, e_db, where=e_db > floor_db + mg,
                          color="tab:green", alpha=0.18)
        if present.any():
            lo_, hi_ = min(e_db.min(), floor_db.min()), e_db.max()
            ax_e.fill_between(t, lo_, hi_, where=present, color="tab:red",
                              alpha=0.10, step="mid")
        ax_e.set_ylabel("Energy (dB)")
        ax_e.grid(alpha=0.3)
        ax_e.legend(loc="upper right", fontsize=7, ncol=3)
        fig.colorbar(mesh, ax=ax_e).ax.set_visible(False)

    ax_pk = axes[row]
    if present.any():
        ax_pk.fill_between(t, f_axis[0], f_axis[-1], where=present,
                           color="tab:red", alpha=0.12, step="mid",
                           label="decision: PRESENT")
    # peak_hz 在"本帧一个 bin 都没命中"时被记成 0，直接画会在 0Hz 堆一条假线，
    # 看起来像"总有个静止目标"。没命中的帧不画。
    hit = n_hits > 0
    ax_pk.plot(t[hit], peak_hz[hit], ".", ms=2.5, color="tab:blue",
               label="peak Doppler (frames with CFAR hits)")
    ax_pk.set_ylabel("Peak (Hz)")
    ax_pk.set_ylim(f_axis[0], f_axis[-1])
    ax_pk.grid(alpha=0.3)
    ax_pk.legend(loc="upper left", fontsize=8)
    ax2 = ax_pk.twinx()
    ax2.plot(t, n_hits, lw=0.7, color="tab:orange", alpha=0.4, label="total hit bins")
    if max_run is not None:
        ax2.plot(t, max_run, lw=1.0, color="tab:red", alpha=0.8, label="longest run")
        ax2.axhline(meta.get("cfar_min_run", 5), color="tab:red", ls=":", lw=1.0)
    else:
        ax2.axhline(meta.get("cfar_min_run", meta.get("cfar_min_bins", 5)),
                    color="tab:orange", ls=":", lw=1.0)
    ax2.set_ylabel("CFAR bins (orange=total, red=longest run)", color="tab:orange")
    ax2.legend(loc="upper right", fontsize=7)
    # 给下面几栏也挂色标再隐藏：只为占住右侧那条宽度，让各栏左右边界对齐
    fig.colorbar(mesh, ax=ax2).ax.set_visible(False)

    if has_pw:
        ax_pw = axes[-1]
        for c in range(pw.shape[1]):
            ax_pw.plot(t, pw[:, c], lw=0.9, label=f"ch{c}")
        ax_pw.set_ylabel("Rx (dBFS)")
        ax_pw.grid(alpha=0.3)
        ax_pw.legend(loc="upper left", fontsize=8, ncol=2)
        fig.colorbar(mesh, ax=ax_pw).ax.set_visible(False)
    axes[-1].set_xlabel("Time (s)")

    out = a.out or os.path.join(HERE, base.replace(".npz", ".png"))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"🖼  {out}")
    frac = float(present.mean()) if len(present) else 0.0
    run_str = f"   最长连续游程中位数 {np.median(max_run):.0f} bin" if max_run is not None else ""
    print(f"判决「有人」占比 {frac:.1%}   峰值多普勒中位数 "
          f"{np.median(peak_hz):+.1f}Hz   CFAR 命中中位数 {np.median(n_hits):.0f} bins"
          f"{run_str}")
    if show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
