#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""上行切窗 vs 下行切窗 —— CAF 和频谱除法各跑一遍。

RealtimeISAC 现在的 TddSync 是**折叠参考通道(ch0)的功率包络**找 ON 段，切出来
的是**上行时隙**（slave 发、ch0 强）。这里加一路：**折叠感知通道(ch1)的功率
包络**找它最强的时隙，切出来的是**下行时隙**（master 发、ch1 强）。两个窗在
1ms 周期里位置不同。

同一份原始 IQ，两种窗各自：
  CAF      : dec = Σ conj(ch0)·ch1 / (‖ch0‖·‖ch1‖)
  频谱除法 : H(f) = FFT(ch1)/FFT(ch0)，median(Re)+j·median(Im)
后端都一样（EMA 去 DC + 200帧/0.2s 滑动 Blackman STFT，256 补零）。

出图 2 行 × 3 列：
  行 = CAF / 频谱除法
  列 = [上行窗谱] [下行窗谱] [时间积分谱：上行 vs 下行叠一起]

    /usr/bin/python3 debug/compare_uplink_vs_downlink.py [iq_xxx.npz] [--fmax 300] [--no-show]
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_RING, HOP, NFFT, FS_OUT = 200, 20, 256, 1000.0
DC_ALPHA, DC_GUARD_HZ = 0.01, 20.0
SYNC_DECIM = 10
EDGE_GUARD_SEC = 10e-6
LINE_HZ = (50.0, 100.0, 150.0, 200.0)


def newest_iq(dirp):
    f = sorted(glob.glob(os.path.join(dirp, "iq_*.npz")))
    return f[-1] if f else None


def longest_run(on):
    """环形最长连续 True 段 -> (起点, 长度)。复刻 rt_sync._longest_run。"""
    n = on.size
    if on.all() or not on.any():
        return 0, 0
    d = np.concatenate([on, on])
    edges = np.flatnonzero(np.diff(np.concatenate(([False], d, [False]))))
    starts, ends = edges[0::2], edges[1::2]
    k = int(np.argmax(ends - starts))
    return int(starts[k] % n), int(min(ends[k] - starts[k], n))


def _fold_power(iq, ch, n_per, n_fr_fold, decim):
    n_per_dec = n_per // decim
    i_comp = np.asarray(iq[ch, 0:2 * n_fr_fold * n_per:2 * decim], dtype=np.int64)
    i_comp = i_comp[:n_fr_fold * n_per_dec]
    return (i_comp * i_comp).reshape(n_fr_fold, n_per_dec).mean(axis=0)


def find_window(iq, ch, fs, n_per, n_fr_fold, decim=SYNC_DECIM, dominant_over=None):
    """折叠 ch 通道功率包络，找最强连续时隙 -> (phase, n_on, contrast, duty, profile)。

    复刻 rt_sync.TddSync.update 的判据：几何中点阈值 + 环形最长连续段 + 边沿内缩。
    dominant_over 给了另一个通道号时，额外要求本通道功率 > 那个通道
    （下行时隙用：ch1 收到上行突发也很强，只折 ch1 会锁到上行段上，
    加"ch1 比 ch0 强"才能把 master 下行那个 ~80µs 小时隙单独抠出来）。
    """
    n_per_dec = n_per // decim
    profile = _fold_power(iq, ch, n_per, n_fr_fold, decim)

    lo = float(np.percentile(profile, 10))
    hi = float(np.percentile(profile, 90))
    thr = np.sqrt(max(lo, 1e-30) * max(hi, 1e-30))
    on = profile >= thr
    if dominant_over is not None:
        other = _fold_power(iq, dominant_over, n_per, n_fr_fold, decim)
        on &= profile > other
    start, length = longest_run(on)
    if not (0.02 * n_per_dec <= length <= 0.98 * n_per_dec):
        return None
    idx = (start + np.arange(length)) % n_per_dec
    on_sum = float(profile[idx].sum())
    on_mean = on_sum / length
    off_mean = (float(profile.sum()) - on_sum) / max(n_per_dec - length, 1)
    contrast = (on_mean - off_mean) / (on_mean + off_mean) if on_mean + off_mean > 0 else 0.0

    guard = int(round(EDGE_GUARD_SEC * fs))
    n_on = max(1, length * decim - 2 * guard)
    phase = (start * decim + guard) % n_per
    return phase, min(n_on, n_per), contrast, length / n_per_dec, profile


def ema_dc_remove(z, hop=HOP, alpha=DC_ALPHA):
    z = z.astype(np.complex128).copy()
    dc, dc_n = 0.0 + 0.0j, 0
    for i0 in range(0, len(z), hop):
        blk = slice(i0, i0 + hop)
        m = z[blk].mean()
        z[blk] -= dc
        dc_n += 1
        a = max(alpha, 1.0 / dc_n)
        dc = (1.0 - a) * dc + a * m
    return z


def stft_short(z):
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


def per_frame(iq, starts, n_int):
    """逐帧 CAF + 频谱除法。"""
    caf = np.empty(len(starts), np.complex128)
    sdv = np.empty(len(starts), np.complex128)
    for k, s in enumerate(starts):
        seg0 = np.asarray(iq[0, s * 2:(s + n_int) * 2], np.float64)
        seg1 = np.asarray(iq[1, s * 2:(s + n_int) * 2], np.float64)
        a0 = seg0[0::2] + 1j * seg0[1::2]
        b0 = seg1[0::2] + 1j * seg1[1::2]
        p0 = np.vdot(a0, a0).real
        p1 = np.vdot(b0, b0).real
        caf[k] = np.vdot(a0, b0) / (np.sqrt(p0) * np.sqrt(p1)) if p0 > 0 and p1 > 0 else 0
        tx = a0 - a0.mean()
        rx = b0 - b0.mean()
        TX = np.fft.fftshift(np.fft.fft(tx))
        RX = np.fft.fftshift(np.fft.fft(rx))
        atx = np.abs(TX)
        fl = max(atx.max() * 1e-9, 1e-12)
        H = RX / np.where(atx < fl, fl, TX)
        sdv[k] = np.median(H.real) + 1j * np.median(H.imag)
    return caf, sdv


def line_excess(g, f_axis):
    base = np.abs(f_axis) >= DC_GUARD_HZ
    for f0 in LINE_HZ:
        base &= np.abs(np.abs(f_axis) - f0) >= 8.0
    floor = np.median(g[base])
    out = {}
    for f0 in LINE_HZ:
        m = np.abs(np.abs(f_axis) - f0) <= 4.0
        out[f0] = float(g[m].max() - floor) if m.any() else np.nan
    return floor, out


def main() -> int:
    p = argparse.ArgumentParser(description="上行切窗 vs 下行切窗")
    p.add_argument("npz", nargs="?", default=None)
    p.add_argument("--fmax", type=float, default=300.0)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()

    path = a.npz or newest_iq(HERE)
    if not path:
        print("没有 iq_*.npz。先采：cd realtime && /usr/bin/python3 rt_iqdump.py --ms 5000",
              file=sys.stderr)
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
    n_per = int(d["n_period"])
    total = iq.shape[1] // 2
    base = os.path.basename(path)
    n_fr_fold = min(total // n_per, 2000)
    print(f"=== {base} ===  {total/fs*1e3:.0f}ms x2ch  fs={fs/1e6:.0f}MHz  n_per={n_per}  "
          f"（折叠 {n_fr_fold} 个周期找窗）")

    win_up = find_window(iq, 0, fs, n_per, n_fr_fold)   # 折叠 ch0 -> 上行
    # 下行：折 ch1，且要求 ch1 比 ch0 强（把 master 下行小时隙抠出来）。
    # 抠不出来时退回"纯折 ch1 最强段"（可能就等于上行段）。
    win_dn = (find_window(iq, 1, fs, n_per, n_fr_fold, dominant_over=0)
              or find_window(iq, 1, fs, n_per, n_fr_fold))
    if win_up is None:
        print("⚠️ ch0 折叠找不到 ON 段（TX 没开？）", file=sys.stderr)
        return 1
    if win_dn is None:
        print("⚠️ ch1 折叠找不到清晰时隙（感知通道堵住了 / 下行太弱？）——"
              "下行那一路用 ch0 的窗兜底，结果仅供参考", file=sys.stderr)
        win_dn = win_up

    ph_u, non_u, ct_u, dt_u, prof_u = win_up
    ph_d, non_d, ct_d, dt_d, prof_d = win_dn
    print(f"上行窗(折ch0): phase={ph_u} ({ph_u/fs*1e6:.0f}µs)  n_on={non_u} "
          f"({non_u/fs*1e6:.0f}µs)  contrast={ct_u:.2f}  duty={dt_u:.1%}")
    print(f"下行窗(折ch1): phase={ph_d} ({ph_d/fs*1e6:.0f}µs)  n_on={non_d} "
          f"({non_d/fs*1e6:.0f}µs)  contrast={ct_d:.2f}  duty={dt_d:.1%}")

    results = {}
    for tag, (ph, non) in (("up", (ph_u, non_u)), ("dn", (ph_d, non_d))):
        n_fr = (total - ph - non) // n_per
        starts = ph + np.arange(n_fr) * n_per
        print(f"[{tag}] 逐帧（{n_fr} 帧 @ {FS_OUT:.0f}Hz）…", flush=True)
        t0 = time.time()
        caf, sdv = per_frame(iq, starts, non)
        print(f"  用时 {time.time()-t0:.1f}s")
        t_c, freqs, S_c = stft_short(ema_dc_remove(caf))
        t_s, _, S_s = stft_short(ema_dc_remove(sdv))
        results[tag] = dict(t=t_c, freqs=freqs, caf=S_c, sdv=S_s)

    fmask = np.abs(freqs) <= a.fmax
    f_axis = freqs[fmask]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9),
                             gridspec_kw={"width_ratios": [3, 3, 1.4]})
    method_rows = [("CAF", "caf"), ("Spectrum division (median)", "sdv")]
    col_titles = ["Uplink-gated window (fold ch0)", "Downlink-gated window (fold ch1)"]
    print("\n时间积分谱 TX 线相对本底 (dB):")
    for r, (mname, mkey) in enumerate(method_rows):
        gset = {}
        for c, tag in enumerate(("up", "dn")):
            S = results[tag][mkey][:, fmask]
            D = 10 * np.log10(S.T + 1e-20)
            vmin, vmax = np.percentile(D, 5), np.percentile(D, 95)
            ax = axes[r, c]
            ax.pcolormesh(results[tag]["t"], f_axis, D, shading="auto", cmap="jet",
                          vmin=vmin, vmax=vmax)
            ax.set_ylim(-a.fmax, a.fmax)
            for sgn in (-DC_GUARD_HZ, DC_GUARD_HZ):
                ax.axhline(sgn, color="w", ls="--", lw=0.6, alpha=0.5)
            ax.set_ylabel(f"{mname}\nDoppler (Hz)" if c == 0 else "")
            if r == 0:
                ax.set_title(col_titles[c])
            if r == 1:
                ax.set_xlabel("Time (s)")
            g = 10 * np.log10(S.mean(axis=0) + 1e-20)
            gset[tag] = g
            fl, ex = line_excess(g, f_axis)
            print(f"  {mname:26s} {tag}  floor={fl:6.1f}  "
                  + " ".join(f"{int(f)}:{ex[f]:+5.1f}" for f in LINE_HZ))

        axg = axes[r, 2]
        for tag, col in (("up", "tab:blue"), ("dn", "tab:red")):
            axg.plot(gset[tag] - gset[tag].max(), f_axis, lw=1.0, color=col, label=tag)
        for f0 in LINE_HZ:
            for sgn in (-1, 1):
                axg.axhline(sgn * f0, color="0.7", lw=0.5, ls=":")
        axg.set_ylim(-a.fmax, a.fmax)
        axg.set_xlabel("dB (norm)")
        axg.set_title("time-integrated\nup(blue) vs dn(red)")
        axg.grid(alpha=0.3)
        axg.legend(fontsize=8)

    fig.suptitle(f"Uplink vs Downlink gating — {base}", fontsize=13)
    out = a.out or os.path.join(HERE, base.replace("iq_", "updn_").replace(".npz", ".png"))
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n🖼  {out}")

    npz_out = out.replace(".png", ".npz")
    np.savez(npz_out, freqs=freqs,
             t_up=results["up"]["t"], t_dn=results["dn"]["t"],
             S_caf_up=results["up"]["caf"].astype(np.float32),
             S_sdv_up=results["up"]["sdv"].astype(np.float32),
             S_caf_dn=results["dn"]["caf"].astype(np.float32),
             S_sdv_dn=results["dn"]["sdv"].astype(np.float32),
             phase_up=ph_u, n_on_up=non_u, phase_dn=ph_d, n_on_dn=non_d,
             prof_up=prof_u, prof_dn=prof_d, fs=fs, n_period=n_per, src=base)
    print(f"💾 {npz_out}")
    if show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
