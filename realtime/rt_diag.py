#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性诊断：TDD 结构到底长什么样。**不落盘**，只在内存里留 100ms。

rt_sync 的盲同步假设「周期 1ms / 占空 50%」。这个假设一旦错了，折叠出来的
包络就是糊的，contrast 会掉到 0.3~0.5 这种「看起来锁上了其实没锁」的区间。
本脚本不做任何假设，直接从数据里量：

1. 每通道的 dBFS / 峰值 / 是否削波
2. 功率包络的自相关 -> **真实突发周期**（不预设 1ms）
3. 按实测周期折叠 -> ASCII 包络图 + 真实占空比 + ON/OFF 比
4. 按 1ms 折叠做对照，解释 rt_sync 为什么给出那个 contrast

    cd RealtimeISAC/realtime
    /usr/bin/python3 rt_diag.py
    /usr/bin/python3 rt_diag.py --fs 30 --gain 40
"""

import argparse
import sys

import numpy as np

from rt_config import RtConfig
from rt_dsp import to_db

BINS_US = 1.0          # 包络分辨率：1µs 一个格


def envelope(ch_i16: np.ndarray, fs: float, bin_us: float = BINS_US):
    """(n*2,) int16 交织 -> 每 bin_us 微秒一格的平均功率（归一化到满量程）。"""
    dec = max(1, int(round(fs * bin_us * 1e-6)))
    i = ch_i16[0::2].astype(np.int64)
    q = ch_i16[1::2].astype(np.int64)
    p = i * i + q * q
    n = (p.size // dec) * dec
    return p[:n].reshape(-1, dec).mean(axis=1) / (32767.0 ** 2)


def find_period(env: np.ndarray, bin_us: float, lo_us: float, hi_us: float):
    """功率包络自相关找周期。返回 (最佳周期µs, 归一化相关峰, 前几名列表)。"""
    x = env - env.mean()
    n = 1 << int(np.ceil(np.log2(len(x) * 2)))
    F = np.fft.rfft(x, n)
    ac = np.fft.irfft(F * np.conj(F), n)[:len(x)]
    if ac[0] <= 0:
        return None, 0.0, []
    ac = ac / ac[0]
    lo, hi = int(lo_us / bin_us), min(int(hi_us / bin_us), len(ac) - 1)
    if hi <= lo:
        return None, 0.0, []
    seg = ac[lo:hi]
    order = np.argsort(seg)[::-1]
    # 取互相隔开的前几个峰，避免同一个峰的邻点刷屏
    tops, used = [], []
    for k in order:
        lag = (lo + int(k)) * bin_us
        if any(abs(lag - u) < 20.0 for u in used):
            continue
        tops.append((lag, float(seg[k])))
        used.append(lag)
        if len(tops) >= 5:
            break
    return tops[0][0], tops[0][1], tops


def fold(env: np.ndarray, period_us: float, bin_us: float):
    """按周期折叠平均。周期不是整数格时用最近整数格（诊断够用）。"""
    n_per = max(2, int(round(period_us / bin_us)))
    n_cyc = len(env) // n_per
    if n_cyc < 2:
        return None
    return env[:n_cyc * n_per].reshape(n_cyc, n_per).mean(axis=0)


def ascii_profile(prof: np.ndarray, width: int = 64, height: int = 12) -> str:
    """把折叠包络画成 ASCII（dB 纵轴）。"""
    n = len(prof)
    idx = (np.arange(width) * n // width)
    v = to_db(np.maximum(prof[idx], 1e-20))
    vmax, vmin = v.max(), max(v.min(), v.max() - 60)
    rows = []
    for r in range(height):
        hi = vmax - (vmax - vmin) * r / height
        lo = vmax - (vmax - vmin) * (r + 1) / height
        rows.append(f"{hi:6.1f}|" + "".join("#" if x >= lo else " " for x in v))
    rows.append(" " * 6 + "+" + "-" * width)
    return "\n".join(rows)


def duty_and_ratio(prof: np.ndarray):
    """用中位分割法估占空比与 ON/OFF 比（不依赖预设 50%）。"""
    lo, hi = np.percentile(prof, 10), np.percentile(prof, 90)
    thr = np.sqrt(max(lo, 1e-20) * max(hi, 1e-20))   # 几何中点，跨数量级更稳
    on = prof >= thr
    if on.all() or not on.any():
        return None, None, None
    on_mean, off_mean = prof[on].mean(), prof[~on].mean()
    return on.mean(), on_mean / max(off_mean, 1e-20), int(on.sum())


def longest_on_run(prof: np.ndarray):
    """不预设占空比：几何中点阈值 + **环形最长连续段** -> (起点, 长度, 阈值)。
    这正是新版 rt_sync 要用的估计器，放在这里是为了先验证再改代码。"""
    lo, hi = np.percentile(prof, 10), np.percentile(prof, 90)
    thr = np.sqrt(max(lo, 1e-20) * max(hi, 1e-20))
    on = prof >= thr
    n = len(on)
    if on.all() or not on.any():
        return None, 0, thr
    # 环形：把序列接一倍，找最长的连续 True（长度上限 n）
    d = np.concatenate([on, on])
    best_len, best_start, cur = 0, 0, 0
    for i in range(2 * n):
        if d[i]:
            cur += 1
            if cur > best_len:
                best_len, best_start = cur, i - cur + 1
        else:
            cur = 0
    return best_start % n, min(best_len, n), thr


def sync_contrast(prof: np.ndarray, n_on: int):
    """复刻 rt_sync 的滑窗+contrast，用来解释它给出的数。"""
    n = len(prof)
    w = max(1, min(n_on, n - 1))
    c = np.concatenate(([0.0], np.cumsum(np.concatenate([prof, prof[:w]]))))
    win = c[w:w + n] - c[:n]
    best = int(np.argmax(win))
    on_mean = win[best] / w
    off_mean = (prof.sum() - win[best]) / max(n - w, 1)
    d = on_mean + off_mean
    return best, ((on_mean - off_mean) / d if d > 0 else 0.0), on_mean / max(off_mean, 1e-20)


def caf_stream(data: np.ndarray, phase: int, n_int: int, n_per: int, n_fr: int,
               norm: str = "none"):
    """按给定相位/积分长度，把 IQ 折成每 TDD 帧一个复数的流（1kHz）。

    和 rt_dsp 里的算法一致（conj(ch0)*ch1 逐帧积分），只是离线版本不追求零分配。

    norm:
      none  —— 原样（rt_dsp 现在的做法）
      p1    —— 除以本帧 ch1 功率
      both  —— 除以 sqrt(P0*P1)，即归一化互相关系数，模长∈[0,1]
    """
    out = np.empty(n_fr, dtype=np.complex128)
    for k in range(n_fr):
        s = (phase + k * n_per) * 2
        seg = data[:, s:s + n_int * 2]
        if seg.shape[1] < n_int * 2:
            out[k] = 0
            continue
        a = seg[0].reshape(-1, 2).astype(np.float64)
        b = seg[1].reshape(-1, 2).astype(np.float64)
        re = (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum()
        im = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()
        v = (re + 1j * im) / n_int
        if norm == "p1":
            p1 = (b * b).sum() / n_int
            v /= max(p1, 1e-12)
        elif norm == "both":
            p0 = (a * a).sum() / n_int
            p1 = (b * b).sum() / n_int
            v /= max(np.sqrt(p0 * p1), 1e-12)
        out[k] = v
    return out


def line_spectrum(x: np.ndarray, fs: float, nseg: int = 512):
    """Welch 平均（Blackman 窗，50% 重叠）-> (freqs, dB)。去均值即去 DC。"""
    x = x - x.mean()
    win = np.blackman(nseg)
    hop = nseg // 2
    n = (len(x) - nseg) // hop + 1
    if n < 1:
        nseg, win, hop, n = len(x), np.blackman(len(x)), len(x), 1
    acc = np.zeros(nseg)
    for i in range(n):
        seg = x[i * hop:i * hop + nseg] * win
        acc += np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(nseg, 1.0 / fs))
    return f, 10 * np.log10(acc / n + 1e-30)


def report_lines(f, S, tag):
    """报告若干关注频点相对局部本底的高出量。"""
    k = 21
    pad = np.pad(S, (k // 2, k // 2), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    base = np.median(sliding_window_view(pad, k), axis=-1)
    ex = S - base
    cells = []
    for tgt in (50, 100, 150, 200, 250, 300):
        i = int(np.argmin(np.abs(f - tgt)))
        j = int(np.argmin(np.abs(f + tgt)))
        cells.append(f"{max(ex[i], ex[j]):+5.1f}")
    # 最强的一条非DC线
    m = np.abs(f) > 20
    top = int(np.argmax(np.where(m, ex, -np.inf)))
    print(f"  {tag:<22} " + " ".join(cells) +
          f"   | 最强线 {f[top]:+7.1f}Hz {ex[top]:+5.1f}dB")


def run_spur(cfg, a) -> int:
    """杂散来源诊断：同一份采集上对比不同积分长度/相位，并量相位漂移。"""
    from rt_source import UsrpSource
    from rt_sync import TddSync

    n_step = cfg.n_step
    n_cap = int(round(a.spur_sec / cfg.step_sec))
    print(f"抓 {n_cap} 步 = {n_cap*cfg.step_sec:.1f}s "
          f"({n_cap*n_step*2*2*cfg.num_channels/1e6:.0f} MB 常驻内存)")
    buf = np.empty((cfg.num_channels, n_cap * n_step * 2), dtype=np.int16)

    sync = TddSync(cfg)
    phases = []
    src = UsrpSource(cfg).open()
    try:
        for k, raw in enumerate(src.steps()):
            if k >= n_cap:
                break
            np.copyto(buf[:, k * n_step * 2:(k + 1) * n_step * 2], raw)
            # 每步都重估相位（正常运行是每 10 步一次），专门用来看它稳不稳
            sync.update(raw, force=True)
            phases.append((sync.phase, sync.contrast, sync.n_integrate))
        errors = str(src.errors)
        dropped = src.dropped
    finally:
        src.close()

    ph = np.array([p[0] for p in phases])
    ct = np.array([p[1] for p in phases])
    ni = np.array([p[2] for p in phases])
    print(f"UHD: {errors}  主动丢步={dropped}")
    if dropped:
        print("⚠️ 有丢步，帧流不连续，下面的谱不可信")

    print(f"\n--- 上升沿稳定性（每 20ms 重估一次，共 {len(ph)} 次）---")
    print(f"  相位: 中位 {int(np.median(ph))}  范围 [{ph.min()}, {ph.max()}]  "
          f"跨度 {ph.max()-ph.min()} 样本 = {(ph.max()-ph.min())/cfg.sample_rate*1e6:.1f}µs")
    d = np.diff(ph.astype(np.int64))
    print(f"  相邻两次跳变: 中位 {int(np.median(np.abs(d)))} 样本, "
          f"最大 {int(np.abs(d).max())} 样本 "
          f"({np.abs(d).max()/cfg.sample_rate*1e6:.1f}µs)")
    slope = np.polyfit(np.arange(len(ph)) * cfg.step_sec, ph, 1)[0]
    print(f"  线性漂移: {slope:+.1f} 样本/秒 = {slope/cfg.sample_rate*1e6:+.2f} µs/s "
          f"({slope/cfg.sample_rate*1e6:+.2f} ppm)")
    print(f"  积分长度: 中位 {int(np.median(ni))} 样本 "
          f"({np.median(ni)/cfg.sample_rate*1e6:.0f}µs)  "
          f"范围 [{ni.min()}, {ni.max()}]")
    print(f"  对比度: 中位 {np.median(ct):.3f}  最低 {ct.min():.3f}")

    n_per, n_fr = cfg.n_period, n_cap * cfg.n_frames_per_step - 1
    phase0 = int(np.median(ph))
    print(f"\n--- 不同积分长度的杂散（同一份数据，固定相位={phase0}）---")
    print(f"  {'配置':<22} {'50Hz':>5} {'100':>5} {'150':>5} {'200':>5} "
          f"{'250':>5} {'300':>5}   (高出本底 dB)")
    for us in (400, 500, 600, 695, 900, 1000):
        n_int = int(round(us * 1e-6 * cfg.sample_rate))
        if n_int > n_per:
            continue
        x = caf_stream(buf, phase0, n_int, n_per, n_fr)
        f, S = line_spectrum(x, cfg.fs_out)
        report_lines(f, S, f"积分 {us}µs")

    print(f"\n--- 同一积分长度、不同起点（测边沿是不是切歪了）---")
    n_int = int(round(500e-6 * cfg.sample_rate))
    for off_us in (-100, -50, 0, 50, 100, 200):
        off = int(round(off_us * 1e-6 * cfg.sample_rate))
        x = caf_stream(buf, (phase0 + off) % n_per, n_int, n_per, n_fr)
        f, S = line_spectrum(x, cfg.fs_out)
        report_lines(f, S, f"500µs, 起点{off_us:+d}µs")

    print(f"\n--- 逐帧幅度归一化（杂散是幅度调制，除掉就该消失）---")
    for us in (500, 695):
        n_int = int(round(us * 1e-6 * cfg.sample_rate))
        for nm, tag in (("none", "原样"), ("p1", "除P1"), ("both", "除√(P0P1)")):
            x = caf_stream(buf, phase0, n_int, n_per, n_fr, norm=nm)
            f, S = line_spectrum(x, cfg.fs_out)
            report_lines(f, S, f"{us}µs {tag}")

    # 调制深度用**无量纲**的调制指数，不用"高出本底多少 dB"——后者受各通道
    # 自身信噪比影响，两个通道之间不可比。
    def mod_depth(seq, fs, targets=(100.0, 200.0)):
        x = np.asarray(seq, float)
        mu = x.mean()
        y = x - mu
        w = np.blackman(len(y))
        F = np.fft.rfft(y * w)
        fr = np.fft.rfftfreq(len(y), 1.0 / fs)
        cg = w.sum() / 2.0          # 相干增益，把窗的幅度损失补回来
        out = []
        for tg in targets:
            i = int(np.argmin(np.abs(fr - tg)))
            amp = np.abs(F[i - 1:i + 2]).max() / cg
            out.append(100.0 * amp / abs(mu))
        return out

    def frame_power(ch, start, n_int):
        p = np.empty(n_fr)
        for k in range(n_fr):
            s = ((start + k * n_per) % (n_cap * n_step)) * 2
            seg = buf[ch, s:s + n_int * 2].astype(np.float64)
            p[k] = (seg * seg).sum() / max(n_int, 1) if seg.size else 0.0
        return p

    # 静默段：ON 段结束后再留 100µs 拖尾，取到下一个 ON 起点前 20µs
    guard_us = 100
    on_us = int(round(np.median(ni) / cfg.sample_rate * 1e6))
    off_start = (phase0 + int((on_us + guard_us) * 1e-6 * cfg.sample_rate)) % n_per
    off_len = n_per - int((on_us + guard_us + 20) * 1e-6 * cfg.sample_rate)
    print(f"\n--- ADC/仪器 vs 信号：ON 段 与 静默段 各自的调制深度 ---")
    if off_len < 0.05 * n_per:
        # TDD 没锁定（TX 没开）时整个周期都算 ON，静默段长度会算成负数。
        # 这时这个判据本身就不成立，别打一堆没意义的数字。
        print("  ⚠️ TDD 未锁定（TX 没开？），没有静默段可比，跳过")
        return 0
    print(f"  ON 段 = phase {phase0}, {on_us}µs   "
          f"静默段 = phase {off_start}, {int(off_len/cfg.sample_rate*1e6)}µs")
    print(f"  {'':22} {'平均功率':>9} {'100Hz深度':>10} {'200Hz深度':>10}")
    for ch in range(cfg.num_channels):
        for tag, st, ln in (("ON段", phase0, int(np.median(ni))),
                            ("静默段", off_start, max(off_len, 1))):
            p = frame_power(ch, st, ln)
            d100, d200 = mod_depth(p, cfg.fs_out)
            print(f"  ch{ch} {tag:<18} {to_db(p.mean()/32767**2):>8.1f}dB "
                  f"{d100:>9.2f}% {d200:>9.2f}%")
    print("  （静默段也有同样深度 -> 仪器/电源/时钟；只有 ON 段有 -> 在发射信号里）")

    # 两路的调制是不是同一个东西
    p0 = frame_power(0, phase0, int(np.median(ni)))
    p1 = frame_power(1, phase0, int(np.median(ni)))
    for tg in (100.0, 200.0):
        z = np.exp(-2j * np.pi * tg * np.arange(n_fr) / cfg.fs_out)
        a0 = np.vdot(z, p0 - p0.mean()) / n_fr
        a1 = np.vdot(z, p1 - p1.mean()) / n_fr
        ph = np.rad2deg(np.angle(a1 / a0)) if abs(a0) > 0 else float("nan")
        print(f"  {tg:.0f}Hz 两路相位差 {ph:+7.1f}°  "
              f"(0°=同一个源同相调制, 随机=互不相干)")

    print(f"\n--- 单通道功率（不做 CAF，直接看 TX 自己有没有超帧）---")
    for us in (500, 695):
        n_int = int(round(us * 1e-6 * cfg.sample_rate))
        for ch in range(cfg.num_channels):
            p = np.empty(n_fr)
            for k in range(n_fr):
                s = (phase0 + k * n_per) * 2
                seg = buf[ch, s:s + n_int * 2].astype(np.float64)
                p[k] = (seg * seg).sum() / n_int
            f, S = line_spectrum(p.astype(np.complex128), cfg.fs_out)
            report_lines(f, S, f"ch{ch} 功率 {us}µs")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="TDD 结构诊断（不落盘）")
    p.add_argument("--spur", action="store_true",
                   help="杂散来源诊断：对比积分长度/起点，并量上升沿漂移")
    p.add_argument("--spur-sec", type=float, default=1.0,
                   help="杂散诊断抓多少秒（1s -> 1Hz 分辨率, ~80MB 内存）")
    p.add_argument("--tracking-cal", action="store_true",
                   help="打开 AD9361 的 DC offset / IQ 平衡跟踪校准（默认关，对比用）")
    p.add_argument("--fs", type=float, default=10.0, help="采样率 MHz")
    p.add_argument("--gain", type=float, default=30.0)
    p.add_argument("--serial", type=str, default="32392D3")
    p.add_argument("--steps", type=int, default=5, help="抓几个 20ms 的 step")
    p.add_argument("--max-period-us", type=float, default=20000.0,
                   help="自相关搜索的最大周期（µs）")
    a = p.parse_args()

    cfg = RtConfig(serial=a.serial, sample_rate=a.fs * 1e6, gain=a.gain,
                   rx_tracking_cal=a.tracking_cal)
    if a.spur:
        return run_spur(cfg, a)

    from rt_source import UsrpSource

    src = UsrpSource(cfg).open()
    try:
        chunks = []
        for k, raw in enumerate(src.steps()):
            chunks.append(raw.copy())          # 视图会被复用，必须拷
            if len(chunks) >= a.steps:
                break
        errors = str(src.errors)
    finally:
        src.close()

    data = np.concatenate(chunks, axis=1)      # (2, n*2) int16
    dur_ms = data.shape[1] / 2 / cfg.sample_rate * 1e3
    print("=" * 70)
    print(f"抓到 {dur_ms:.0f} ms x {cfg.num_channels}ch @ {cfg.sample_rate/1e6:.0f}MHz "
          f"gain={cfg.gain:.0f}dB  fc={cfg.center_freq/1e9:.3f}GHz")
    print(f"UHD: {errors}")

    envs = []
    for ch in range(cfg.num_channels):
        d = data[ch]
        pk = int(np.abs(d).max())
        e = envelope(d, cfg.sample_rate)
        envs.append(e)
        print(f"\n--- ch{ch} ---")
        print(f"  平均功率 {to_db(e.mean()):.1f} dBFS   峰值样点 {pk} / 32767 "
              f"({to_db((pk/32767.0)**2):.1f} dBFS){'  ⚠️削波' if pk > 32000 else ''}")
        print(f"  包络分位 p10={to_db(np.percentile(e,10)):.1f}  "
              f"p50={to_db(np.percentile(e,50)):.1f}  "
              f"p90={to_db(np.percentile(e,90)):.1f}  "
              f"p99={to_db(np.percentile(e,99)):.1f} dBFS  "
              f"(p90-p10={to_db(np.percentile(e,90))-to_db(np.percentile(e,10)):+.1f}dB)")

        best, peak, tops = find_period(e, BINS_US, 50.0, a.max_period_us)
        if best is None:
            print("  自相关：数据太短，测不了")
            continue
        print(f"  自相关候选周期: " +
              "  ".join(f"{t:.0f}µs({v:.2f})" for t, v in tops))

        for tag, per in (("实测", best), ("假设", cfg.tdd_period_sec * 1e6)):
            prof = fold(e, per, BINS_US)
            if prof is None:
                continue
            duty, ratio, n_on = duty_and_ratio(prof)
            if duty is None:
                print(f"  [{tag} {per:.0f}µs] 折叠包络是平的，没有 ON/OFF 结构")
                continue
            _, ctr, ctr_ratio = sync_contrast(prof, int(round(len(prof) * 0.5)))
            print(f"  [{tag}周期 {per:.0f}µs] 占空={duty:.1%} ({n_on}µs)  "
                  f"ON/OFF={to_db(ratio):.1f}dB  "
                  f"rt_sync式contrast(按50%窗)={ctr:.2f}")
            if tag != "实测":
                continue
            print(ascii_profile(prof))
            s, L, thr = longest_on_run(prof)
            if s is None:
                print("  最长ON段：找不到（包络是平的）")
            else:
                _, ctr2, r2 = sync_contrast(prof, L)
                print(f"  最长ON段: [{s}µs, {(s+L) % len(prof)}µs) 长 {L}µs "
                      f"({L/len(prof):.1%})  阈值={to_db(thr):.1f}dBFS")
                print(f"  -> 用这个窗宽重算: contrast={ctr2:.2f}  "
                      f"ON/OFF={to_db(r2):.1f}dB")
            # 20µs 分辨率的数值表：看清主突发之外还有没有别的东西
            m = (len(prof) // 20) * 20
            coarse = to_db(prof[:m].reshape(-1, 20).mean(axis=1))
            print("  包络(20µs/格, dBFS):")
            for r0 in range(0, len(coarse), 10):
                seg = coarse[r0:r0 + 10]
                print(f"    {r0*20:5d}µs " + " ".join(f"{v:6.1f}" for v in seg))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
