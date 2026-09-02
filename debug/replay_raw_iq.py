#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把一份**已录好的原始 2 通道 sc16 IQ**（experiment_*/2ch_iq_data.bin 那种格式）
喂进**实时链路真正在用的** DSP（rt_sync + rt_dsp + rt_detect），
逐 step 处理，存成和 `rt_main --debug` 完全一样的 doppler_*.npz，用 plot_doppler.py 出图。

## 为什么写这个

`debug/replay.py` 只能重放**已经算好的多普勒谱**（改判决参数用）。
这个脚本从**原始 IQ**开始跑整条 DSP —— 目的是把某次别的采集
（比如 `experiment_30MHz_static_20260710_155243`，2026-07-10 用另一台 B210
serial=321D889 采的）按实时链路**一模一样**的处理跑一遍，看多普勒谱里会不会
出现同样的 ±50 / ±100 / ±200Hz 横线：

  - 如果 155243 也有同样的线   -> 这几条线是**信号 / TX 自带**的（跨设备、跨日期复现），
                                  不是 0816 那台硬件或这套算法造成的。
  - 如果 155243 干净           -> 那几条线跟 0816 当时的硬件状态 / 配置有关。

import 的就是 rt_sync / rt_dsp / rt_detect 本身，不存在"离线脚本和实时代码
两套实现慢慢漂掉"的问题（跟 replay.py 一个思路）。

## 数据结构对齐

- 输入文件：interleaved sc16，`ch0[0](re,im int16), ch1[0](re,im int16), ch0[1], ...`
  （= USRPSaveData2Channel.py 的存盘格式 = analyze_caf.py 读的格式）。
- 实时链路每个 step 要的是 `raw_i16` : (num_channels, n_step*2) int16，
  每行是该通道的 `[re0, im0, re1, im1, ...]`。
- 下面 `iter_steps()` 就是把前者切成后者，每个 step = window/step 参数决定的 n_step 个复样点。

## 用法

    cd RealtimeISAC/realtime         # 模块是平铺 import，必须在这个目录下跑
    python  ../debug/replay_raw_iq.py  ../../experiment_30MHz_static_20260710_155243
    python  ../debug/replay_raw_iq.py  <folder_or_bin>  --max-sec 3      # 先跑 3 秒试
    python  ../debug/replay_raw_iq.py  <folder_or_bin>  --no-plot

采样率 / step / window 默认从 acquisition_parameters.json + rt_config 默认值取，
可用 --fs / --step / --window 覆盖。其余 TDD / nfft / 陷波 / 判决参数**全部用
rt_config 的默认值**——这正是"把同一套算法应用到另一份数据上"的意思。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

# Windows 控制台默认 GBK，rt_sync.status() 里有 µ 之类字符会 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "realtime"))

from rt_config import RtConfig                                    # noqa: E402
from rt_detect import PowerMonitor, PresenceDetector              # noqa: E402
from rt_dsp import DopplerEngine, to_db                           # noqa: E402
from rt_record import DopplerRecorder                             # noqa: E402
from rt_sync import TddSync                                       # noqa: E402


def resolve_input(path):
    """接受 experiment 文件夹 或 直接的 .bin。返回 (bin_path, params_dict, base_name)。"""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        binf = os.path.join(path, "2ch_iq_data.bin")
        pf = os.path.join(path, "acquisition_parameters.json")
        params = json.load(open(pf)) if os.path.exists(pf) else {}
        return binf, params, os.path.basename(os.path.normpath(path))
    # 直接给 .bin：同目录找 acquisition_parameters.json
    pf = os.path.join(os.path.dirname(path), "acquisition_parameters.json")
    params = json.load(open(pf)) if os.path.exists(pf) else {}
    base = os.path.splitext(os.path.basename(path))[0]
    return path, params, base


def serial_from_params(params):
    s = str(params.get("sdr_args", ""))
    m = re.search(r"serial=([A-Za-z0-9]+)", s)
    return m.group(1) if m else (s or "offline")


def iter_synthetic_steps(cfg, n_steps, tone_hz=0.0, seed=0):
    """对照组：合成 TDD 门控信号，**不注入任何周期性调幅**，喂进同一条链路。

    共享波形 = 几个**相位连续**(按绝对时间求值，跨 step 不断)的 CW 音调之和，两路都灌这
    一份，各自再叠独立白噪声；ON 段有信号，OFF 段只有低噪声。可选给 ch1 叠一个干净的
    多普勒单音 tone_hz 验证频率轴。整个信号里**没有任何 50Hz 相关的东西**——
    如果输出谱里还是没有 ±50Hz 梳状线，就说明那套梳状线不是这条 DSP 链自己造出来的。

    关键：共享波形按绝对样本索引求值，`conj(ch0)*ch1` 跨 step 平滑，不会因为"每 step
    重新抽一次随机信号"而在 step 率(50Hz)上引入台阶。
    """
    rng = np.random.default_rng(seed)
    n = cfg.n_step
    n_per, n_on = cfg.n_period, cfg.n_on
    fs = cfg.sample_rate
    buf = np.zeros((cfg.num_channels, n * 2), dtype=np.int16)
    on = np.zeros(n, dtype=bool)
    for k in range(cfg.n_frames_per_step):
        on[k * n_per: k * n_per + n_on] = True
    on2 = np.repeat(on, 2)
    n_on2 = int(on2.sum())
    n_off2 = int((~on2).sum())

    # 相位连续的共享 CW 音调（频率随手挑几个，跟 50 无任何公约关系）
    pilot_hz = np.array([-3.11e6, -0.77e6, 1.9e6, 4.3e6])
    pilot_amp = np.array([2600.0, 3400.0, 3000.0, 2200.0])
    base_idx = 0
    for _ in range(n_steps):
        idx = base_idx + np.arange(n)
        t = idx / fs
        shared = np.zeros(n, dtype=np.complex128)
        for f0, amp in zip(pilot_hz, pilot_amp):
            shared += amp * np.exp(1j * 2 * np.pi * f0 * t)
        c0 = shared + (rng.normal(0, 500, n) + 1j * rng.normal(0, 500, n))
        c1 = shared + (rng.normal(0, 500, n) + 1j * rng.normal(0, 500, n))
        if tone_hz:
            c1 = c1 * np.exp(1j * 2 * np.pi * tone_hz * t)
        for ch, c in ((0, c0), (1, c1)):
            re = np.clip(np.round(c.real), -32767, 32767).astype(np.int16)
            im = np.clip(np.round(c.imag), -32767, 32767).astype(np.int16)
            iq = np.empty(n * 2, dtype=np.int16)
            iq[0::2] = re
            iq[1::2] = im
            buf[ch, :] = iq
        # OFF 段清成低噪声（门控）
        buf[:, ~on2] = rng.integers(-150, 150, (cfg.num_channels, n_off2), dtype=np.int16)
        base_idx += n
        yield buf
    return


def iter_steps(bin_path, cfg, max_steps):
    """产出 (2, n_step*2) int16。每个 step 是 cfg 决定的 n_step 个复样点的新数据。

    内存：memmap 打开，每 step 只把 2*n_step*2 个 int16 拷成连续（几 MB），
    不整盘加载。
    """
    n_step = cfg.n_step
    mm = np.memmap(bin_path, dtype=np.int16, mode="r")
    total_i16 = int(mm.shape[0])
    total_cplx_per_ch = total_i16 // 4          # 每复样点: ch0(re,im) + ch1(re,im) = 4 个 int16
    n_avail = total_cplx_per_ch // n_step
    n = n_avail if max_steps is None else min(n_avail, max_steps)
    raw = np.empty((2, n_step * 2), dtype=np.int16)
    for si in range(n):
        off = si * n_step * 4
        blk = mm[off: off + n_step * 4].reshape(n_step, 4)
        raw[0] = np.ascontiguousarray(blk[:, 0:2]).reshape(-1)   # ch0 交织 re/im
        raw[1] = np.ascontiguousarray(blk[:, 2:4]).reshape(-1)   # ch1 交织 re/im
        yield raw
    return


def spur_readout(spec, freqs, ready, dc_guard_hz, targets=(50, 100, 150, 200, 250, 300, 350, 400, 450)):
    """对整段时间平均的功率谱，量每个 ±k·50Hz 处相对局部本底高多少 dB。
    直接回答"155243 有没有这几条线、多强"。"""
    S = spec[ready.astype(bool)] if ready.any() else spec
    P = (np.abs(S) ** 2).mean(axis=0)                     # 时间平均线性功率
    Pdb = 10 * np.log10(P + 1e-20)
    # 局部本底：31 点滑动中值
    k = 31
    pad = np.pad(Pdb, k // 2, mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    base = np.median(sliding_window_view(pad, k), axis=-1)
    excess = Pdb - base
    rows = []
    for f0 in targets:
        best = -np.inf
        loc = None
        for sgn in (+1, -1):
            i = int(np.argmin(np.abs(freqs - sgn * f0)))
            j = slice(max(0, i - 1), i + 2)
            v = float(excess[j].max())
            if v > best:
                best, loc = v, sgn * f0
        rows.append((f0, best, loc))
    return rows, freqs, excess


def main():
    p = argparse.ArgumentParser(description="把原始 IQ 喂进实时 DSP，出多普勒图")
    p.add_argument("input", nargs="?", default=None, help="experiment 文件夹 或 2ch_iq_data.bin")
    p.add_argument("--max-sec", type=float, default=None, help="只处理前 N 秒（默认全部）")
    p.add_argument("--fs", type=float, default=None, help="采样率 MHz（默认从 json 取）")
    p.add_argument("--step", type=float, default=None, help="step 秒（默认 rt_config = 0.02）")
    p.add_argument("--window", type=float, default=None, help="窗长 秒（默认 rt_config = 0.2）")
    p.add_argument("--out-dir", type=str, default=HERE, help="npz/png 输出目录（默认 debug/）")
    p.add_argument("--tag", type=str, default=None, help="输出文件名里的标签（默认用文件夹名）")
    p.add_argument("--no-plot", action="store_true", help="只存 npz，不调 plot_doppler.py")
    p.add_argument("--self", type=int, default=None, choices=(0, 1), dest="self_ch",
                   help="自相关模式: 把另一路换成该通道的副本，于是链路里的 conj(ch0)*ch1 "
                        "变成 conj(chN)*chN —— 直接看该通道信号自己有没有 ±50Hz 周期结构。"
                        "0 = reference(ch0) 自相关, 1 = sensing(ch1) 自相关。")
    p.add_argument("--synthetic", action="store_true",
                   help="对照组: 用 TDD 门控白噪声(无任何周期调幅)跑同一条链路，"
                        "看这套 DSP 自己会不会造出 ±50Hz 梳状线")
    p.add_argument("--syn-sec", type=float, default=20.0, help="--synthetic 时合成多少秒")
    p.add_argument("--syn-tone", type=float, default=0.0,
                   help="--synthetic 时给 ch1 叠一个干净单音(Hz)，验证频率轴")
    a = p.parse_args()

    synthetic = a.synthetic
    if synthetic:
        params = {}
        bin_path = "<synthetic>"
        base_name = f"synthetic_{a.syn_sec:.0f}s" + (f"_tone{a.syn_tone:.0f}" if a.syn_tone else "")
    else:
        if not a.input:
            print("需要给一个 experiment 文件夹/.bin，或用 --synthetic", file=sys.stderr)
            return 1
        bin_path, params, base_name = resolve_input(a.input)
        if not os.path.exists(bin_path):
            print(f"找不到数据文件: {bin_path}", file=sys.stderr)
            return 1
    tag = a.tag or base_name
    if a.self_ch is not None and not a.tag:
        tag = f"{base_name}_selfCAF_ch{a.self_ch}"

    sr = (a.fs * 1e6) if a.fs else float(params.get("sample_rate_hz", 10e6))
    kw = dict(serial=serial_from_params(params), sample_rate=sr,
              center_freq=float(params.get("center_freq_hz", 1.89e9)),
              gain=float(params.get("gain_db", 30.0)))
    if a.step:
        kw["step_sec"] = a.step
    if a.window:
        kw["window_sec"] = a.window
    cfg = RtConfig(**kw)

    print("=== replay_raw_iq (原始 IQ -> 实时 DSP) ===")
    print(f"输入: {bin_path}" + ("  [对照组: TDD门控白噪声, 无周期调幅]" if synthetic else ""))
    print(f"采集参数: {params.get('data_format')}  fs={sr/1e6:.1f}MHz  "
          f"fc={cfg.center_freq/1e9:.3f}GHz  gain={cfg.gain:.0f}dB  serial={cfg.serial}")
    print(cfg.describe())
    print(f"TDD 处理: 周期 {cfg.tdd_period_sec*1e3:.1f}ms, 未锁定则自由运行(积分整个周期)")
    print(f"陷波口(仅影响判决, 不影响谱本身): {cfg.notch_freqs_hz} ±{cfg.notch_guard_hz:.0f}Hz")
    print("-" * 68)

    max_steps = None if a.max_sec is None else max(1, int(round(a.max_sec / cfg.step_sec)))

    sync = TddSync(cfg)
    engine = DopplerEngine(cfg)
    pmon = PowerMonitor(cfg)
    det = PresenceDetector(cfg, engine.dc_mask, engine.freqs)

    # 先数一下总 step 数，给 recorder 预分配
    if synthetic:
        n_avail = max(1, int(round(a.syn_sec / cfg.step_sec)))
    else:
        mm = np.memmap(bin_path, dtype=np.int16, mode="r")
        n_avail = int(mm.shape[0]) // 4 // cfg.n_step
        del mm
    n_steps = n_avail if max_steps is None else min(n_avail, max_steps)
    rec = DopplerRecorder(cfg, a.out_dir, max_sec=n_steps * cfg.step_sec + 5.0)
    print(f"总 {n_avail} step (~{n_avail*cfg.step_sec:.1f}s)，本次处理 {n_steps} step", flush=True)

    stream = (iter_synthetic_steps(cfg, n_steps, tone_hz=a.syn_tone) if synthetic
              else iter_steps(bin_path, cfg, max_steps))
    if a.self_ch is not None:
        src, dst = a.self_ch, 1 - a.self_ch
        print(f"⚠️ 自相关模式: ch{dst} <- ch{src} 副本，链路计算的是 conj(ch{src})*ch{src}")
        print("⚠️ 注意: rt_dsp 会把 CAF 除以 sqrt(Σ|a|²)·sqrt(Σ|b|²)，自相关时分子分母相等，"
              "每帧 dec≡1+0j，DC 一扣就只剩量化噪声——**这个模式在当前归一化下是退化的**，"
              "看参考信号自己的周期结构请用 analyze_caf_autocorr.py（有 delay 轴、不做相关系数归一化）。")
    t0 = time.perf_counter()
    dt_acc = 0.0
    for si, raw in enumerate(stream):
        if a.self_ch is not None:
            raw[1 - a.self_ch] = raw[a.self_ch]
        ts = time.perf_counter()
        sync.update(raw)
        pmon.update(raw)
        engine.ingest(raw)
        power = engine.process(sync.phase, sync.n_integrate)
        fd, pk = engine.peak(power)
        present = det.update(power, engine.freqs) if engine.ready else False
        # 离线：时间轴用精确栅格 si*step，不用 wall clock
        rec.add(si * cfg.step_sec, engine.last_spectrum, present,
                det.n_hits, det.max_run, det.peak_hz, engine.ready,
                pmon.inst, det.energy.e_db, det.energy.floor_db or 0.0)
        dt_acc += time.perf_counter() - ts
        if si == 0 or (si + 1) % max(1, n_steps // 20) == 0:
            print(f"  step {si+1}/{n_steps}  t={si*cfg.step_sec:5.1f}s  "
                  f"{sync.status()}  peak={fd:+7.1f}Hz  {pmon.status()}", flush=True)

    wall = time.perf_counter() - t0
    print("-" * 68)
    print(f"处理完成: {n_steps} step, DSP 累计 {dt_acc:.1f}s (p_mean {dt_acc/max(n_steps,1)*1e3:.1f}ms/step), "
          f"总耗时 {wall:.1f}s")
    print(sync.status())
    print(pmon.status())

    npz_path = rec.save(engine.freqs, extra=dict(
        sync_locked=bool(sync.locked), sync_contrast=round(sync.contrast, 4),
        sync_duty=round(sync.duty, 4), sync_phase=int(sync.phase),
        sync_n_integrate=int(sync.n_integrate),
        power_dbfs=[round(d, 2) for d in pmon.dbfs if d is not None],
        dry_run=False, uhd_errors=None,
        replay_source=bin_path, replay_of=base_name,
    ))
    if not npz_path:
        print("没有可保存的帧", file=sys.stderr)
        return 1
    # 改个一眼能认出来源的名字（仍匹配 doppler_*.npz，plot_doppler 照样能自动挑）
    nice = os.path.join(a.out_dir, f"doppler_replay_{tag}.npz")
    if os.path.abspath(nice) != os.path.abspath(npz_path):
        os.replace(npz_path, nice)
        npz_path = nice
    print(f"💾 已存 {npz_path}")

    # ---- 杂散线读数 ----
    d = np.load(npz_path)
    rows, faxis, excess = spur_readout(d["spec"], d["freqs"], d["ready"], cfg.dc_guard_hz)
    print("\n--- ±k·50Hz 线谱强度（整段时间平均谱，相对 31 点滑动中值本底）---")
    print(f"  {'目标':>6} {'高出本底':>10}   位置")
    for f0, ex, loc in rows:
        flag = "  <== 明显" if ex >= 8 else ("  <  弱" if ex >= 3 else "")
        print(f"  ±{f0:>4.0f}Hz  {ex:>+8.1f}dB   {loc:+.0f}Hz{flag}")
    print(f"\n  sync_locked={sync.locked}  contrast={sync.contrast:.3f}  duty={sync.duty:.1%}  "
          f"n_integrate={sync.n_integrate}  ({sync.n_integrate/cfg.sample_rate*1e6:.0f}µs)")

    if not a.no_plot:
        plot = os.path.join(HERE, "plot_doppler.py")
        for extra in ([], ["--fmax", "250"]):
            out_png = os.path.join(a.out_dir,
                                   f"doppler_replay_{tag}{'_fmax250' if extra else ''}.png")
            cmd = [sys.executable, plot, npz_path, "--no-show", "--out", out_png] + extra
            print(f"\n$ {' '.join(cmd)}")
            subprocess.run(cmd, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
