#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RealtimeISAC 多普勒链路的**消融实验**：一次拿掉一个处理步骤，看 ±100Hz(及 ±50·k) 梳
会不会消失，从而定位梳的核心成因。

这里**故意重写**了 rt_dsp.DopplerEngine.process() 的核心链路（gather -> CAF 乘积 ->
归一化 -> 去DC -> 加窗 -> FFT），因为要逐项开关。baseline 配置和 rt_dsp 数值一致
（已核对：跟 replay_raw_iq.py 的梳强度对得上）。TDD 相位仍用真的 rt_sync.TddSync。

四个可开关的因素：
  norm  : coeff  = 每帧除以 sqrt(Σ|ch0|²)·sqrt(Σ|ch1|²)  (= rt_dsp 现状, 相关系数)
          n_int  = 只除以固定的积分点数           (时不变归一化)
  gate  : on     = 每帧只积分实测 ON 突发          (= rt_dsp 现状)
          off    = 每帧积分整个 1ms 周期           (自由运行)
  dc    : ema    = 扣慢速 EMA 均值                 (= rt_dsp 现状)
          none   = 完全不去 DC
          winmean= 扣当前 0.2s 窗的均值
  win   : blackman (= rt_dsp 现状)  /  rect

用法:
    cd RealtimeISAC/realtime
    python ../debug/ablate_doppler.py ../../experiment_30MHz_static_20260710_155243 [--sec 12]
"""

import argparse
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "realtime"))

from rt_config import RtConfig                       # noqa: E402
from rt_sync import TddSync                          # noqa: E402

import json as _json


def load_params(folder):
    p = os.path.join(folder, "acquisition_parameters.json")
    return _json.load(open(p)) if os.path.exists(p) else {}


def run_chain(bin_path, cfg, n_steps, norm="coeff", gate="on", dc="ema", win="blackman",
              phase_override=None):
    """重写的多普勒链路，逐项可开关。返回 (spec (n_steps, nfft) complex, freqs)。"""
    n_step, n_per, n_fr = cfg.n_step, cfg.n_period, cfg.n_frames_per_step
    n_ring, nfft = cfg.n_ring, cfg.nfft
    SCALE = 32767.0

    mm = np.memmap(bin_path, dtype=np.int16, mode="r")
    n_avail = int(mm.shape[0]) // 4 // n_step
    n_steps = min(n_steps, n_avail)

    sync = TddSync(cfg)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / cfg.fs_out))
    w = (np.blackman(n_ring) if win == "blackman" else np.ones(n_ring)).astype(np.float64)

    ring = np.zeros(n_ring, dtype=np.complex128)
    dc_val = 0.0 + 0j
    dc_n = 0
    primed = 0
    spec = np.zeros((n_steps, nfft), dtype=np.complex64)

    raw = np.empty((2, n_step * 2), dtype=np.int16)
    for si in range(n_steps):
        off = si * n_step * 4
        blk = mm[off: off + n_step * 4].reshape(n_step, 4)
        raw[0] = np.ascontiguousarray(blk[:, 0:2]).reshape(-1)
        raw[1] = np.ascontiguousarray(blk[:, 2:4]).reshape(-1)
        sync.update(raw)

        phase = int(phase_override) % n_per if phase_override is not None else int(sync.phase) % n_per
        if gate == "on":
            n_int = min(sync.n_integrate, n_per)
        else:
            n_int = n_per
        # 保证 phase + (n_fr-1)*n_per + n_int 落在 raw 内
        n_int = min(n_int, n_step - (phase + (n_fr - 1) * n_per))
        if n_int < 8:
            phase = 0
            n_int = min(n_per, n_step - (n_fr - 1) * n_per)

        # gather: (2, n_fr, n_int, 2)
        g = np.empty((2, n_fr, n_int, 2), dtype=np.int64)
        for ch in range(2):
            src = raw[ch]
            for k in range(n_fr):
                s = (phase + k * n_per) * 2
                g[ch, k] = src[s: s + n_int * 2].reshape(n_int, 2)
        ar, ai = g[0, ..., 0], g[0, ..., 1]
        br, bi = g[1, ..., 0], g[1, ..., 1]
        re = np.einsum("ij,ij->i", ar, br) + np.einsum("ij,ij->i", ai, bi)
        im = np.einsum("ij,ij->i", ar, bi) - np.einsum("ij,ij->i", ai, br)
        re = re.astype(np.float64)
        im = im.astype(np.float64)

        if norm == "coeff":
            ea = (np.einsum("ij,ij->i", ar, ar) + np.einsum("ij,ij->i", ai, ai)).astype(np.float64)
            eb = (np.einsum("ij,ij->i", br, br) + np.einsum("ij,ij->i", bi, bi)).astype(np.float64)
            d = np.sqrt(ea) * np.sqrt(eb)
        else:  # n_int : 时不变
            d = np.full(n_fr, SCALE * SCALE * n_int, dtype=np.float64)
        dec = np.divide(re, d, out=np.zeros_like(re), where=d > 0) + \
            1j * np.divide(im, d, out=np.zeros_like(im), where=d > 0)

        if dc == "ema":
            # rt_dsp 现状：每 step(50Hz) 算一次、整段 step 保持不变地减
            m = dec.mean()
            dec = dec - dc_val
            dc_n += 1
            a = max(cfg.dc_ema_alpha, 1.0 / dc_n)
            dc_val = (1.0 - a) * dc_val + a * m
        elif dc == "ema_perframe":
            # 同样的 EMA，但按 1kHz 逐帧更新/相减（不再是 50Hz 阶梯保持）
            out = np.empty_like(dec)
            for kf in range(n_fr):
                out[kf] = dec[kf] - dc_val
                dc_n += 1
                a = max(cfg.dc_ema_alpha, 1.0 / dc_n)
                dc_val = (1.0 - a) * dc_val + a * dec[kf]
            dec = out
        # winmean / none 在下面窗口层处理

        ring[:-n_fr] = ring[n_fr:]
        ring[-n_fr:] = dec
        primed = min(primed + n_fr, n_ring)

        buf = ring.copy()
        if dc == "winmean":
            buf = buf - buf.mean()
        fb = np.zeros(nfft, dtype=np.complex128)
        fb[:n_ring] = buf * w
        S = np.fft.fftshift(np.fft.fft(fb))
        spec[si] = S
        # priming 不够就记 0（后面按 ready 掩掉）
        if primed < n_ring:
            spec[si] = 0.0

    return spec, freqs, sync


def comb_readout(spec, freqs, targets=(50, 100, 150, 200, 250, 300, 350, 400, 450)):
    from numpy.lib.stride_tricks import sliding_window_view
    rows = np.any(spec != 0, axis=1)
    S = spec[rows]
    P = (np.abs(S) ** 2).mean(axis=0)
    Pdb = 10 * np.log10(P + 1e-20)
    k = 31
    base = np.median(sliding_window_view(np.pad(Pdb, k // 2, mode="edge"), k), axis=-1)
    ex = Pdb - base
    out = []
    for f0 in targets:
        best = -9.0
        for s in (1, -1):
            i = int(np.argmin(np.abs(freqs - s * f0)))
            j = slice(max(0, i - 1), i + 2)
            best = max(best, float(ex[j].max()))
        out.append(best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="experiment 文件夹")
    ap.add_argument("--sec", type=float, default=12.0, help="每个配置处理多少秒 (默认 12)")
    ap.add_argument("--fs", type=float, default=None)
    a = ap.parse_args()

    params = load_params(a.folder)
    sr = (a.fs * 1e6) if a.fs else float(params.get("sample_rate_hz", 30e6))
    cfg = RtConfig(sample_rate=sr, center_freq=float(params.get("center_freq_hz", 1.89e9)),
                   gain=float(params.get("gain_db", 30.0)))
    bin_path = os.path.join(a.folder, "2ch_iq_data.bin")
    n_steps = int(round(a.sec / cfg.step_sec))

    configs = [
        ("baseline (rt_dsp 现状)",           dict(norm="coeff", gate="on",  dc="ema",         win="blackman")),
        ("拿掉逐帧归一化 (÷n_int)",           dict(norm="n_int", gate="on",  dc="ema",         win="blackman")),
        ("拿掉 TDD 门控 (积分整个 1ms)",      dict(norm="coeff", gate="off", dc="ema",         win="blackman")),
        ("拿掉去DC (完全不去)",              dict(norm="coeff", gate="on",  dc="none",        win="blackman")),
        ("去DC: EMA 改成逐帧(1kHz)更新",     dict(norm="coeff", gate="on",  dc="ema_perframe", win="blackman")),
        ("去DC 换成扣窗均值",                dict(norm="coeff", gate="on",  dc="winmean",     win="blackman")),
        ("窗换成矩形窗",                     dict(norm="coeff", gate="on",  dc="ema",         win="rect")),
        ("归一化+门控 都拿掉 (最接近老CAF)",  dict(norm="n_int", gate="off", dc="ema",         win="blackman")),
    ]

    print(f"=== 多普勒链路消融 @ {os.path.basename(a.folder)} ===")
    print(f"fs={sr/1e6:.0f}MHz  每配置 {n_steps} step (~{a.sec:.0f}s)  bin={cfg.bin_hz:.1f}Hz\n")
    hdr = f"{'配置':<34}" + "".join(f"{f:>6}" for f in (50, 100, 150, 200, 250, 300, 350, 400, 450))
    print(hdr)
    print("-" * len(hdr))
    base_row = None
    for name, kw in configs:
        spec, freqs, sync = run_chain(bin_path, cfg, n_steps, **kw)
        row = comb_readout(spec, freqs)
        if base_row is None:
            base_row = row
        s = f"{name:<34}" + "".join(f"{v:>+6.1f}" for v in row)
        print(s, flush=True)
    print("\n(数字 = 该多普勒频点相对 31 点滑动中值本底高多少 dB；>+5 算有梳，~0 算没梳)")
    print(f"TDD: {sync.status()}")


if __name__ == "__main__":
    main()
