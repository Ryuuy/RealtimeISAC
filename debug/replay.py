#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把录好的多普勒谱重放进**实时链路真正在用的** PresenceDetector，按真值标注打分。

这是改判决参数之后的回归测试：改完 `rt_config.py` 就跑一遍，看检出/虚警/延迟/
翻转有没有退化。因为 import 的就是 rt_detect 本身，不存在"离线脚本和实时代码
两套实现慢慢漂掉"的问题。

    /usr/bin/python3 debug/replay.py                         # 最新一次录制
    /usr/bin/python3 debug/replay.py --label labels.json     # 自带真值
    /usr/bin/python3 debug/replay.py --sweep                 # 扫参找更优组合

真值标注格式（JSON）：
    {"present": [[0.5,14.5],[22.5,26.5]], "absent": [[16.5,21.5]],
     "ambiguous": [[26.5,30.0]]}
不给 --label 时用下面 DEFAULT_LABELS（2026-08-15 那段 30s 的用户口述标注）。
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "realtime"))

from rt_config import RtConfig, valid_mask                         # noqa: E402
from rt_detect import PresenceDetector                             # noqa: E402

# 用户口述：<15s 一直在动 / 16-22s 没人 / 22-27s 又在动 / 27-30s 落座几乎不动
# 转换处各留 ~1s 缓冲不计分——那里本来就允许有延迟
DEFAULT_LABELS = {
    "present": [[0.5, 14.5], [22.5, 26.5]],
    "absent": [[16.5, 21.5]],
    "ambiguous": [[14.5, 16.5], [21.5, 22.5], [26.5, 30.0]],
}


def newest(d):
    f = sorted(glob.glob(os.path.join(d, "doppler_*.npz")))
    return f[-1] if f else None


def mask_of(t, regions):
    m = np.zeros(len(t), bool)
    for a, b in regions:
        m |= (t >= a) & (t <= b)
    return m


def cfg_from_meta(meta, **over):
    """用录制时的采集参数 + 当前 rt_config 的判决参数重建 cfg。
    判决参数**不从 meta 取**——重放的目的正是拿新参数去打老数据。"""
    base = RtConfig(sample_rate=meta["sample_rate"], nfft=meta["nfft"],
                    dc_guard_hz=meta["dc_guard_hz"], step_sec=meta["step_sec"],
                    window_sec=meta["window_sec"])
    for k, v in over.items():
        setattr(base, k, v)
    return base


def replay(cfg, power, freqs):
    det = PresenceDetector(cfg, valid_mask(freqs, cfg), freqs)
    n = len(power)
    out = np.zeros(n, bool)
    e = np.zeros(n)
    fl = np.zeros(n)
    for i in range(n):
        out[i] = det.update(power[i], freqs)
        e[i] = det.energy.e_db
        fl[i] = det.energy.floor_db
    return out, e, fl


def score(pres, t, lab, onset):
    mP, mA = mask_of(t, lab["present"]), mask_of(t, lab["absent"])
    mX = mask_of(t, lab.get("ambiguous", []))
    idx = np.flatnonzero(pres & (t >= onset))
    lat = t[idx[0]] - onset if len(idx) else np.nan
    flips = np.flatnonzero(pres[1:] != pres[:-1])
    return {
        "tpr": pres[mP].mean() if mP.any() else np.nan,
        "far": pres[mA].mean() if mA.any() else np.nan,
        "amb": pres[mX].mean() if mX.any() else np.nan,
        "lat": lat,
        "flips": len(flips),
        "flip_t": t[flips],
    }


def main():
    p = argparse.ArgumentParser(description="重放录制数据评估判决参数")
    p.add_argument("npz", nargs="?", default=None)
    p.add_argument("--label", type=str, default=None, help="真值标注 JSON")
    p.add_argument("--sweep", action="store_true", help="扫门限/去抖")
    a = p.parse_args()

    path = a.npz or newest(HERE)
    if not path:
        print("debug/ 下没有录制文件", file=sys.stderr)
        return 1
    lab = json.load(open(a.label)) if a.label else DEFAULT_LABELS

    d = np.load(path)
    meta = json.loads(str(d["meta_json"]))
    r = d["ready"].astype(bool)
    spec, t, freqs = d["spec"][r], d["t"][r], d["freqs"]
    power = (np.abs(spec) ** 2).astype(np.float64)

    mP, mA = mask_of(t, lab["present"]), mask_of(t, lab["absent"])
    print(f"=== {os.path.basename(path)} ===")
    print(f"{len(t)} 帧 / {t[-1]:.1f}s   真值: 有人 {mP.sum()} 帧 / 无人 {mA.sum()} 帧")
    if not mA.any():
        print("⚠️ 标注里没有「无人」段，虚警率无法评估", file=sys.stderr)

    # 客观起跳时刻：能量跨过"有人中位"和"无人中位"的中点。
    # 不用标注的区间起点当基准——那是人凭记忆说的，会把标注误差算进延迟里。
    band = (np.abs(freqs) >= meta["dc_guard_hz"]) & (np.abs(freqs) <= 450)
    E = 10 * np.log10(power[:, band].mean(axis=1) + 1e-30)
    onset = np.nan
    if mP.any() and mA.any():
        mid = (np.median(E[mA]) + np.median(E[mP])) / 2
        cand = np.flatnonzero((t > np.min([b for _, b in lab["absent"]])) & (E > mid))
        onset = t[cand[0]] if len(cand) else np.nan

    cfg = cfg_from_meta(meta)
    pres, e, fl = replay(cfg, power, freqs)
    s = score(pres, t, lab, onset)
    print(f"\n--- 当前 rt_config 参数 ---")
    print(f"  能量: 基线环{cfg.energy_ring_sec:.0f}s p{cfg.energy_pct:.0f} "
          f"棘轮{cfg.energy_rise_db_s}dB/s  门限 +{cfg.energy_margin_db:.0f}dB")
    print(f"  CFAR: {'开' if cfg.use_cfar else '关'} "
          f"(门限+{cfg.cfar_threshold_db:g}dB, 连续游程>={cfg.cfar_min_run}bin)")
    print(f"  去抖: {cfg.debounce_enter}/{cfg.debounce_exit} of {cfg.debounce_n} "
          f"(窗 {cfg.debounce_n*cfg.step_sec:.2f}s)")
    print(f"\n  检出率 {s['tpr']:.1%}   虚警率 {s['far']:.1%}   "
          f"模糊段判有人 {s['amb']:.1%}")
    print(f"  检出延迟 {s['lat']:.2f}s (相对能量客观起跳 t={onset:.2f}s)")
    print(f"  状态翻转 {s['flips']} 次: " + " ".join(f"{x:.1f}s" for x in s["flip_t"]))
    print(f"  能量: 无人段 {np.median(E[mA]):.1f}dB  有人段 {np.median(E[mP]):.1f}dB  "
          f"差 {np.median(E[mP])-np.median(E[mA]):+.1f}dB")

    if a.sweep:
        print(f"\n--- 扫参 (按 检出率-3*虚警率 排序) ---")
        rows = []
        for mg in (6, 8, 10, 12, 15):
            for (N, en, ex) in ((10, 6, 3), (15, 8, 4), (20, 12, 3),
                                (25, 12, 3), (30, 15, 4), (30, 18, 5), (40, 20, 5)):
                for uc in (True, False):
                    c = cfg_from_meta(meta, energy_margin_db=mg, debounce_n=N,
                                      debounce_enter=en, debounce_exit=ex,
                                      use_cfar=uc)
                    pr, _, _ = replay(c, power, freqs)
                    sc = score(pr, t, lab, onset)
                    rows.append((sc["tpr"] - 3 * sc["far"], mg, N, en, ex, uc, sc))
        rows.sort(key=lambda r: -r[0])
        print(f"{'门限':>6} {'N/en/ex':>10} {'CFAR':>5} | {'检出':>6} {'虚警':>6} "
              f"{'模糊':>6} {'延迟':>7} {'翻转':>4}")
        for _, mg, N, en, ex, uc, sc in rows[:15]:
            print(f"{mg:>5}dB {f'{N}/{en}/{ex}':>10} {'开' if uc else '关':>5} | "
                  f"{sc['tpr']:>6.1%} {sc['far']:>6.1%} {sc['amb']:>6.1%} "
                  f"{sc['lat']:>6.2f}s {sc['flips']:>4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
