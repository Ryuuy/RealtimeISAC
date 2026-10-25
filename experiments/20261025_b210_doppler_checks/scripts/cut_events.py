#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按排定时刻把 dec / ch0 / ch1 幅度切成一个个事件窗口，并给每个事件配上检测、NTG 断连、ping 的汇总。

    /usr/bin/python3 cut_events.py --archive ../../../data_archive/20261025 --runs P1 F G H
    （在家用：archive 指向仓库里的 data_archive/20261025）

窗口规则（用户 10-25 指定）：10 s 一次的实验用排定时刻 S 的 [S-5, S+5)；其它间隔 Δ 一般化为 [S-5, S+Δ-5)，相邻窗口刚好首尾相接。
排定时刻 S_k = 起点 + k·间隔（walk_schedule.csv 里 start / spacing_s / n_slots）；P1 的 ±1 s 是近似，F/G/H 是整点。

输出到 <archive>/events/：
  <run>_events.npz   slot_idx, slot_time, win_t0(窗口第 1 个样本的真实时刻), fs=1000, dec(n×L complex64), amp0, amp1(n×L float32), ok(n×L bool)
  <run>_slots.csv    每个事件一行：n_enter、首次 enter 相对 S 的偏移、该窗口内是否有 NTG 断连/断开偏移/时长、ping 丢包比例、
                     ch0/ch1 中位 dBFS、|dec| 中位、管理口卡死标记（NTG XML 连续无数据 >60 s 视为管理口卡死，不是链路断）
时间都是 Ubuntu 本机 unix time（和 isac_logs / h_logs 同一个钟）。
"""
import argparse
import csv
import datetime as dt
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_h(prefix):
    import json
    meta = json.load(open(prefix + ".json", encoding="utf-8"))
    n_fr = meta["n_frames_per_step"]; fs = meta["fs_out"]
    t_step = np.fromfile(prefix + ".t.bin", dtype=np.float64)
    flag = np.fromfile(prefix + ".flag.bin", dtype=np.uint8)
    h = np.fromfile(prefix + ".h.bin", dtype=np.complex64)
    amp = np.fromfile(prefix + ".amp.bin", dtype=np.float32).reshape(-1, 2)
    ns = min(len(t_step), len(flag), len(h) // n_fr, len(amp) // n_fr)
    t_step, flag = t_step[:ns], flag[:ns]; h, amp = h[:ns * n_fr], amp[:ns * n_fr]
    t = (t_step[:, None] - (n_fr - 1 - np.arange(n_fr))[None, :] / fs).reshape(-1)
    ok = np.repeat((flag & 1 == 1) & (flag & 2 == 0), n_fr)
    return dict(h=h, amp=amp, t=t, ok=ok, fs=fs)


def hms(day, s):
    h, m, sec = [int(x) for x in s.split(":")]
    return day + h * 3600 + m * 60 + sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.path.join(HERE, "..", "..", "..", "data_archive", "20261025"))
    ap.add_argument("--runs", nargs="+", default=["P1", "F", "G", "H"])
    ap.add_argument("--date", default="2026-10-25")
    a = ap.parse_args()
    arc = os.path.abspath(a.archive); out = os.path.join(arc, "events"); os.makedirs(out, exist_ok=True)
    day = dt.datetime.strptime(a.date, "%Y-%m-%d").timestamp()
    sched = {r["run_id"]: r for r in csv.DictReader(open(os.path.join(arc, "walk_schedule.csv"), encoding="utf-8"))}

    # ---- detections (all session npz of the day) ----
    det_t, det_s = [], []
    for f in sorted(glob.glob(os.path.join(arc, "sessions_raw", "isac_2026????_*_p*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "det_t_ms" in z.files and len(z["det_t_ms"]):
            det_t += list(z["det_t_ms"] / 1000.0); det_s += [str(s) for s in z["det_state"]]
    det_t = np.array(det_t); det_s = np.array(det_s); o = np.argsort(det_t); det_t, det_s = det_t[o], det_s[o]
    enter = det_t[(det_s == "1") | (det_s == "doppler_detected")]; exit_ = det_t[(det_s == "0") | (det_s == "normal")]

    # ---- NTG XML csv (all loggers of the day) -> outage runs ----
    U, V = [], []
    for f in sorted(glob.glob(os.path.join(arc, "sessions_raw", "ntg_2026????_*.csv"))):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            try:
                U.append(float(r["unix_time"])); V.append(r["master_valid"] == "1")
            except (KeyError, ValueError):
                pass
    U = np.array(U); V = np.array(V, bool); o = np.argsort(U); U, V = U[o], V[o]
    inv = np.flatnonzero(~V); rr = np.split(inv, np.flatnonzero(np.diff(inv) > 3) + 1) if len(inv) else []
    outages = [(U[r[0]], U[r[-1]] + 0.1) for r in rr]

    # ---- ping (dataplane csv) ----
    PT, PO = [], []
    for f in sorted(glob.glob(os.path.join(arc, "sessions_raw", "dataplane_2026????_*.csv"))):
        for r in csv.DictReader(open(f)):
            PT.append(float(r["unix_time"])); PO.append(r["ok"] == "1")
    PT = np.array(PT); PO = np.array(PO, bool); o = np.argsort(PT); PT, PO = PT[o], PO[o]

    for run in a.runs:
        r = sched[run]; dlog = r["dec_log"].split("（")[0]
        if not dlog.startswith("h_"):
            print(run, "no dec log, skipped"); continue
        h = load_h(os.path.join(arc, "h_logs", dlog)); fs = h["fs"]
        n = int(r["n_slots"]); sp = float(r["spacing_s"]); S0 = hms(day, r["start"]); L = int(round(sp * fs))
        dec = np.zeros((n, L), np.complex64); a0 = np.zeros((n, L), np.float32); a1 = np.zeros((n, L), np.float32); okm = np.zeros((n, L), bool); wt0 = np.full(n, np.nan)
        rows = [["slot_idx", "slot_time", "win_start", "win_end", "n_enter", "first_enter_off_s", "enter_to_exit_s", "ntg_outage", "outage_start_off_s", "outage_len_s",
                 "mgmt_hung", "ping_lost_frac", "ch0_med_dbfs", "ch1_med_dbfs", "dec_abs_med", "dec_usable_frac"]]
        for k in range(n):
            S = S0 + k * sp; w0, w1 = S - 5.0, S - 5.0 + sp
            i0 = int(np.searchsorted(h["t"], w0)); sl = slice(i0, i0 + L)
            if i0 + L <= len(h["t"]) and abs(h["t"][i0] - w0) < 0.05:
                dec[k] = h["h"][sl]; a0[k] = h["amp"][sl, 0]; a1[k] = h["amp"][sl, 1]; okm[k] = h["ok"][sl]; wt0[k] = h["t"][i0]
            en = enter[(enter >= w0) & (enter < w1)]
            exs = exit_[exit_ > en[0]] if len(en) else []
            oo = [(a_, b_) for a_, b_ in outages if w0 <= a_ < w1]
            hung = bool(oo and (oo[0][1] - oo[0][0]) > 60)
            pm = (PT >= w0) & (PT < w1)
            rows.append([k, f"{S:.3f}", f"{w0:.3f}", f"{w1:.3f}", len(en), f"{en[0]-S:.2f}" if len(en) else "", f"{exs[0]-en[0]:.2f}" if len(en) and len(exs) else "",
                         int(bool(oo)), f"{oo[0][0]-S:.2f}" if oo else "", f"{oo[0][1]-oo[0][0]:.1f}" if oo else "", int(hung),
                         f"{1-PO[pm].mean():.3f}" if pm.sum() > 20 else "",
                         f"{20*np.log10(np.median(a0[k])+1e-9):.1f}", f"{20*np.log10(np.median(a1[k])+1e-9):.1f}", f"{np.median(np.abs(dec[k])):.3f}", f"{okm[k].mean():.3f}"])
        np.savez_compressed(os.path.join(out, f"{run}_events.npz"), slot_idx=np.arange(n), slot_time=S0 + np.arange(n) * sp, win_t0=wt0, fs=fs, dec=dec, amp0=a0, amp1=a1, ok=okm)
        with open(os.path.join(out, f"{run}_slots.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print(f"{run}: {n} slots x {L} samples -> events/{run}_events.npz, {run}_slots.csv ; slots with outage {sum(int(x[7]) for x in rows[1:])}")


if __name__ == "__main__":
    sys.exit(main())
