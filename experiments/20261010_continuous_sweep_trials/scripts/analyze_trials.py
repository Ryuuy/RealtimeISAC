#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-trial summary of the 2026-10-10 20:31+ continuous metal-plate sweep (one trial per minute).

Protocol (as told by the operator): trial k starts at 20:(31+k):00. At second 0 of every minute the plate EDGE is at
-30 cm (outside the LOS); at second 30 the edge is exactly on the LOS; at second 50 the plate CENTRE is on the LOS.
Assuming a constant 1 cm/s, edge position past the LOS [cm] = (second of the minute) - 30.
Wall clock = host local time (JST). Link up/down comes from the USRP reference channel (ch0): > -30 dBFS = up.

    python3 analyze_trials.py ../raw ..       # writes trials.csv, transitions.csv, trials.md, summary.txt
"""
import csv
import datetime
import json
import os
import sys

import numpy as np

RAW = sys.argv[1] if len(sys.argv) > 1 else "../raw"
OUT = sys.argv[2] if len(sys.argv) > 2 else ".."
STAMP = "20261010_203031"
ZERO = (20, 31, 0)
N_MIN = 14            # minutes 20:31 .. 20:44
UP_DBFS = -30.0


def main():
    z = np.load(os.path.join(RAW, f"isac_{STAMP}_p001.npz"), allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    zero = datetime.datetime.fromtimestamp(meta["start_ms"] / 1000).replace(
        hour=ZERO[0], minute=ZERO[1], second=ZERO[2], microsecond=0).timestamp() * 1000
    clock = lambda s: datetime.datetime.fromtimestamp((zero + s * 1000) / 1000).strftime("%H:%M:%S")
    t = (z["dbfs_t_ms"] - zero) / 1000
    c0 = z["dbfs_ch0"]
    up = c0 > UP_DBFS
    wt = (z["wf_t_ms"] - zero) / 1000
    ver, ridge = z["wf_verdict"], z["wf_ridge_hz"]
    rows = list(csv.DictReader(open(os.path.join(RAW, f"ntg_{STAMP}.csv"), encoding="utf-8")))
    nt = (np.array([float(r["unix_time"]) * 1000 for r in rows]) - zero) / 1000
    f = lambda k: np.array([float(r[k]) if r[k] else np.nan for r in rows])
    rx, cnr = f("slave_rx_dbm"), f("rx_down_cnr_db")

    ev = [(t[i], bool(up[i])) for i in range(1, len(up)) if up[i] != up[i - 1]]
    with open(os.path.join(OUT, "transitions.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s_since_203100", "clock", "event", "minute_clock", "second_of_minute"])
        for s, u in ev:
            m = int(s // 60)
            w.writerow([f"{s:.2f}", clock(s), "UP" if u else "DOWN", clock(60 * m)[:5], f"{s - 60 * m:.2f}"])

    out = []
    for k in range(N_MIN):
        a, b = 60.0 * k, 60.0 * k + 60
        evk = [(s, u) for s, u in ev if a <= s < b]
        downs = [s - a for s, u in evk if not u]
        ups = [s - a for s, u in evk if u]
        rec = dict(trial_clock=clock(a)[:5], n_down_events=len(downs), down_s=";".join(f"{d:.1f}" for d in downs),
                   up_s=";".join(f"{d:.1f}" for d in ups))
        if downs:
            d0 = a + downs[0]
            n = (nt > d0 - 1.5) & (nt < d0) & np.isfinite(rx)
            rec["first_down_s"] = downs[0]
            rec["edge_past_los_cm_assumed"] = downs[0] - 30
            rec["last_rx_before_down_dbm"] = float(rx[n][-1]) if n.any() else float("nan")
            rec["last_cnr_before_down_db"] = float(cnr[(nt > d0 - 1.5) & (nt < d0) & np.isfinite(cnr)][-1]) if ((nt > d0 - 1.5) & (nt < d0) & np.isfinite(cnr)).any() else float("nan")
            base = np.nanmedian(rx[(nt >= a) & (nt < a + 10)])
            w3 = np.where((nt >= a) & (nt < b) & (rx < base - 3))[0]
            rec["rx_baseline_dbm"] = float(base)
            rec["rx_drop_3db_start_s"] = float(nt[w3[0]] - a) if len(w3) else float("nan")
            ww = (wt > d0 - 10) & (wt < d0)
            rec["doppler_verdict_in_prev10s"] = int((ver[ww] > 0).any())
            rec["ridge_max_prev10s_hz"] = float(np.nanmax(ridge[ww])) if ww.any() else float("nan")
            nxt = [s for s, u in ev if s > d0 and u]
            rec["first_up_after_s"] = nxt[0] - a if nxt else float("nan")
            rec["first_outage_s"] = (nxt[0] - d0) if nxt else float("nan")
        out.append(rec)

    cols = ["trial_clock", "first_down_s", "edge_past_los_cm_assumed", "last_rx_before_down_dbm", "last_cnr_before_down_db",
            "rx_baseline_dbm", "rx_drop_3db_start_s", "doppler_verdict_in_prev10s", "ridge_max_prev10s_hz",
            "first_up_after_s", "first_outage_s", "n_down_events", "down_s", "up_s"]
    with open(os.path.join(OUT, "trials.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow({c: (f"{r[c]:.2f}" if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})

    # markdown table (trial minutes 20:32..20:43 = clean trials; 20:31 and 20:44 are listed separately)
    with open(os.path.join(OUT, "trials.md"), "w", encoding="utf-8") as fh:
        fh.write("| minute | 1st DOWN (s of minute) | edge past LOS (cm, 1 cm/s) | Rx −3 dB at (s) | last Rx before down (dBm) | "
                 "Doppler verdict in prev 10 s | UP (s) | 1st outage (s) | all DOWN (s) | all UP (s) |\n|---|---:|---:|---:|---:|---|---:|---:|---|---|\n")
        for r in out:
            if "first_down_s" not in r:
                fh.write(f"| {r['trial_clock']} | – | – | – | – | – | – | – | {r['down_s'] or '–'} | {r['up_s'] or '–'} |\n")
                continue
            fh.write(f"| {r['trial_clock']} | {r['first_down_s']:.1f} | {r['edge_past_los_cm_assumed']:+.1f} | {r['rx_drop_3db_start_s']:.1f} | "
                     f"{r['last_rx_before_down_dbm']:.1f} | {'yes' if r['doppler_verdict_in_prev10s'] else 'no'} | "
                     f"{r['first_up_after_s']:.1f} | {r['first_outage_s']:.1f} | {r['down_s']} | {r['up_s']} |\n")

    clean = [r for r in out if r["trial_clock"] in {f"20:{m}" for m in range(32, 44)} and "first_down_s" in r]
    d = np.array([r["first_down_s"] for r in clean])
    rd = np.array([r["rx_drop_3db_start_s"] for r in clean])
    lr = np.array([r["last_rx_before_down_dbm"] for r in clean])
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"clean trials 20:32..20:43: n={len(d)}\n")
        fh.write(f"first DOWN second-of-minute: mean {d.mean():.2f} std {d.std(ddof=1):.2f} min {d.min():.1f} max {d.max():.1f}\n")
        fh.write(f"=> edge past LOS at disconnect (assuming 1 cm/s): mean {d.mean() - 30:.1f} cm (range {d.min() - 30:.1f} .. {d.max() - 30:.1f})\n")
        fh.write(f"Rx -3 dB start: mean {np.nanmean(rd):.1f} s (range {np.nanmin(rd):.1f} .. {np.nanmax(rd):.1f}); lead before first DOWN: {np.nanmean(d - rd):.1f} s\n")
        fh.write(f"last Rx before down: mean {np.nanmean(lr):.1f} dBm (range {np.nanmin(lr):.1f} .. {np.nanmax(lr):.1f})\n")
        fh.write(f"Doppler verdict fired in the 10 s before the first DOWN: {sum(r['doppler_verdict_in_prev10s'] for r in clean)}/{len(clean)}\n")
    print(open(os.path.join(OUT, "summary.txt"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
