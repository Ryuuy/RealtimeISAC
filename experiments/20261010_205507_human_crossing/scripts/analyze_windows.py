#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Event table for the 2026-10-10 20:55 human-crossing session (three operator-defined windows).

Windows (host local time, as told by the operator):
  A 20:56:00-21:03:00  lateral pass (20 s per pass), 'thin' (sideways) posture
  B 21:03:00-21:08:00  sideways walking
  C 21:08:00-21:12:40  diagonal 45 deg (operator said "until :12"; data ends 21:12:40)
Link up/down from the USRP reference channel ch0 (> -30 dBFS = up).

    python3 analyze_windows.py ../raw ..      # writes events.csv, windows.md
"""
import csv, datetime, json, os, sys
import numpy as np

RAW = sys.argv[1] if len(sys.argv) > 1 else "../raw"
OUT = sys.argv[2] if len(sys.argv) > 2 else ".."
STAMP = "20261010_205507"
WINDOWS = [("A", "lateral_thin", (20, 56, 0), (21, 3, 0)),
           ("B", "sideways_walk", (21, 3, 0), (21, 8, 0)),
           ("C", "diagonal45", (21, 8, 0), (21, 12, 40))]
UP = -30.0


def main():
    z = np.load(os.path.join(RAW, f"isac_{STAMP}_p001.npz"), allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    day = datetime.datetime.fromtimestamp(meta["start_ms"] / 1000).replace(microsecond=0)
    ms = lambda hms: day.replace(hour=hms[0], minute=hms[1], second=hms[2]).timestamp() * 1000
    clk = lambda m: datetime.datetime.fromtimestamp(m / 1000).strftime("%H:%M:%S")
    t, c0, c1 = z["dbfs_t_ms"], z["dbfs_ch0"], z["dbfs_ch1"]
    wt, ver, ridge = z["wf_t_ms"], z["wf_verdict"], z["wf_ridge_hz"]
    rows = list(csv.DictReader(open(os.path.join(RAW, f"ntg_{STAMP}.csv"), encoding="utf-8")))
    nt = np.array([float(r["unix_time"]) * 1000 for r in rows])
    f = lambda k: np.array([float(r[k]) if r[k] else np.nan for r in rows])
    rx, cnr = f("slave_rx_dbm"), f("rx_down_cnr_db")
    up = c0 > UP
    ev = [(t[i], bool(up[i])) for i in range(1, len(up)) if up[i] != up[i - 1]]

    out, summ = [], []
    for name, label, a, b in WINDOWS:
        ta, tb = ms(a), ms(b)
        downs = [(s, u) for s, u in ev if ta <= s < tb]
        # pair each DOWN with the next UP
        outs = []
        for i, (s, u) in enumerate(ev):
            if not u and ta <= s < tb:
                nx = [x for x, uu in ev[i + 1:] if uu]
                outs.append((s, nx[0] if nx else float("nan")))
        for s, e in outs:
            n = (nt > s - 6000) & (nt < s) & np.isfinite(rx)
            lastrx = float(rx[n][-1]) if n.any() else float("nan")
            lastrx_age = (s - nt[n][-1]) / 1000 if n.any() else float("nan")
            # lead = time from the START of the latest contiguous Doppler-verdict run (gaps < 0.6 s) to the DOWN,
            # only if that run ended within 3 s before the DOWN (otherwise: no precursor)
            red = np.where((wt < s) & (wt > s - 12000) & (ver > 0))[0]
            lead = float("nan")
            if len(red) and (s - wt[red[-1]]) / 1000 <= 3.0:
                j = len(red) - 1
                while j > 0 and wt[red[j]] - wt[red[j - 1]] < 600: j -= 1
                lead = (s - wt[red[j]]) / 1000
            w3 = (wt > s - 3000) & (wt < s)
            out.append(dict(window=name, label=label, down_clock=clk(s), up_clock=clk(e) if np.isfinite(e) else "",
                            outage_s=(e - s) / 1000 if np.isfinite(e) else float("nan"), last_rx_dbm=lastrx, last_rx_age_s=lastrx_age,
                            doppler_first_red_before_down_s=lead,
                            doppler_in_prev3s=int((ver[w3] > 0).any()), ridge_max_prev3s_hz=float(np.nanmax(ridge[w3])) if w3.any() else float("nan")))
        # Doppler verdict episodes
        wm = (wt >= ta) & (wt < tb)
        vv = (ver[wm] > 0).astype(int); tt = wt[wm]
        eps = 0; i = 0
        while i < len(vv):
            if vv[i]:
                j = i
                while j + 1 < len(vv) and (vv[j + 1] or (j + 2 < len(vv) and vv[j + 2] and tt[j + 2] - tt[j] < 600)): j += 1
                eps += 1; i = j + 1
            else: i += 1
        summ.append((name, label, clk(ta), clk(tb), len(outs), eps, float(np.nanmedian([o[1] - o[0] for o in outs]) / 1000) if outs else float("nan")))

    cols = list(out[0].keys())
    with open(os.path.join(OUT, "events.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for o in out: w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in o.items()})
    with open(os.path.join(OUT, "windows.md"), "w", encoding="utf-8") as fh:
        fh.write("| window | label | from | to | link outages (ch0) | Doppler verdict episodes | median outage (s) |\n|---|---|---|---|---:|---:|---:|\n")
        for s in summ: fh.write(f"| {s[0]} | {s[1]} | {s[2]} | {s[3]} | {s[4]} | {s[5]} | {s[6]:.1f} |\n")
        fh.write("\n| window | DOWN | UP | outage (s) | last valid Rx before DOWN (dBm) / age (s) | Doppler lead = start of latest verdict run → DOWN (s) | Doppler in prev 3 s |\n|---|---|---|---:|---|---:|---|\n")
        for o in out:
            fh.write(f"| {o['window']} | {o['down_clock']} | {o['up_clock']} | {o['outage_s']:.1f} | {o['last_rx_dbm']:.1f} / {o['last_rx_age_s']:.1f} | "
                     f"{'' if not np.isfinite(o['doppler_first_red_before_down_s']) else format(o['doppler_first_red_before_down_s'], '.1f')} | {'yes' if o['doppler_in_prev3s'] else 'no'} |\n")
    print(open(os.path.join(OUT, "windows.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
