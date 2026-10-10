#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-step summary of the 2026-10-10 metal-plate knife-edge experiment.

Protocol (as told by the operator): plate centre on the LOS at 18:09:00 (0 cm), then every 30 s the plate moves
2 cm outward; last move at 18:19:30 (the retreat after that is NOT analysed). So step k covers
[18:09:00 + 30k, +30) s and the plate offset is 2k cm, k = 0..20 (0..40 cm). Wall clock = host local time (JST).

    python3 analyze_steps.py ../raw ..            # writes steps.csv, transitions.csv, steps.md next to README
"""
import csv
import datetime
import json
import os
import sys

import numpy as np

RAW = sys.argv[1] if len(sys.argv) > 1 else "../raw"
OUT = sys.argv[2] if len(sys.argv) > 2 else ".."
STEP_S, STEP_CM, N_STEPS = 30.0, 2.0, 21
ZERO_CLOCK = (18, 9, 0)
LINK_UP_DBFS = -30.0        # ch0 (reference) is ~ -6 dBFS with link up, ~ -52 dBFS while the link is down
# geometry / wavelength for the knife-edge inversion (see README)
C = 299792458.0
F_HZ = 25e9
D1 = D2 = 3.0
LAM = C / F_HZ
K_NU = np.sqrt(2 * (D1 + D2) / (LAM * D1 * D2))      # nu = h * K_NU   (h in m)


def knife_edge_loss_db(nu):
    nu = np.asarray(nu, float)
    return np.where(nu > -0.78, 6.9 + 20 * np.log10(np.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1), 0.0)


def invert_loss(loss_db):
    grid = np.linspace(-0.7, 6, 6701)
    return float(np.interp(loss_db, knife_edge_loss_db(grid), grid))


def main():
    z = np.load(os.path.join(RAW, "isac_20261010_180657_p001.npz"), allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    zero = datetime.datetime.fromtimestamp(meta["start_ms"] / 1000).replace(
        hour=ZERO_CLOCK[0], minute=ZERO_CLOCK[1], second=ZERO_CLOCK[2], microsecond=0).timestamp() * 1000
    t = (z["dbfs_t_ms"] - zero) / 1000
    c0, c1, dl = z["dbfs_ch0"], z["dbfs_ch1"], z["dlpower_dbfs"]
    dt = (z["dlpower_t_ms"] - zero) / 1000
    wt = (z["wf_t_ms"] - zero) / 1000
    ver, ridge = z["wf_verdict"], z["wf_ridge_hz"]
    rows = list(csv.DictReader(open(os.path.join(RAW, "ntg_20261010_180657.csv"), encoding="utf-8")))
    nt = (np.array([float(r["unix_time"]) * 1000 for r in rows]) - zero) / 1000
    f = lambda k: np.array([float(r[k]) if r[k] else np.nan for r in rows])
    rx_dn, rx_up, cnr_dn, cnr_up = f("slave_rx_dbm"), f("master_rx_dbm"), f("rx_down_cnr_db"), f("rx_up_cnr_db")
    tx_dn, wr_dn = f("master_tx_dbm"), f("wireless_range_down")

    med = lambda a: float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")
    # unblocked reference: last 5 steps (30..40 cm) where the link is 100% up
    ref_n = (nt >= STEP_S * 15) & (nt < STEP_S * N_STEPS)
    rx_clear = med(rx_dn[ref_n])

    steps = []
    for k in range(N_STEPS):
        a, b = STEP_S * k, STEP_S * (k + 1)
        m, n, w = (t >= a) & (t < b), (nt >= a) & (nt < b), (wt >= a) & (wt < b)
        rxm = med(rx_dn[n])
        loss = rx_clear - rxm if np.isfinite(rxm) else float("nan")
        nu = invert_loss(loss) if np.isfinite(loss) and 0.5 < loss < 19 else float("nan")
        steps.append(dict(
            step=k, offset_cm=k * STEP_CM,
            clock=(datetime.datetime.fromtimestamp((zero + a * 1000) / 1000)).strftime("%H:%M:%S"),
            t_start_s=a,
            link_up_pct=100 * float((c0[m] > LINK_UP_DBFS).mean()),
            ch0_med_dbfs=med(c0[m]), ch1_med_dbfs=med(c1[m]), dlpower_med_dbfs=med(dl[(dt >= a) & (dt < b)]),
            ntg_valid_pct=100 * float(np.isfinite(rx_dn[n]).mean()),
            slave_rx_dbm=rxm, master_rx_dbm=med(rx_up[n]), master_tx_dbm=med(tx_dn[n]),
            cnr_down_db=med(cnr_dn[n]), cnr_up_db=med(cnr_up[n]), phy_down_mbps=med(wr_dn[n]),
            loss_vs_clear_db=loss, nu_est=nu, edge_minus_los_cm_est=(nu / K_NU * 100) if np.isfinite(nu) else float("nan"),
            doppler_verdict_pct=100 * float(ver[w].mean()), ridge_med_hz=med(ridge[w])))

    with open(os.path.join(OUT, "steps.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(steps[0]))
        wr.writeheader()
        for s in steps:
            wr.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in s.items()})

    # link up/down transitions from ch0 (50 Hz), restricted to the analysed window
    up = (c0 > LINK_UP_DBFS)
    tr = []
    for i in range(1, len(up)):
        if up[i] != up[i - 1] and 0 <= t[i] < STEP_S * N_STEPS:
            tr.append((t[i], "UP" if up[i] else "DOWN"))
    with open(os.path.join(OUT, "transitions.csv"), "w", newline="", encoding="utf-8") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["t_s_since_180900", "clock", "event", "plate_offset_cm"])
        for ts, ev in tr:
            clk = datetime.datetime.fromtimestamp((zero + ts * 1000) / 1000).strftime("%H:%M:%S")
            w_.writerow([f"{ts:.2f}", clk, ev, f"{STEP_CM * int(ts // STEP_S):g}"])

    with open(os.path.join(OUT, "steps.md"), "w", encoding="utf-8") as fh:
        fh.write("| offset | clock | link up % | Slave Rx dBm | loss vs clear dB | CNR down dB | PHY Mbps | ch0 dBFS | ch1 dBFS | Doppler verdict % |\n")
        fh.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for s in steps:
            fmt = lambda v, d=1: "–" if not np.isfinite(v) else f"{v:.{d}f}"
            fh.write(f"| {s['offset_cm']:g} cm | {s['clock']} | {s['link_up_pct']:.0f} | {fmt(s['slave_rx_dbm'])} | "
                     f"{fmt(s['loss_vs_clear_db'])} | {fmt(s['cnr_down_db'])} | {fmt(s['phy_down_mbps'], 0)} | "
                     f"{fmt(s['ch0_med_dbfs'])} | {fmt(s['ch1_med_dbfs'])} | {s['doppler_verdict_pct']:.0f} |\n")
    est = [s["edge_minus_los_cm_est"] for s in steps if np.isfinite(s["nu_est"])]
    print("clear Rx", rx_clear, "K_nu", K_NU, "lambda_mm", LAM * 1000)
    print("transitions", len(tr))
    for s in steps:
        if np.isfinite(s["nu_est"]):
            print("offset %4.0f cm: loss %.1f dB -> nu %.2f -> LOS is %.1f cm behind the plate edge -> edge at LOS when offset = %.1f cm" % (
                s["offset_cm"], s["loss_vs_clear_db"], s["nu_est"], s["edge_minus_los_cm_est"], s["offset_cm"] + s["edge_minus_los_cm_est"]))


if __name__ == "__main__":
    main()
