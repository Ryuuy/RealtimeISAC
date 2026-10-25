#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Very wide 3-panel figure for scrolling through a long session:
   1 Doppler waterfall   2 NTG Tx/Rx level (+CNR, right axis)   3 USRP energy (ch0 / ch1 / downlink slot)

    python3 debug/plot_long3.py isac_logs/isac_20261010_170034          # prefix
    python3 debug/plot_long3.py isac_logs/isac_20261010_170034 --px-per-s 16 --t0 0 --t1 600

x axis = seconds since session start (same as isac_overview.py), grid every --grid s (default 20 = one dwell),
top axis = wall-clock HH:MM:SS. Writes <prefix>_long3.png and <prefix>_long3.html (horizontal scroll).
"""
import argparse
import datetime
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from isac_overview import load_ntg, load_parts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix")
    ap.add_argument("--px-per-s", type=float, default=10.0)
    ap.add_argument("--grid", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=500.0)
    ap.add_argument("--t0", type=float, default=None, help="crop start (s since session start)")
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--from-clock", default=None, help="crop start as wall clock HH:MM[:SS] (host local time), to the end unless --t1")
    ap.add_argument("--zero-clock", default=None, help="make this wall clock HH:MM[:SS] the time zero of the x axis (grid then starts there) and crop to it")
    ap.add_argument("--step-s", type=float, default=None, help="dwell per position (s); sets the grid and labels each step")
    ap.add_argument("--step-cm", type=float, default=None, help="cm moved per step (label = k*step-cm from the zero clock)")
    ap.add_argument("--minor-grid", type=float, default=None, help="light grid every N s (e.g. 10)")
    ap.add_argument("--landmarks", default=None, help="comma list of seconds within each step to mark with dashed lines, e.g. 30,50 (needs --step-s)")
    ap.add_argument("--landmark-labels", default=None, help="comma list of labels for --landmarks")
    ap.add_argument("--trial-clock-labels", action="store_true", help="label each step with its wall-clock minute (HH:MM)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    S = load_parts(a.prefix + "_p*.npz")
    stamp = S["meta"].get("session_stamp", "session")
    d0 = os.path.dirname(S["files"][0]) or "."
    N = load_ntg(os.path.join(d0, f"ntg_{stamp}.csv"))
    dl_t, dl_v = [], []
    for f in S["files"]:
        z = np.load(f, allow_pickle=False)
        if "dlpower_t_ms" in z.files:
            dl_t.append(z["dlpower_t_ms"]); dl_v.append(z["dlpower_dbfs"])
    dl_t = np.concatenate(dl_t) if dl_t else np.array([]); dl_v = np.concatenate(dl_v) if dl_v else np.array([])

    starts = [S["wf_t_ms"][0], S["dbfs_t_ms"][0]]
    if N is not None:
        starts.append(np.nanmin(N["t_ms"]))
    t0ms = min(starts)
    if a.zero_clock:
        hms = [int(v) for v in a.zero_clock.split(":")] + [0, 0]
        t0ms = datetime.datetime.fromtimestamp(t0ms / 1000).replace(
            hour=hms[0], minute=hms[1], second=hms[2], microsecond=0).timestamp() * 1000
    sec = lambda ms: (np.asarray(ms, float) - t0ms) / 1000.0
    end = max(sec(S["wf_t_ms"][-1]), sec(S["dbfs_t_ms"][-1]), sec(np.nanmax(N["t_ms"])) if N else 0)
    x0 = a.t0 if a.t0 is not None else (0.0 if a.zero_clock else float(min(sec(S['wf_t_ms'][0]), 0.0)) * 0 + 0.0)
    if a.from_clock:
        hms = [int(v) for v in a.from_clock.split(":")] + [0, 0]
        day = datetime.datetime.fromtimestamp(t0ms / 1000).replace(hour=hms[0], minute=hms[1], second=hms[2], microsecond=0)
        x0 = day.timestamp() - t0ms / 1000
    x1 = a.t1 if a.t1 is not None else end

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    dpi = 100
    left_in, right_in = 0.9, 0.9
    width_in = (x1 - x0) * a.px_per_s / dpi + left_in + right_in
    fig, axs = plt.subplots(3, 1, figsize=(width_in, 14), sharex=True, dpi=dpi,
                            gridspec_kw={"height_ratios": [3.2, 2.2, 2.2], "hspace": 0.12})
    fig.subplots_adjust(left=left_in / width_in, right=1 - right_in / width_in, top=0.93, bottom=0.07)

    # 1 Doppler
    ax = axs[0]
    t = sec(S["wf_t_ms"])
    m = (t >= x0 - 1) & (t <= x1 + 1)
    db = S["dbmin"] + S["wf_db_q"][m].astype(np.float32) * (S["dbmax"] - S["dbmin"]) / 255.0
    fr = S["freqs"]
    ax.imshow(db.T, aspect="auto", origin="lower", extent=[t[m][0], t[m][-1], fr[0], fr[-1]],
              vmin=np.percentile(db, 5), vmax=np.percentile(db, 99.5), cmap="viridis", interpolation="nearest")
    ax.set_ylim(-a.fmax, a.fmax)
    v = S["wf_verdict"][m].astype(float)
    ax.fill_between(t[m], a.fmax * 0.93, a.fmax, where=v > 0.5, color="red", alpha=0.9, step="mid")
    ax.set_ylabel("Doppler (Hz)\nred strip = verdict")

    # 2 NTG levels
    ax = axs[1]
    if N is not None:
        tn = sec(N["t_ms"])
        ax.plot(tn, N["master_rx_dbm"], lw=1.0, label="Master Rx (uplink)", color="tab:blue")
        ax.plot(tn, N["slave_rx_dbm"], lw=1.0, label="Slave Rx (downlink)", color="tab:orange")
        ax.plot(tn, N["slave_tx_dbm"], lw=1.0, label="Slave Tx (uplink)", color="tab:green", ls="--")
        ax.plot(tn, N["master_tx_dbm"], lw=1.0, label="Master Tx (downlink)", color="tab:red", ls=":")
        ax.legend(loc="upper left", fontsize=9, ncol=4)
        a2 = ax.twinx()
        a2.plot(tn, N["rx_down_cnr_db"], lw=0.8, color="tab:purple", alpha=0.7)
        a2.set_ylabel("CNR down (dB)", color="tab:purple")
        a2.set_ylim(0, 40)
    ax.set_ylabel("NTG level (dBm)")

    # 3 USRP energy
    ax = axs[2]
    ts = sec(S["dbfs_t_ms"])
    ax.plot(ts, S["dbfs_ch0"], lw=0.9, color="tab:blue", label="ch0 reference")
    ax.plot(ts, S["dbfs_ch1"], lw=0.9, color="tab:orange", label="ch1 sensing")
    if len(dl_t):
        ax.plot(sec(dl_t), dl_v, lw=0.7, color="tab:red", alpha=0.7, label="downlink-slot power")
    ax.legend(loc="upper left", fontsize=9, ncol=3)
    ax.set_ylabel("USRP energy (dBFS)")
    ax.set_xlabel("seconds since session start")

    marks = list(zip(S["marker_t_ms"], [str(x) for x in S["marker_label"]]))
    if not marks and N is not None:
        marks = N["marks"]
    if a.step_s:
        a.grid = a.step_s
    for ax in axs:
        ax.set_xlim(x0, x1)
        ax.xaxis.set_major_locator(MultipleLocator(a.grid))
        ax.grid(alpha=0.35, which="major")
        for tm, lab in marks:
            ax.axvline(sec(tm), color="magenta", lw=1.0, alpha=0.8)
    for tm, lab in marks:
        axs[0].text(sec(tm), a.fmax * 0.85, lab, color="magenta", fontsize=8, rotation=90, va="top")
    axs[2].tick_params(axis="x", labelsize=8)
    if a.minor_grid:
        from matplotlib.ticker import MultipleLocator as _ML
        for ax in axs:
            ax.xaxis.set_minor_locator(_ML(a.minor_grid))
            ax.grid(alpha=0.18, which="minor")
    if a.step_s and a.landmarks:
        lm = [float(v) for v in a.landmarks.split(",")]
        lb = a.landmark_labels.split(",") if a.landmark_labels else [f"{v:g}s" for v in lm]
        k = int(np.floor(x0 / a.step_s))
        while k * a.step_s < x1:
            for v, lab in zip(lm, lb):
                xx = k * a.step_s + v
                if x0 <= xx <= x1:
                    for ax in axs:
                        ax.axvline(xx, color="red", ls=(0, (4, 3)), lw=0.9, alpha=0.65)
                    axs[0].text(xx, a.fmax * 0.60, lab, color="red", fontsize=8, rotation=90, va="top", ha="right",
                                bbox=dict(fc="white", alpha=0.6, lw=0, pad=0.5))
            k += 1
    if a.step_s and a.trial_clock_labels:
        k = int(np.floor(x0 / a.step_s))
        while k * a.step_s < x1:
            xm = (k + 0.5) * a.step_s
            if x0 <= xm <= x1:
                lab = datetime.datetime.fromtimestamp((t0ms + k * a.step_s * 1000) / 1000).strftime("%H:%M")
                for ax in axs:
                    ax.text(xm, 0.02, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10,
                            fontweight="bold", color="black", bbox=dict(fc="white", alpha=0.75, lw=0, pad=1))
            k += 1
    if a.step_s and a.step_cm is not None:
        k = int(np.floor(x0 / a.step_s))
        while k * a.step_s < x1:
            xm = (k + 0.5) * a.step_s
            if x0 <= xm <= x1:
                for ax in axs:
                    ax.text(xm, 0.02, f"{k * a.step_cm:g} cm", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                            fontsize=9, fontweight="bold", color="black", bbox=dict(fc="white", alpha=0.75, lw=0, pad=1))
            k += 1

    # top axis: wall clock
    top = axs[0].secondary_xaxis("top", functions=(lambda s_: s_, lambda s_: s_))
    top.xaxis.set_major_locator(MultipleLocator(60))
    top.set_xticklabels([])
    for s_ in np.arange(np.ceil(x0 / 60) * 60, x1, 60):
        wall = datetime.datetime.fromtimestamp((t0ms + s_ * 1000) / 1000).strftime("%H:%M:%S")
        axs[0].text(s_, 1.01, wall, transform=axs[0].get_xaxis_transform(), fontsize=8, ha="center", va="bottom")
    fig.suptitle(f"session {stamp}   t={x0:.0f}..{x1:.0f}s   grid {a.grid:.0f}s   top labels = wall clock (host local time)",
                 x=0.01, ha="left", fontsize=12)

    out = a.out or f"{a.prefix}_long3.png".replace("isac_logs/isac_", "isac_logs/long3_")
    fig.savefig(out, dpi=dpi)
    html = out[:-4] + ".html"
    with open(html, "w", encoding="utf-8") as f:
        f.write(f'<html><meta charset="utf-8"><body style="margin:0"><div style="overflow-x:auto;white-space:nowrap">'
                f'<img src="{os.path.basename(out)}"></div></body></html>')
    print(out, f"{width_in * dpi:.0f}x{14 * dpi} px", html)


if __name__ == "__main__":
    main()
