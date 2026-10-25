#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One figure per session: Doppler waterfall + reference/sensing power + DC ridge width + NTG-2501 link.

    python3 isac_overview.py "isac_logs/isac_20261009_150000_p*.npz" --ntg isac_logs/ntg_20261009_150000.csv
    python3 isac_overview.py isac_logs/isac_20261009_150000          # prefix: finds _p*.npz and ntg_*.csv itself

isac_bridge.py runs this on exit. Stand-alone (numpy + matplotlib only), so it also works on the
other computer. Everything is on ONE time axis: seconds since the first sample, converted from
server time (ms since epoch on the Ubuntu host); the NTG CSV's `unix_time` column is the same clock.

Rows:  1 Doppler waterfall (+ verdict strip)   2 ch0 power (reference)   3 ch1 power (sensing)
       4 DC ridge width + threshold            5 NTG Rx up/down          6 NTG Tx up/down
       7 NTG path loss (Tx-Rx) + CNR           8 NTG PHY rate + input/output rate
Vertical lines = operator markers (Enter in the bridge terminal).
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np


def load_parts(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files match {pattern}")
    keys = ["wf_t_ms", "wf_db_q", "wf_ridge_hz", "wf_verdict", "dbfs_t_ms", "dbfs_ch0", "dbfs_ch1",
            "marker_t_ms", "marker_label", "iface_server_recv_ms", "iface_enabled"]
    out = {k: [] for k in keys}
    meta = {}
    for f in files:
        d = np.load(f, allow_pickle=False)
        if not meta:
            meta = json.loads(str(d["meta_json"]))
            out["freqs"] = d["wf_freqs_hz"]
            out["dbmin"], out["dbmax"] = float(d["wf_db_min"]), float(d["wf_db_max"])
        for k in keys:
            if k in d.files:
                out[k].append(d[k])
    for k in keys:
        out[k] = np.concatenate(out[k]) if out[k] else np.array([])
    out["meta"] = meta
    out["files"] = files
    return out


def load_ntg(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    def col(name):
        return np.array([float(r[name]) if r.get(name) not in (None, "") else np.nan for r in rows])

    out = {k: col(k) for k in (
        "unix_time", "slave_tx_dbm", "master_rx_dbm", "master_tx_dbm", "slave_rx_dbm",
        "rx_up_cnr_db", "rx_down_cnr_db", "wireless_range_up", "wireless_range_down",
        "input_rate_up", "input_rate_down", "output_rate_up", "output_rate_down",
        "pathloss_up_db", "pathloss_down_db") if k in rows[0]}
    out["t_ms"] = out["unix_time"] * 1000.0
    marks = []
    for r in rows:
        for m in (r.get("marker") or "").split(";"):
            if "@" in m:
                lab, ts = m.rsplit("@", 1)
                marks.append((float(ts) * 1000.0, lab))
    out["marks"] = marks
    out["n_fail"] = int(np.isnan(out["master_rx_dbm"]).sum())
    return out


def block_mean(t, y, n_max=6000):
    """Stride-average so a multi-hour session still draws fast."""
    if len(t) <= n_max:
        return t, y
    k = int(np.ceil(len(t) / n_max))
    m = (len(t) // k) * k
    return (t[:m].reshape(-1, k).mean(1),
            np.nanmean(y[:m].reshape(-1, k, *y.shape[1:]), axis=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("isac", help='isac npz glob, or the prefix "isac_logs/isac_<stamp>"')
    ap.add_argument("--ntg", default=None, help="NTG CSV (default: ntg_<stamp>.csv next to the npz)")
    ap.add_argument("--out", default=None, help="PNG path (default overview_<stamp>.png next to the npz)")
    ap.add_argument("--fmax", type=float, default=500.0)
    ap.add_argument("--thresh-hz", type=float, default=None, help="ridge threshold line (default from file)")
    a = ap.parse_args()

    pat = a.isac if any(c in a.isac for c in "*?[") else a.isac + "_p*.npz"
    S = load_parts(pat)
    stamp = S["meta"].get("session_stamp", "session")
    d0 = os.path.dirname(S["files"][0]) or "."
    ntg_path = a.ntg or os.path.join(d0, f"ntg_{stamp}.csv")
    N = load_ntg(ntg_path)
    out = a.out or os.path.join(d0, f"overview_{stamp}.png")

    # ---- common time axis ----
    starts = [S["wf_t_ms"][0]] if len(S["wf_t_ms"]) else []
    if len(S["dbfs_t_ms"]):
        starts.append(S["dbfs_t_ms"][0])
    if N is not None:
        starts.append(np.nanmin(N["t_ms"]))
    t0 = min(starts)
    sec = lambda ms: (np.asarray(ms, float) - t0) / 1000.0

    marks = list(zip(S["marker_t_ms"], [str(x) for x in S["marker_label"]]))
    if not marks and N is not None:
        marks = N["marks"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_ntg = N is not None
    nrows = 8 if has_ntg else 4
    fig, axs = plt.subplots(nrows, 1, figsize=(18, 2.3 * nrows + 1), sharex=True,
                            gridspec_kw={"height_ratios": [2.2] + [1] * (nrows - 1)})
    ax = axs[0]

    # 1 waterfall
    if len(S["wf_t_ms"]):
        t = sec(S["wf_t_ms"])
        db = S["dbmin"] + S["wf_db_q"].astype(np.float32) * (S["dbmax"] - S["dbmin"]) / 255.0
        tt, dd = block_mean(t, db)
        fr = S["freqs"]
        ax.imshow(dd.T, aspect="auto", origin="lower", extent=[tt[0], tt[-1], fr[0], fr[-1]],
                  vmin=np.percentile(dd, 5), vmax=np.percentile(dd, 99.5), cmap="viridis")
        ax.set_ylim(-a.fmax, a.fmax)
        v = S["wf_verdict"].astype(float)
        tv, vv = block_mean(t, v)
        ax.fill_between(tv, a.fmax * 0.92, a.fmax, where=vv > 0.5, color="red", alpha=0.9, step="mid")
    ax.set_ylabel("Doppler (Hz)")
    ax.set_title(f"session {stamp}   (red strip on top = verdict 'Doppler detected')   "
                 f"parts={len(S['files'])}")

    # 2/3 power
    for i, (key, lab, col) in enumerate((("dbfs_ch0", "ch0 (reference) dBFS", "tab:blue"),
                                         ("dbfs_ch1", "ch1 (sensing) dBFS", "tab:orange"))):
        ax = axs[1 + i]
        if len(S["dbfs_t_ms"]):
            tt, yy = block_mean(sec(S["dbfs_t_ms"]), S[key].astype(float))
            ax.plot(tt, yy, lw=0.7, color=col)
        ax.set_ylabel(lab)

    # 4 ridge
    ax = axs[3]
    if len(S["wf_t_ms"]):
        tt, yy = block_mean(sec(S["wf_t_ms"]), S["wf_ridge_hz"].astype(float))
        ax.plot(tt, yy, lw=0.7, color="k")
        th = a.thresh_hz or float((S["meta"].get("rt_meta") or {}).get("ridge_thresh_hz", 90.0))
        ax.axhline(th, color="r", ls="--", lw=0.8)
    ax.set_ylabel("DC ridge width (Hz)")

    if has_ntg:
        t = sec(N["t_ms"])
        ok = ~np.isnan(t)
        t = t[ok]
        g = lambda k: N[k][ok] if k in N else np.full(len(t), np.nan)
        axs[4].plot(t, g("master_rx_dbm"), lw=0.8, label="Master Rx (uplink)")
        axs[4].plot(t, g("slave_rx_dbm"), lw=0.8, label="Slave Rx (downlink)")
        axs[4].set_ylabel("NTG Rx (dBm)"); axs[4].legend(loc="upper right", fontsize=7)
        axs[5].plot(t, g("slave_tx_dbm"), lw=0.8, label="Slave Tx (uplink)")
        axs[5].plot(t, g("master_tx_dbm"), lw=0.8, label="Master Tx (downlink)")
        axs[5].set_ylabel("NTG Tx (dBm)"); axs[5].legend(loc="upper right", fontsize=7)
        axs[6].plot(t, g("pathloss_up_db"), lw=0.8, color="tab:red", label="path loss up (Tx-Rx)")
        axs[6].plot(t, g("pathloss_down_db"), lw=0.8, color="tab:purple", label="path loss down")
        axs[6].set_ylabel("path loss (dB)"); axs[6].legend(loc="upper left", fontsize=7)
        a2 = axs[6].twinx()
        a2.plot(t, g("rx_up_cnr_db"), lw=0.6, color="tab:green", alpha=0.7)
        a2.set_ylabel("CNR up (dB)", color="tab:green")
        axs[7].plot(t, g("wireless_range_up"), lw=1.0, ls="--", color="gray", label="PHY rate up")
        axs[7].plot(t, g("wireless_range_down"), lw=1.0, ls=":", color="gray", label="PHY rate down")
        axs[7].plot(t, g("output_rate_down"), lw=0.8, color="tab:blue", label="output down")
        axs[7].plot(t, g("output_rate_up"), lw=0.8, color="tab:orange", label="output up")
        axs[7].set_ylabel("rate (Mbps)"); axs[7].legend(loc="upper right", fontsize=7, ncol=4)
        axs[7].set_xlabel("time since start (s)   [NTG: %d rows, %d without Master reading]"
                          % (len(t), N["n_fail"]))
    else:
        axs[-1].set_xlabel(f"time since start (s)   [no NTG CSV found: {ntg_path}]")

    for ax in axs:
        ax.grid(alpha=0.3)
        for tm, lab in marks:
            ax.axvline(sec(tm), color="magenta", lw=0.8, alpha=0.7)
    for tm, lab in marks:
        axs[0].text(sec(tm), a.fmax * 0.8, lab, color="magenta", fontsize=7, rotation=90, va="top",
                    bbox=dict(fc="white", alpha=0.6, lw=0, pad=0.5))

    plt.tight_layout()
    fig.savefig(out, dpi=80)
    print(out)


if __name__ == "__main__":
    sys.exit(main())
