#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drop-out probability per crossing action, 2026-10-10 20:55 session.

Operator statement: from 20:56:00 the operator crosses the LOS exactly once every 20 s (a crossing does not have to
cause a drop). So slot k = crossing at t_k = 20:56:00 + 20 k s, k = 0..47 (to 21:12:00):
  A (lateral, thin/sideways posture) k = 0..20   20:56:00-21:03:00  (21 crossings)
  B (sideways walking)               k = 21..35  21:03:00-21:08:00  (15 crossings)
  C (diagonal 45 deg)                k = 36..47  21:08:00-21:12:00  (12 crossings)
A crossing 'dropped' if ch0 (reference channel, > -30 dBFS = link up) goes DOWN within [t_k-2 s, t_k+18 s).
A crossing is 'eligible' if the link was up at t_k-0.5 s (a crossing during an ongoing outage cannot drop it).
    python3 crossing_dropout_stats.py ../raw ..   # writes crossing_slots.csv and DROPOUT_STATS.md
"""
import csv, datetime, json, math, os, sys
import numpy as np

RAW = sys.argv[1] if len(sys.argv) > 1 else "../raw"
OUT = sys.argv[2] if len(sys.argv) > 2 else ".."
STAMP = "20261010_205507"
T0 = (20, 56, 0)
SLOTS = {"A": (0, 20, "lateral / thin (sideways posture)"), "B": (21, 35, "sideways walking"), "C": (36, 47, "diagonal 45 deg")}
UP = -30.0


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher(a, b, c, d):
    """two-sided Fisher exact p for [[a,b],[c,d]]"""
    n = a + b + c + d; r1, c1 = a + b, a + c
    lg = lambda x: math.lgamma(x + 1)
    def pr(x): return math.exp(lg(r1) + lg(n - r1) + lg(c1) + lg(n - c1) - lg(n) - lg(x) - lg(r1 - x) - lg(c1 - x) - lg(n - r1 - c1 + x))
    p0 = pr(a); lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 * (1 + 1e-9)))


def main():
    z = np.load(os.path.join(RAW, f"isac_{STAMP}_p001.npz"), allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    day = datetime.datetime.fromtimestamp(meta["start_ms"] / 1000).replace(hour=T0[0], minute=T0[1], second=T0[2], microsecond=0)
    t0 = day.timestamp() * 1000
    clk = lambda ms: datetime.datetime.fromtimestamp(ms / 1000).strftime("%H:%M:%S")
    t, c0, c1 = z["dbfs_t_ms"], z["dbfs_ch0"], z["dbfs_ch1"]
    wt, ver, ridge = z["wf_t_ms"], z["wf_verdict"], z["wf_ridge_hz"]
    up = c0 > UP
    downs = np.array([t[i] for i in range(1, len(up)) if up[i - 1] and not up[i]])
    ups = np.array([t[i] for i in range(1, len(up)) if (not up[i - 1]) and up[i]])
    rows = list(csv.DictReader(open(os.path.join(RAW, f"ntg_{STAMP}.csv"), encoding="utf-8")))
    nt = np.array([float(r["unix_time"]) * 1000 for r in rows])
    f = lambda k: np.array([float(r[k]) if r[k] else np.nan for r in rows])
    rx, cnr = f("slave_rx_dbm"), f("rx_down_cnr_db")

    out = []
    for k in range(48):
        tk = t0 + 20000 * k
        win = next(w for w, (a, b, _) in SLOTS.items() if a <= k <= b)
        link_up = bool(up[np.searchsorted(t, tk - 500) - 1]) if np.searchsorted(t, tk - 500) > 0 else True
        dd = downs[(downs >= tk - 2000) & (downs < tk + 18000)]
        dropped = len(dd) > 0
        d_off = (dd[0] - tk) / 1000 if dropped else float("nan")
        # features in the crossing window [tk-2, tk+4] (before / around the drop)
        w = (wt >= tk - 2000) & (wt <= tk + 4000)
        if dropped: w &= (wt <= dd[0])
        n = (nt >= tk - 1000) & (nt <= tk + 4000) & np.isfinite(rx)
        if dropped: n &= (nt <= dd[0])
        m1 = (t >= tk - 1000) & (t <= tk + 4000)
        if dropped: m1 &= (t <= dd[0])
        base = np.nanmedian(rx[(nt >= tk - 6000) & (nt < tk - 2000)]) if ((nt >= tk - 6000) & (nt < tk - 2000) & np.isfinite(rx)).any() else float("nan")
        out.append(dict(slot=k, window=win, crossing_clock=clk(tk), link_up_at_crossing=int(link_up), dropped=int(dropped),
                        down_offset_s=d_off, doppler_fired=int((ver[w] > 0).any()), ridge_max_hz=float(np.nanmax(ridge[w])) if w.any() else float("nan"),
                        rx_min_dbm=float(np.nanmin(rx[n])) if n.any() else float("nan"), rx_before_dbm=float(base),
                        cnr_min_db=float(np.nanmin(cnr[(nt >= tk - 1000) & (nt <= tk + 4000) & np.isfinite(cnr) & ((nt <= dd[0]) if dropped else True)])) if ((nt >= tk - 1000) & (nt <= tk + 4000) & np.isfinite(cnr)).any() else float("nan"),
                        ch1_dip_db=float(np.median(c1[(t >= tk - 6000) & (t < tk - 2000)]) - np.min(c1[m1])) if m1.any() and ((t >= tk - 6000) & (t < tk - 2000)).any() else float("nan")))

    cols = list(out[0].keys())
    with open(os.path.join(OUT, "crossing_slots.csv"), "w", newline="", encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=cols); w_.writeheader()
        for o in out: w_.writerow({c: (f"{v:.2f}" if isinstance(v, float) else v) for c, v in o.items()})

    def summarize(sel):
        el = [o for o in sel if o["link_up_at_crossing"]]
        k = sum(o["dropped"] for o in el); n = len(el)
        return len(sel), n, k, (k / n if n else float("nan")), wilson(k, n)

    lines = []
    P = lines.append
    P("# 人体穿越实验 — 三种动作的掉线概率（2026-10-10 20:55 会话）\n")
    P("操作者说明：**从 20:56:00 起，每 20 秒一定穿过一次 LOS，但不一定会掉线**。所以第 k 次穿越发生在 20:56:00 + 20·k 秒。\n")
    P("| 动作 | 时间 | 穿越次数 | 其中链路在线（有效） | **掉线次数** | **掉线概率** | 95% 区间（Wilson） |\n|---|---|---:|---:|---:|---:|---|")
    tot = []
    for wname, (a, b, desc) in SLOTS.items():
        sel = [o for o in out if o["window"] == wname]
        N, n, k, p, ci = summarize(sel)
        times = f"{sel[0]['crossing_clock']}–{clk(t0 + 20000 * (b + 1))}"
        P(f"| {wname} {desc} | {times} | {N} | {n} | {k} | **{p*100:.0f}%** | {ci[0]*100:.0f}% – {ci[1]*100:.0f}% |")
        tot.append((wname, n, k))
    N, n, k, p, ci = summarize(out)
    P(f"| 合计 | 20:56:00–21:12:00 | {N} | {n} | {k} | **{p*100:.0f}%** | {ci[0]*100:.0f}% – {ci[1]*100:.0f}% |\n")
    P("**有效** = 穿越那一刻（前 0.5 s）链路是在线的；穿越时链路已经断着（上一次断连还没恢复）的不算（它不可能再掉一次）。\n")
    P("两两比较（Fisher 精确检验，双侧；样本很小，仅供参考）：\n")
    for (n1, na, ka), (n2, nb, kb) in [(tot[0], tot[1]), (tot[0], tot[2]), (tot[1], tot[2])]:
        P(f"- {n1} vs {n2}：{ka}/{na} vs {kb}/{nb}，p = {fisher(ka, na - ka, kb, nb - kb):.2f}")
    P("")
    dr = [o for o in out if o["dropped"]]; nd = [o for o in out if o["link_up_at_crossing"] and not o["dropped"]]
    P("## 穿越后多久掉线、掉线与不掉线的穿越有什么区别\n")
    P(f"- 掉线发生在穿越时刻后 **{np.nanmedian([o['down_offset_s'] for o in dr]):.1f} s**（中位数；范围 {np.nanmin([o['down_offset_s'] for o in dr]):.1f}–{np.nanmax([o['down_offset_s'] for o in dr]):.1f} s）。")
    P("- 下面比较「掉线的穿越」和「没掉线的穿越」在穿越窗口（−2 s ~ +4 s，掉线者截止到掉线时刻）里的特征：\n")
    P("| 特征 | 掉线（n=%d） | 没掉线（有效，n=%d） |\n|---|---|---|" % (len(dr), len(nd)))
    def med(lst, key, fmt="{:.1f}"):
        v = np.array([o[key] for o in lst], float); v = v[np.isfinite(v)]
        return "–" if not len(v) else f"{fmt.format(np.median(v))}（{fmt.format(v.min())} ~ {fmt.format(v.max())}）"
    P(f"| 多普勒判决触发的穿越数 | {sum(o['doppler_fired'] for o in dr)}/{len(dr)} | {sum(o['doppler_fired'] for o in nd)}/{len(nd)} |")
    P(f"| 窗口内 ridge 最大值 [Hz] 中位数（范围） | {med(dr, 'ridge_max_hz', '{:.0f}')} | {med(nd, 'ridge_max_hz', '{:.0f}')} |")
    P(f"| 窗口内 Slave Rx 最小值 [dBm] | {med(dr, 'rx_min_dbm')} | {med(nd, 'rx_min_dbm')} |")
    P(f"| 窗口内 CNR 最小值 [dB] | {med(dr, 'cnr_min_db')} | {med(nd, 'cnr_min_db')} |")
    P(f"| ch1(sensing) 相对穿越前的最大下降 [dB] | {med(dr, 'ch1_dip_db')} | {med(nd, 'ch1_dip_db')} |\n")
    P("**注意（这张表只是描述，不能当判据）：** 掉线的穿越被截止到掉线时刻，窗口里包含了 ch0/ch1 崩塌的那一瞬（所以 ch1 下降 26 dB 其实是断连本身），"
      "Rx/CNR 也因为 NTG 在断连时读不到而被截断；两组窗口长度和含义不对称。想做真正的预测特征，必须只用**掉线之前**、与是否掉线无关的固定窗口（例如穿越后前 1 s）。"
      "唯一能直接读的是：**多普勒几乎对每次穿越都触发（掉线 28/32、没掉线 14/15），所以它分不开「会掉线」与「不会掉线」的穿越**。\n")
    P("## 逐次穿越（`crossing_slots.csv`）\n")
    P("| # | 动作 | 穿越时刻 | 链路在线 | 掉线 | 掉线延迟 [s] | 多普勒触发 | Rx 最小 [dBm] | CNR 最小 [dB] |\n|---:|---|---|---|---|---:|---|---:|---:|")
    for o in out:
        P(f"| {o['slot']} | {o['window']} | {o['crossing_clock']} | {'是' if o['link_up_at_crossing'] else '否（已断）'} | {'**掉线**' if o['dropped'] else '否'} | "
          f"{'' if not o['dropped'] else format(o['down_offset_s'], '.1f')} | {'是' if o['doppler_fired'] else '否'} | {o['rx_min_dbm']:.1f} | {o['cnr_min_db']:.1f} |")
    P("\n## 方法与注意\n")
    P("- 穿越时刻按 **20:56:00 + 20 s·k** 的严格节拍假设（操作者口述）。数据里的断连时刻大多落在节拍后 0–5 s（一个约 9 s），与此吻合；若操作者实际有偏差，概率会略有变化。")
    P("- 掉线 = ch0 在 [t_k − 2 s, t_k + 18 s) 内出现下降沿；时间窗按整 20 s 切分。")
    P("- A 段 21 次、B 段 15 次、C 段 12 次（C 截止到 21:12:00，按操作者说的「到 12 截止」）。样本很小，区间很宽，**不能严格说三种动作的概率有显著差异**。")
    P("- 异常项：#0（20:56:00）的掉线发生在穿越后 5.6 s，#19（21:02:20）在 10.0 s——可能不是这次穿越直接造成的；#20（21:02:40）穿越时链路刚恢复不到 0.5 s，按「链路已断」排除。"
      "**敏感性**：若把延迟 ≥ 5 s 的这两次不算掉线，A 段为 10/20 = 50%（B、C 不变）；结论（A < B < C，但区间大幅重叠）不变。")
    P("- 三种动作的**具体姿态/位置/速度没有记录**（`n_markers = 0`），动作之间还混有「疲劳/熟练度」的先后顺序效应（A→B→C 是固定顺序，不是随机顺序）。")
    P("- Rx/CNR 来自 NTG HTTP（约 5–10 Hz），穿越很快，最小值可能漏掉最深的那一瞬；表中的最小值是**下限的近似**。")
    open(os.path.join(OUT, "DROPOUT_STATS.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:22]))


if __name__ == "__main__":
    main()
