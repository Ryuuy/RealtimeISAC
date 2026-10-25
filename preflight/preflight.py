#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每天第一次实验前的 10 s 体检（说明见 ../每次做实验前都要确认的事.md）。

    /usr/bin/python3 RealtimeISAC/preflight/preflight.py                # 增益 20 dB，抓 10 s，查 NTG
    /usr/bin/python3 RealtimeISAC/preflight/preflight.py --gain 20 --sec 10 --no-ntg
    /usr/bin/python3 RealtimeISAC/preflight/preflight.py --save-iq      # 顺便存原始 IQ（10 s ≈ 800 MB）

抓 10 s 原始 IQ（int16，不落盘，除非 --save-iq）-> 查四件事 -> 出一张多普勒图：
  1. ch0/ch1 有没有超量程风险（离满量程还差多少 dB）
  2. NTG-2501 的发射功率是不是 5 dBm（现在的 Master 是 192.168.1.101）
  3. TDD 对齐：锁没锁、占空、对比度、是不是单一突发、窗在不在 ON 段里、ON 段漂移
  4. 出图：多普勒谱 + ch0/ch1 窗内电平 + D=P+ − P-（dB re 噪声），时间轴对齐
抓的这 10 s 里前 2 s 请保持静止（D_dB 的噪声基准取自这 2 s）。
必须用 /usr/bin/python3（conda 的 python3 没有 uhd）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.join(os.path.dirname(HERE), "realtime")
sys.path.insert(0, RT)
NTG_DIR = "/home/usrpb210/DASH/DASH_html_old-main/ntg2501_power_logger"

from rt_config import RtConfig            # noqa: E402
from rt_dsp import DopplerEngine          # noqa: E402
from rt_sync import TddSync, _longest_run  # noqa: E402

FS_FULL = 32768.0
OK, WARN, FAIL, SKIP = "✅", "⚠️ ", "❌", "⏭ "
HEADROOM_OK_DB = 12.0     # 人穿过 LOS 时 ch0 会再抬高 6~10 dB（10-25 实测），静止峰值要留出这么多
HEADROOM_FAIL_DB = 6.0
NTG_TX_EXPECT_DBM = 5.0
NTG_TX_TOL_DB = 1.0


def capture(cfg, sec):
    from rt_source import UsrpSource
    n = int(round(sec / cfg.step_sec)); ns = cfg.n_step
    buf = np.empty((2, n * ns * 2), dtype=np.int16)
    src = UsrpSource(cfg).open(); k = 0
    try:
        for raw in src.steps():
            if k >= n:
                break
            buf[:, k * ns * 2:(k + 1) * ns * 2] = raw; k += 1
        errs, dropped = str(src.errors), src.dropped
    finally:
        src.close()
    return buf[:, :k * ns * 2], errs, dropped


def check_clipping(iq, gain):
    rows, worst = [], OK
    for c in range(2):
        mx = n90 = nfs = 0
        for s in range(0, iq.shape[1], 40_000_000):
            a = np.abs(iq[c, s:s + 40_000_000].astype(np.int32))
            mx = max(mx, int(a.max())); n90 += int((a >= 0.9 * FS_FULL).sum()); nfs += int((a >= 32767).sum())
        head = 20 * np.log10(FS_FULL / max(mx, 1))
        rows.append(dict(ch=c, max=mx, pct=mx / FS_FULL * 100, peak_dbfs=-head, headroom_db=head, n90=n90, n_fullscale=nfs))
    r0, r1 = rows
    if r0["n_fullscale"] > 0 or r0["headroom_db"] < HEADROOM_FAIL_DB:
        v = FAIL
    elif r0["headroom_db"] < HEADROOM_OK_DB:
        v = WARN
    else:
        v = OK
    advice = ""
    if v != OK:
        cut = HEADROOM_OK_DB - r0["headroom_db"]
        advice = f"要留够 {HEADROOM_OK_DB:.0f} dB 余量，增益建议降 {cut:.0f} dB（{gain:.0f} -> {gain - cut:.0f} dB）；不要自己改，先问用户"
    ch1_note = ""
    if r1["pct"] < 0.5:
        ch1_note = "ch1 峰值不到 0.5% 满量程，接近量化底，相关会被噪声主导"
    return v, rows, advice, ch1_note


def check_ntg(master_ip, timeout):
    sys.path.insert(0, NTG_DIR)
    try:
        import ntg2501_http_logger as ntg
    except Exception as e:  # noqa: BLE001
        return SKIP, {}, f"没法 import logger: {e}"
    vals, ms, err = ntg.fetch_monitor(master_ip, timeout)
    if err:
        return SKIP, {}, f"连不上 Master {master_ip}（{err}）——网线/IP 对不对？Master 现在应该是 .101"
    f = ntg.as_float
    d = dict(tx_up=f(vals["quality_tx_up_level"]), tx_down=f(vals["quality_tx_down_level"]),
             rx_up=f(vals["quality_rx_up_level"]), rx_down=f(vals["quality_rx_down_level"]),
             cnr_up=f(vals["quality_rx_up_cnr"]), cnr_down=f(vals["quality_rx_down_cnr"]),
             mod_up=vals["up_mod_val"], mod_down=vals["down_mod_val"], query_ms=round(ms))
    if d["rx_up"] is None:
        return WARN, d, f"{master_ip} 返回的 quality_rx_up_level 是空的——查的可能是 Slave，不是 Master"
    bad = [k for k in ("tx_up", "tx_down") if d[k] is None or abs(d[k] - NTG_TX_EXPECT_DBM) > NTG_TX_TOL_DB]
    return (WARN if bad else OK), d, (f"{'/'.join(bad)} 偏离 {NTG_TX_EXPECT_DBM:.0f} dBm 超过 ±{NTG_TX_TOL_DB:.0f} dB（ATPC 在调功率？）" if bad else "")


def fold_profile(iq, ch, npr, frames, bin_samples):
    """ch 的功率按 TDD 周期折叠，bin_samples 个样本一格。frames=(start, stop)。"""
    acc = np.zeros(npr // bin_samples, dtype=np.float64); n = 0
    k0, k1 = frames
    for k in range(k0, k1, 200):
        k2 = min(k + 200, k1)
        seg = iq[ch, k * npr * 2:k2 * npr * 2].reshape(-1, npr, 2).astype(np.float32)
        acc += (seg ** 2).sum(-1).mean(0).reshape(-1, bin_samples).mean(1) * (k2 - k); n += k2 - k
    return acc / max(n, 1)


def run_replay(iq, cfg):
    eng = DopplerEngine(cfg); sync = TddSync(cfg)
    ns = cfg.n_step; K = iq.shape[1] // 2 // ns
    P, amp, ph, ni, lk, ct, du = [], [], [], [], [], [], []
    for k in range(K):
        raw = np.ascontiguousarray(iq[:, k * ns * 2:(k + 1) * ns * 2])
        sync.update(raw); eng.ingest(raw)
        P.append(eng.process(sync.phase, sync.n_integrate).copy()); amp.append(eng.last_amp.copy())
        ph.append(sync.phase); ni.append(sync.n_integrate); lk.append(sync.locked); ct.append(sync.contrast); du.append(sync.duty)
    return eng, np.array(P), np.array(amp), np.array(ph), np.array(ni), np.array(lk), np.array(ct), np.array(du)


def check_tdd(iq, cfg, ph, ni, lk, ct, du, amp):
    npr = cfg.n_period; fs = cfg.sample_rate; P = iq.shape[1] // 2 // npr
    lock_frac = float(lk.mean()); out = dict(lock_frac=lock_frac)
    if lock_frac < 0.5:
        return FAIL, out, ["TDD 没锁（TX 没开？增益/频点不对？）"]
    last = slice(max(0, P - 1000), P)
    prof = fold_profile(iq, 0, npr, (last.start, last.stop), 10)          # 1 µs 一格
    lo, hi = np.percentile(prof, 10), np.percentile(prof, 90); thr = np.sqrt(lo * hi)
    start, length = _longest_run(prof >= thr)
    on = prof >= thr
    n_runs = int((on & ~np.roll(on, 1)).sum())                             # 环形连续 ON 段数（False->True 跳变次数）
    on_db = 10 * np.log10(prof[on].mean() / FS_FULL ** 2); off_db = 10 * np.log10(prof[~on].mean() / FS_FULL ** 2 + 1e-20)
    phase_us = ph[-1] / fs * 1e6; win_us = ni[-1] / fs * 1e6
    m_start = (phase_us - start) % (npr / 10)                             # 窗起点比 ON 起点晚多少 µs
    m_end = length - (m_start + win_us)                                    # 窗终点比 ON 终点早多少 µs
    ph_u = np.unwrap(ph.astype(float), period=npr)
    dur = len(ph) * cfg.step_sec; drift = (ph_u[-1] - ph_u[0]) / fs * 1e6 / dur
    p1 = fold_profile(iq, 1, npr, (last.start, last.stop), 10)
    ch1_contrast = 10 * np.log10(p1[on].mean() / max(p1[~on].mean(), 1e-9))
    fr_db = 20 * np.log10(amp[1:, :, 0].ravel() + 1e-12)            # 去掉第 1 步（启动帧幅度为 0）                     # 每帧 ch0 窗内 dBFS
    gone = float((fr_db < np.median(fr_db) - 25).mean())
    out.update(contrast=float(np.median(ct)), duty=float(np.median(du)), phase_us=float(phase_us), win_us=float(win_us),
               on_us=float(length), n_runs=n_runs, on_dbfs=float(on_db), off_dbfs=float(off_db), on_off_db=float(on_db - off_db),
               margin_start_us=float(m_start), margin_end_us=float(m_end), drift_us_per_s=float(drift),
               ch1_on_off_db=float(ch1_contrast), frames_gone_frac=gone)
    notes, worst = [], OK
    def bump(level, msg):
        nonlocal worst
        notes.append(msg)
        if level == FAIL or (level == WARN and worst == OK):
            worst = level
    if out["contrast"] < 0.3: bump(FAIL, f"对比度 {out['contrast']:.2f} < 0.3")
    elif out["contrast"] < 0.8: bump(WARN, f"对比度 {out['contrast']:.2f} < 0.8")
    if not (0.68 <= out["duty"] <= 0.74): bump(WARN, f"占空 {out['duty']*100:.1f}% 不在 68–74%（正常约 70–71.5%，29%/39% 是 10-25 锁错窗的症状）")
    if n_runs != 1: bump(FAIL, f"ch0 有 {n_runs} 个 ON 突发（应为 1 个，Master/Slave 两个突发都强说明会跳窗）")
    if m_start < 0 or m_end < 0: bump(WARN, f"积分窗越出 ON 段（起点余量 {m_start:.0f} µs，终点余量 {m_end:.0f} µs）")
    if out["on_off_db"] < 30: bump(WARN, f"ch0 ON/OFF 只差 {out['on_off_db']:.0f} dB（正常 ≥ 30 dB）")
    if abs(drift) > 15: bump(WARN, f"ON 段漂移 {drift:+.1f} µs/s（正常 2–4 µs/s；sync 每 0.2 s 重估，超过 ~25 µs/s 会跟不上）")
    if ch1_contrast < 6: bump(WARN, f"ch1 的 ON/OFF 只差 {ch1_contrast:.1f} dB（<6 dB：ch1 基本没收到突发，相关会是噪声）")
    if gone > 0.005: bump(WARN, f"{gone*100:.1f}% 的帧 ch0 信号消失（<中位数-25 dB）：链路在抖/断")
    return worst, out, notes


def plot(P, amp, eng, cfg, out_png, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fr = eng.freqs; K = len(P); t = np.arange(K) * cfg.step_sec
    pb = (fr >= 20) & (fr <= 500); nb = (fr <= -20) & (fr >= -500); N = int(pb.sum())
    st = (t >= 0.4) & (t < 2.0); fl = float(P[st][:, pb | nb].mean()); ref = N * fl
    D = P[:, pb].sum(1) - P[:, nb].sum(1); Ddb = np.sign(D) * 10 * np.log10(np.abs(D) / ref + 1e-12)
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    Dd = 10 * np.log10(P + 1e-20)
    im = ax[0].pcolormesh(t, fr, (Dd - Dd.max()).T, cmap="jet", vmin=-60, vmax=0, shading="auto")
    ax[0].set_ylabel("Doppler Hz"); ax[0].set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, location="right", fraction=0.025, pad=0.01, label="dB rel. max")
    ax[1].plot(t, 20 * np.log10(amp[:, :, 0].mean(1) + 1e-12), label="ch0"); ax[1].plot(t, 20 * np.log10(amp[:, :, 1].mean(1) + 1e-12), label="ch1")
    ax[1].set_ylabel("in-window dBFS"); ax[1].legend(loc="upper right", fontsize=8); ax[1].grid(alpha=.3)
    ax[2].plot(t, Ddb, lw=.6, alpha=.5); ax[2].plot(t, np.convolve(Ddb, np.ones(5) / 5, "same"), lw=1.5)
    ax[2].axhline(0, c="k", lw=.6); ax[2].set_ylabel("D_dB re noise"); ax[2].set_xlabel("s"); ax[2].grid(alpha=.3); ax[0].set_xlim(0, t[-1])
    fig.text(0.01, 0.0, "D = P+ - P-, P+ = sum|S[f]|^2 (+20..+500Hz), P- = same for -500..-20Hz; D_dB = sign(D)*10log10(|D|/(N*floor)), floor from the first 0.4-2.0 s (keep still)",
             fontsize=7, va="bottom", family="monospace")
    fig.savefig(out_png, dpi=85); plt.close(fig)
    return fl


def main():
    ap = argparse.ArgumentParser(description="实验前 10 s 体检")
    ap.add_argument("--gain", type=float, default=20.0); ap.add_argument("--sec", type=float, default=10.0)
    ap.add_argument("--fs", type=float, default=10.0, help="采样率 MHz")
    ap.add_argument("--serial", default="32392D3"); ap.add_argument("--ntg-ip", default="192.168.1.101")
    ap.add_argument("--no-ntg", action="store_true"); ap.add_argument("--save-iq", action="store_true")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = a.out_dir or os.path.join(os.path.dirname(HERE), "preflight_runs", stamp); os.makedirs(out, exist_ok=True)
    cfg = RtConfig(serial=a.serial, sample_rate=a.fs * 1e6, gain=a.gain)
    print(f"== 实验前体检 {stamp}  gain={a.gain:.0f} dB  {a.sec:.0f} s  （前 2 s 请保持静止）==")

    ntg_v, ntg, ntg_msg = (SKIP, {}, "已跳过（--no-ntg）") if a.no_ntg else check_ntg(a.ntg_ip, 3.0)
    iq, errs, dropped = capture(cfg, a.sec)
    print(f"抓到 {iq.shape[1]/2/cfg.sample_rate:.1f} s x2ch   UHD: {errs}  丢步={dropped}")
    clip_v, clip, clip_adv, ch1_note = check_clipping(iq, a.gain)
    eng, P, amp, ph, ni, lk, ct, du = run_replay(iq, cfg)
    tdd_v, tdd, tdd_notes = check_tdd(iq, cfg, ph, ni, lk, ct, du, amp)

    print("\n[1] 量程（ch0 是参考通道，最怕它削顶）")
    for r in clip:
        print(f"    ch{r['ch']}: 峰值 {r['max']:5d} = {r['pct']:5.1f}% 满量程 ({r['peak_dbfs']:+.1f} dBFS)  离满量程还差 {r['headroom_db']:.1f} dB  "
              f">=90%FS 样本 {r['n90']}  到顶样本 {r['n_fullscale']}")
    print(f"    {clip_v} ch0：{'余量 ≥ ' + str(int(HEADROOM_OK_DB)) + ' dB，人穿过 LOS（+6~10 dB）也不会顶' if clip_v == OK else clip_adv}")
    if ch1_note: print(f"    {WARN} {ch1_note}")
    print(f"\n[2] NTG-2501（Master {a.ntg_ip}，期望两个方向发射都是 {NTG_TX_EXPECT_DBM:.0f} dBm ±{NTG_TX_TOL_DB:.0f}）")
    if ntg:
        print(f"    发射 上行(Slave) {ntg['tx_up']} dBm  下行(Master) {ntg['tx_down']} dBm | 接收 上行 {ntg['rx_up']} 下行 {ntg['rx_down']} dBm | CNR 上行 {ntg['cnr_up']} 下行 {ntg['cnr_down']} dB | 调制 {ntg['mod_up']}/{ntg['mod_down']}")
    print(f"    {ntg_v} {ntg_msg or '发射功率正常'}")
    print("\n[3] TDD 对齐")
    if "contrast" in tdd:
        print(f"    锁定 {tdd['lock_frac']*100:.0f}%  对比度 {tdd['contrast']:.2f}  占空 {tdd['duty']*100:.1f}%  ON 段 {tdd['on_us']:.0f} µs  ON 突发数 {tdd['n_runs']}")
        print(f"    ch0 ON {tdd['on_dbfs']:.1f} dBFS / OFF {tdd['off_dbfs']:.1f} dBFS（差 {tdd['on_off_db']:.0f} dB）  ch1 ON/OFF 差 {tdd['ch1_on_off_db']:.1f} dB")
        print(f"    积分窗 [{tdd['phase_us']%1000:.0f}, {(tdd['phase_us']+tdd['win_us'])%1000:.0f}) µs（对 1000 µs 取模），距 ON 起点 {tdd['margin_start_us']:.0f} µs / 距 ON 终点 {tdd['margin_end_us']:.0f} µs  ON 段漂移 {tdd['drift_us_per_s']:+.1f} µs/s  信号消失帧 {tdd['frames_gone_frac']*100:.1f}%")
    print(f"    {tdd_v} {'; '.join(tdd_notes) if tdd_notes else 'TDD 锁定正常，单一突发，窗在 ON 段内'}")

    verdicts = [clip_v, tdd_v] + ([ntg_v] if ntg_v != SKIP else [])
    overall = FAIL if FAIL in verdicts else (WARN if WARN in verdicts else OK)
    png = os.path.join(out, "preflight.png")
    tag = {OK: "OK", WARN: "WARN", FAIL: "FAIL", SKIP: "skipped"}
    fl = plot(P, amp, eng, cfg, png, f"preflight {stamp}  gain {a.gain:.0f} dB   clipping {tag[clip_v]}   NTG {tag[ntg_v]}   TDD {tag[tdd_v]}")
    rep = dict(stamp=stamp, gain=a.gain, sec=a.sec, overall=overall, clipping=dict(verdict=clip_v, rows=clip, advice=clip_adv, ch1_note=ch1_note),
               ntg=dict(verdict=ntg_v, values=ntg, msg=ntg_msg), tdd=dict(verdict=tdd_v, values=tdd, notes=tdd_notes), noise_floor_per_bin=fl, uhd=errs, dropped=int(dropped))
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), ensure_ascii=False, indent=1, default=float)
    if a.save_iq:
        np.savez(os.path.join(out, f"iq_{stamp}.npz"), iq=iq, fs=cfg.sample_rate, gain=cfg.gain)
    print(f"\n总评 {overall}   图: {png}\n      报告: {os.path.join(out, 'report.json')}")
    return 0 if overall != FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
