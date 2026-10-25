#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 10-25 当天的实验数据整理进 data_archive/20261025/（在家分析用）。可以重复运行：只复制变化过的文件（边录边增长的 h_logs/isac_logs 会更新）。

    /usr/bin/python3 sync_archive_20261025.py            # 复制 + 重写 MANIFEST.md
    /usr/bin/python3 sync_archive_20261025.py --events   # 之后再切事件窗口（cut_events.py）

布局（沿用 data_archive/MANIFEST.md 的约定：isac_logs 被 .gitignore 忽略，所以存档时叫 sessions_raw）：
  h_logs/              rt_main --save-h：dec(复数) / 每帧 ch0 ch1 幅度 / 时间戳 / 标志
  sessions_raw/        bridge 的会话(.npz)、NTG XML csv(10 Hz)、六状态 csv、ping csv、日志
  dash_snapshots_gz/   浏览器 Save Full Snapshot（无损 gzip -9）
  probe_runs/          无头页面 + 数据面探测（不含 chrome_profile）
  dash_update_packages/ 给 Windows server 的更新包（DASH 客户端代码）
  events/              cut_events.py 的输出
没有存的：原始 IQ（debug/iq_20261025_*.npz，每份 1.6–2.4 GB，超过 GitHub 100 MB 限制，只在实验室电脑上）。
"""
import argparse
import glob
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ARC = os.path.join(ROOT, "data_archive", "20261025")
HOME = os.path.expanduser("~")
DAY = "20261025"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def copy_if_changed(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src) and os.path.getmtime(dst) >= os.path.getmtime(src) - 1:
        return False
    shutil.copy2(src, dst); return True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--events", action="store_true"); a = ap.parse_args()
    n_new = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "h_logs", f"h_{DAY}_*"))):
        if os.path.getsize(f) == 0:
            continue
        n_new += copy_if_changed(f, os.path.join(ARC, "h_logs", os.path.basename(f)))
    for f in sorted(glob.glob(os.path.join(ROOT, "isac_logs", f"*{DAY}*"))):
        n_new += copy_if_changed(f, os.path.join(ARC, "sessions_raw", os.path.basename(f)))
    for f in sorted(glob.glob(os.path.join(HOME, "Downloads", "dashSnapshot_2026-10-25_*.json"))):
        dst = os.path.join(ARC, "dash_snapshots_gz", os.path.basename(f) + ".gz")
        if not (os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(f)):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(f, "rb") as fi, gzip.GzipFile(dst, "wb", compresslevel=9, mtime=0) as fo:
                shutil.copyfileobj(fi, fo)
            n_new += 1
    for d in sorted(glob.glob(os.path.join(ROOT, "probe_runs", f"{DAY}_*"))):
        for dp, dn, fn in os.walk(d):
            dn[:] = [x for x in dn if x != "chrome_profile"]
            for f in fn:
                src = os.path.join(dp, f); rel = os.path.relpath(src, os.path.join(ROOT, "probe_runs"))
                n_new += copy_if_changed(src, os.path.join(ARC, "probe_runs", rel))
    for f in sorted(glob.glob(os.path.join(HOME, "DASH_update_20261025", "*.zip")) + glob.glob(os.path.join(HOME, "DASH_update_20261025", "README*")) + glob.glob(os.path.join(HOME, "DASH_update_20261025", "SHA256SUMS*"))):
        n_new += copy_if_changed(f, os.path.join(ARC, "dash_update_packages", os.path.basename(f)))
    print(f"copied/updated {n_new} files")

    if a.events:
        subprocess.run([sys.executable, os.path.join(HERE, "cut_events.py"), "--archive", ARC, "--runs", "P1", "F", "G", "H"], check=False)

    rows = []; big = []
    for dp, dn, fn in os.walk(ARC):
        for f in sorted(fn):
            p = os.path.join(dp, f)
            if f == "MANIFEST.md":
                continue
            sz = os.path.getsize(p); rows.append((os.path.relpath(p, ARC), sz, sha256(p)))
            if sz > 90 * 1024 * 1024:
                big.append((p, sz))
    rows.sort()
    with open(os.path.join(ARC, "MANIFEST.md"), "w", encoding="utf-8") as m:
        m.write(f"# data_archive/20261025 — 2026-10-25 实验数据（在家分析用）\n\n生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}（`sync_archive_20261025.py`，可重复运行）。\n\n")
        m.write("先读 `../../20261025_测量记录_论文用.md`（结论和限制）和 `walk_schedule.csv`（每一轮走了什么、几点开始、多少次）。\n\n")
        m.write("- `h_logs/h_20261025_<时间戳>.*`：dec（复数 complex64，1 kHz）、每帧 ch0/ch1 幅度、时间戳、可用标志。用 `debug/load_h.py` 或 `experiments/20261025_b210_doppler_checks/scripts/cut_events.py` 里的 `load_h()` 读。\n")
        m.write("- `sessions_raw/`：`isac_*.npz`（多普勒瀑布、检测、ch0/ch1 dBFS）、`ntg_*.csv`（NTG XML，10 Hz，`master_valid`）、`ntgstate_*.csv`（NTG 六状态）、`dataplane_*.csv`（ping 5 Hz）。**NTG 管理口在 21:21:48–~21:36 卡死，csv 里那段 625 s 的“断连”不是链路断，是接口没回应（ping 100% 通）。**\n")
        m.write("- `dash_snapshots_gz/`：浏览器快照（gzip -9，`gunzip` 后逐字节还原）；`telemetry` 里有 `requests`（每个分片请求含失败/超时）、`ntgStates`、`dataplane`（19:55 之后的快照才有这三项）。\n")
        m.write("- `probe_runs/`：无头页面 + 探测（requests.csv / state.csv / console.log / ping.log / newc.csv / keep.csv / ntg_states.csv）。\n")
        m.write("- `events/`：按排定时刻切好的事件窗口（窗口 [S−5, S+间隔−5)，10 s 一次的就是 ±5 s）：`<run>_events.npz`（dec、amp0、amp1）和 `<run>_slots.csv`（每个事件的汇总）。\n")
        m.write("- **没有存**：原始 IQ（`debug/iq_20261025_*.npz`，每份 1.6–2.4 GB，超过 GitHub 100 MB 限制）。\n\n")
        m.write("| 文件 | 大小 [B] | sha256 |\n|---|---:|---|\n")
        for rel, sz, sh in rows:
            m.write(f"| `{rel}` | {sz} | `{sh}` |\n")
    tot = sum(r[1] for r in rows)
    print(f"archive: {len(rows)} files, {tot/1e6:.1f} MB; files > 90 MB: {big or 'none'}")
    return 1 if big else 0


if __name__ == "__main__":
    sys.exit(main())
