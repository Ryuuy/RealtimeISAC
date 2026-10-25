#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NTG 链路"通了"之后，数据面（ping / HTTP）多久才恢复？—— 和 NTG csv 的 master_valid 对时间用。

    /usr/bin/python3 link_dataplane_probe.py --sec 540            # 默认探测 192.168.1.5:8080（Windows server）

三路同时跑（都经过 NTG 无线链路，和浏览器走的是同一条路）：
  ping  : 系统 ping，每 0.2 s 一个包，-D 带 unix 时间戳，-O 报告没回的包
  newc  : 每 0.5 s 新建一条 TCP 连接取一个 ~25 KB 的小分片（3 s 超时，和页面的 XHR 一致）
  keep  : 一直复用同一条 HTTP keep-alive 连接取同一个分片（模拟浏览器复用连接；超时/出错后是否能自己恢复）
输出到 probe_runs/<时间戳>/{ping.log,newc.csv,keep.csv}，时间列都是本机 unix time（和 isac_logs/ntg_*.csv 同一个钟）。
对页面是只读 GET 一个小文件，负载 ~2×50 KB/s，不影响视频以外的任何东西。
"""
import argparse
import csv
import http.client
import os
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "probe_runs")
PATH = "/tos_2s_1080p/tos_2s_100kbit/tos_2s5.m4s"


def newc(host, port, t_end, out):
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["unix_time", "ok", "http", "bytes", "t_connect_ms", "t_total_ms", "err"])
        while time.time() < t_end:
            t0 = time.time(); ok = 0; code = ""; n = 0; tc = ""; err = ""
            try:
                c = http.client.HTTPConnection(host, port, timeout=3)
                c.connect(); tc = f"{(time.time()-t0)*1000:.0f}"
                c.request("GET", PATH, headers={"Connection": "close", "Cache-Control": "no-cache"})
                r = c.getresponse(); code = r.status; n = len(r.read()); ok = int(r.status == 200); c.close()
            except Exception as e:  # noqa: BLE001
                err = type(e).__name__
            w.writerow([f"{t0:.3f}", ok, code, n, tc, f"{(time.time()-t0)*1000:.0f}", err]); f.flush()
            time.sleep(max(0, 0.5 - (time.time() - t0)))


def keep(host, port, t_end, out):
    conn = None
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["unix_time", "ok", "http", "bytes", "t_total_ms", "err", "reconnected"])
        while time.time() < t_end:
            t0 = time.time(); ok = 0; code = ""; n = 0; err = ""; rec = 0
            try:
                if conn is None:
                    conn = http.client.HTTPConnection(host, port, timeout=3); rec = 1
                conn.request("GET", PATH, headers={"Cache-Control": "no-cache"})
                r = conn.getresponse(); code = r.status; n = len(r.read()); ok = int(r.status == 200)
            except Exception as e:  # noqa: BLE001
                err = type(e).__name__
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
                conn = None            # 下一轮重连（浏览器遇到超时也会丢掉这条连接，但可能晚一点）
            w.writerow([f"{t0:.3f}", ok, code, n, f"{(time.time()-t0)*1000:.0f}", err, rec]); f.flush()
            time.sleep(max(0, 0.5 - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.1.5"); ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--sec", type=float, default=540)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    out = a.out_dir or os.path.join(OUT_ROOT, time.strftime("%Y%m%d_%H%M%S")); os.makedirs(out, exist_ok=True)
    t_end = time.time() + a.sec
    print(f"probe -> {out}  ({a.sec:.0f} s, start {time.strftime('%H:%M:%S')}, end {time.strftime('%H:%M:%S', time.localtime(t_end))})", flush=True)
    pg = subprocess.Popen(["ping", "-D", "-O", "-i", "0.2", "-W", "1", "-w", str(int(a.sec)), a.host],
                          stdout=open(os.path.join(out, "ping.log"), "w"), stderr=subprocess.STDOUT)
    th = [threading.Thread(target=newc, args=(a.host, a.port, t_end, os.path.join(out, "newc.csv"))),
          threading.Thread(target=keep, args=(a.host, a.port, t_end, os.path.join(out, "keep.csv")))]
    for t in th: t.start()
    for t in th: t.join()
    pg.wait(timeout=10)
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
