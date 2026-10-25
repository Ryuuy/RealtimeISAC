#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NTG 状态 + 数据面通不通：给 isac_bridge.py 用的两个后台监视器（client 侧，不动 server）。

1. NtgStateMonitor  每 1/hz 秒 POST http://<Master>/main_frm_xml.cgi（NTG 网页自己每 1 s 轮询的接口，不用登录），
   读 装置状態 / 無線状態 / 認証状態 / 監視信号状態 / イーサネットリンク状態(自局/対向局)。
   全部行写 csv；状态有变化、或每 5 s 心跳，就回调一次 callback({"type":"ntg_state", ...})。
2. DataplaneMonitor 系统 ping（-i 0.2）到视频 server（默认 192.168.1.5，Windows），逐包写 csv；
   连续 3 个有回应 = up、连续 3 个没回应 = down，状态翻转、或每 5 s 心跳，回调 {"type":"dataplane", ...}。

回调里拿到的都是 Ubuntu 本机时钟（ms），和 isac_logs/ntg_*.csv / h_logs 同一个钟。页面（isac_ws_client.js）收到后记进快照。
用 1 Hz 是因为 NTG 网页自己就是 1 Hz；5 Hz 那一轮疑似影响了设备（见 20261025_测量记录_论文用.md D 轮）。
"""
import csv
import os
import re
import subprocess
import threading
import time
import urllib.request

TAGS = ["sys_status", "modem_sync_status", "auth_status", "signal_rx_status", "ether_if_0_now", "ether_if_1_now"]


class NtgStateMonitor(threading.Thread):
    def __init__(self, host, out_csv, callback=None, hz=1.0, timeout=2.0):
        super().__init__(daemon=True)
        self.host, self.out_csv, self.callback, self.period, self.timeout = host, out_csv, callback, 1.0 / hz, timeout
        self._stopev = threading.Event()
        self.last = None

    def stop(self):
        self._stopev.set()

    def run(self):
        last_emit = 0.0
        with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["unix_time", "ok", "query_ms"] + TAGS + [t + "_stat" for t in TAGS])
            while not self._stopev.is_set():
                t0 = time.time(); row = None; ok = 0; st = {}
                try:
                    req = urllib.request.Request(f"http://{self.host}/main_frm_xml.cgi", data=b"", method="POST")
                    xml = urllib.request.urlopen(req, timeout=self.timeout).read().decode("utf-8", "replace")
                    vals = [re.search(rf"<{t}>(.*?)</{t}>", xml, re.S) for t in TAGS]
                    stats = [re.search(rf"<{t}_stat>(.*?)</{t}_stat>", xml, re.S) for t in TAGS]
                    st = {t: (m.group(1) if m else "") for t, m in zip(TAGS, vals)}
                    ok = 1
                    row = [f"{t0:.3f}", 1, f"{(time.time()-t0)*1000:.0f}"] + [st[t] for t in TAGS] + [(m.group(1) if m else "") for m in stats]
                except Exception as e:  # noqa: BLE001
                    row = [f"{t0:.3f}", 0, f"{(time.time()-t0)*1000:.0f}", type(e).__name__] + [""] * 11
                w.writerow(row); f.flush()
                key = tuple(st.get(t, "") for t in TAGS) if ok else ("<no answer>",)
                now = time.time()
                if self.callback and (key != self.last or now - last_emit >= 5.0):
                    try:
                        self.callback({"type": "ntg_state", "ts": round(t0 * 1000.0, 3), "ok": ok, "states": st, "changed": key != self.last})
                    except Exception:  # noqa: BLE001
                        pass
                    last_emit = now
                self.last = key
                self._stopev.wait(max(0.0, self.period - (time.time() - t0)))


class DataplaneMonitor(threading.Thread):
    def __init__(self, host, out_csv, callback=None, interval=0.2, need=3):
        super().__init__(daemon=True)
        self.host, self.out_csv, self.callback, self.interval, self.need = host, out_csv, callback, interval, need
        self._stopev = threading.Event(); self.proc = None
        self.state = None            # True = up, False = down

    def stop(self):
        self._stopev.set()
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:  # noqa: BLE001
                pass

    def run(self):
        run_ok = run_bad = 0; last_emit = 0.0; last_rtt = None
        with open(self.out_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["unix_time", "ok", "rtt_ms"])
            # -D: 行首带 unix 时间戳；-O: 没回应的包也打印一行；-w 很大：一直 ping 到被 stop
            self.proc = subprocess.Popen(["ping", "-D", "-O", "-i", str(self.interval), "-W", "1", self.host],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            for line in self.proc.stdout:
                if self._stopev.is_set():
                    break
                m = re.match(r"\[(\d+\.\d+)\] 64 bytes.*time=([\d.]+) ms", line)
                if m:
                    t, ok, rtt = float(m[1]), 1, float(m[2])
                else:
                    m = re.match(r"\[(\d+\.\d+)\] no answer", line)
                    if not m:
                        continue
                    t, ok, rtt = float(m[1]), 0, None
                w.writerow([f"{t:.3f}", ok, f"{rtt:.2f}" if rtt is not None else ""]); f.flush()
                run_ok, run_bad = (run_ok + 1, 0) if ok else (0, run_bad + 1)
                new = self.state
                if run_ok >= self.need:
                    new = True
                elif run_bad >= self.need:
                    new = False
                if rtt is not None:
                    last_rtt = rtt
                now = time.time()
                if self.callback and (new != self.state or now - last_emit >= 5.0):
                    try:
                        self.callback({"type": "dataplane", "ts": round(t * 1000.0, 3), "up": bool(new) if new is not None else None,
                                       "rtt_ms": last_rtt, "changed": new != self.state, "host": self.host})
                    except Exception:  # noqa: BLE001
                        pass
                    last_emit = now
                self.state = new


def start_monitors(log_dir, stamp, ntg_host, ping_host, callback, ntg_hz=1.0, ntg=True, dataplane=True):
    mons = []
    if ntg:
        m = NtgStateMonitor(ntg_host, os.path.join(log_dir, f"ntgstate_{stamp}.csv"), callback, hz=ntg_hz); m.start(); mons.append(m)
    if dataplane:
        m = DataplaneMonitor(ping_host, os.path.join(log_dir, f"dataplane_{stamp}.csv"), callback); m.start(); mons.append(m)
    return mons
