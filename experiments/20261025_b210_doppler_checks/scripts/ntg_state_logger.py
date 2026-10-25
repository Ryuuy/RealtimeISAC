#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NTG-2501 网页"メインモニタ"上方那六个状态，按 5 Hz 存成 csv（和 isac_logs/ntg_*.csv 同一个本机钟）。

  装置状態 / 無線状態(modem_sync) / 認証状態 / 監視信号状態 / イーサネットリンク状態(自局) / (対向局)

接口：POST http://<Master>/main_frm_xml.cgi（NTG 网页自己每 1 s 轮询的接口，不需要登录；和 xml_master_monitor.cgi 不是同一个）。
只读，5 Hz（设备 httpd 在 >50 Hz 才会拒连；Ubuntu 的 ntg2501_http_logger 另外还有 10 Hz）。
    /usr/bin/python3 ntg_state_logger.py --sec 60 --out states.csv
"""
import argparse
import csv
import re
import time
import urllib.request

TAGS = ["sys_status", "modem_sync_status", "auth_status", "signal_rx_status", "ether_if_0_now", "ether_if_1_now"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.1.101"); ap.add_argument("--sec", type=float, default=60)
    ap.add_argument("--hz", type=float, default=5.0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t_end = time.time() + a.sec; per = 1.0 / a.hz
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["unix_time", "ok", "query_ms"] + TAGS + [t + "_stat" for t in TAGS])
        while time.time() < t_end:
            t0 = time.time(); row = [f"{t0:.3f}", 0, ""] + [""] * 12
            try:
                req = urllib.request.Request(f"http://{a.host}/main_frm_xml.cgi", data=b"", method="POST")
                xml = urllib.request.urlopen(req, timeout=2).read().decode("utf-8", "replace")
                vals = [re.search(rf"<{t}>(.*?)</{t}>", xml, re.S) for t in TAGS]
                stats = [re.search(rf"<{t}_stat>(.*?)</{t}_stat>", xml, re.S) for t in TAGS]
                row = [f"{t0:.3f}", 1, f"{(time.time()-t0)*1000:.0f}"] + [(m.group(1) if m else "") for m in vals] + [(m.group(1) if m else "") for m in stats]
            except Exception as e:  # noqa: BLE001
                row[2] = f"{(time.time()-t0)*1000:.0f}"; row[3] = type(e).__name__
            w.writerow(row); f.flush()
            time.sleep(max(0, per - (time.time() - t0)))


if __name__ == "__main__":
    main()
