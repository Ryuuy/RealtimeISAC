#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""无头 Chrome 替人开 DASH 页面，把"页面每个请求什么时候发、什么时候回、怎么失败的"按本机 unix 时间存下来，
同时可以起 link_dataplane_probe.py（ping + 新连接 + 复用连接），用来对齐 NTG 的 master_valid。

    python3 headless_page_probe.py --secs 600 [--url http://192.168.1.5:8080/?mpd=tos] [--isac-interface off|on] [--no-probe]

用 miniconda 的 python3（要 websockets）。输出到 probe_runs/<时间戳>/：
  requests.csv  每个网络请求：start/firstbyte/end(unix)、state(done|failed)、status、errorText、bytes、url 尾部
  console.log   页面控制台（DASH JS Client fetching / retry in / ISAC / FILLPLAN ...）
  state.csv     每 0.5 s 一行：内容时间、是否暂停、缓冲(overlay/MSE)、码率档、带宽估计、ISAC 模式
  ping.log / newc.csv / keep.csv   (没加 --no-probe 时)
"""
import argparse
import asyncio
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request

import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "probe_runs")

JS_STATE = r"""
(function(){
  try{
    var v=document.querySelector('video'); var ct=v.currentTime, mse=0;
    for(var i=0;i<v.buffered.length;i++){ if(ct>=v.buffered.start(i)-0.05 && ct<=v.buffered.end(i)){ mse=v.buffered.end(i)-ct; } }
    var fill=null; try{ fill=overlayBuffer.fillState.seconds }catch(e){}
    var tgt=null; try{ tgt=overlayBuffer.bufferSize.maxseconds }catch(e){}
    return JSON.stringify({ct:+ct.toFixed(2), paused:v.paused, rs:v.readyState, mse:+mse.toFixed(2), fill:fill, tgt:tgt,
      rep:(typeof adaptation!=='undefined')?adaptation.representationID:null, est:(typeof myBandwidth!=='undefined')?myBandwidth.getBps():null,
      mode:(typeof isac!=='undefined')?isac.mode:null, rec:(typeof isac!=='undefined')?isac.recovering:null, vis:document.visibilityState});
  }catch(e){return JSON.stringify({jsErr:String(e)})}
})()
"""


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.1.5:8080/?mpd=tos")
    ap.add_argument("--secs", type=float, default=600)
    ap.add_argument("--chrome-port", type=int, default=9223)
    ap.add_argument("--isac-interface", choices=["on", "off"], default="off")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--final-eval", default=None, help="结束前在页面里执行这段 JS（returnByValue），结果存 final_eval.json（测试用）")
    ap.add_argument("--ntg-state-hz", type=float, default=1.0, help="NTG 六状态轮询频率（默认 1 Hz，和 NTG 网页自己一样；5 Hz 那轮疑似影响了设备）")
    ap.add_argument("--no-ntg-states", action="store_true", help="不记 NTG 网页上的六个状态（認証/監視信号/イーサネットリンク…）")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    out = a.out_dir or os.path.join(OUT_ROOT, time.strftime("%Y%m%d_%H%M%S") + "_page"); os.makedirs(out, exist_ok=True)

    probe = None
    if not a.no_probe:
        probe = subprocess.Popen([sys.executable if False else "/usr/bin/python3", os.path.join(HERE, "link_dataplane_probe.py"),
                                  "--sec", str(a.secs), "--out-dir", out], stdout=open(os.path.join(out, "probe.out"), "w"), stderr=subprocess.STDOUT)
    stl = None
    if not a.no_ntg_states:
        stl = subprocess.Popen(["/usr/bin/python3", os.path.join(HERE, "ntg_state_logger.py"), "--sec", str(a.secs), "--hz", str(a.ntg_state_hz), "--out", os.path.join(out, "ntg_states.csv")],
                               stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    prof = os.path.join(out, "chrome_profile")
    chrome = subprocess.Popen(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--mute-audio",
                               "--autoplay-policy=no-user-gesture-required", f"--remote-debugging-port={a.chrome_port}",
                               f"--user-data-dir={prof}", "--window-size=1280,720", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws_url = None
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{a.chrome_port}/json"))
                pages = [t for t in tabs if t.get("type") == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]; break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.3)
        if not ws_url:
            print("no chrome page"); return 2
        reqf = open(os.path.join(out, "requests.csv"), "w", newline=""); rq = csv.writer(reqf)
        rq.writerow(["start_unix", "firstbyte_unix", "end_unix", "state", "status", "error", "bytes", "type", "url_tail"])
        conf = open(os.path.join(out, "console.log"), "w")
        stf = open(os.path.join(out, "state.csv"), "w", newline=""); st = csv.writer(stf)
        st.writerow(["unix", "ct", "paused", "readyState", "mse_ahead", "overlay_fill", "target", "rep", "est_bps", "mode", "recovering", "vis"])
        reqs = {}
        nid = 0
        async with websockets.connect(ws_url, max_size=None) as ws:
            async def call(method, params=None):
                nonlocal nid
                nid += 1
                await ws.send(json.dumps({"id": nid, "method": method, "params": params or {}})); return nid
            for m in ("Runtime.enable", "Log.enable", "Page.enable", "Network.enable"):
                await call(m)
            await call("Page.navigate", {"url": a.url})
            t0 = time.time(); next_s = t0 + 3; set_if = t0 + 4; if_done = False; pend = {}
            print(f"page opened {a.url} at {time.strftime('%H:%M:%S')}  -> {out}", flush=True)
            while time.time() - t0 < a.secs:
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.2))
                except asyncio.TimeoutError:
                    m = None
                if m:
                    meth = m.get("method"); p = m.get("params", {})
                    if meth == "Network.requestWillBeSent":
                        reqs[p["requestId"]] = dict(start=p["wallTime"], ts0=p["timestamp"], url=p["request"]["url"], type=p.get("type", ""), fb=None)
                    elif meth == "Network.responseReceived" and p["requestId"] in reqs:
                        r = reqs[p["requestId"]]; r["fb"] = r["start"] + (p["timestamp"] - r["ts0"]); r["status"] = p["response"]["status"]
                    elif meth in ("Network.loadingFinished", "Network.loadingFailed") and p["requestId"] in reqs:
                        r = reqs.pop(p["requestId"]); end = r["start"] + (p["timestamp"] - r["ts0"])
                        failed = meth == "Network.loadingFailed"
                        rq.writerow([f"{r['start']:.3f}", f"{r['fb']:.3f}" if r["fb"] else "", f"{end:.3f}", "failed" if failed else "done", r.get("status", ""),
                                     (p.get("errorText", "") + (" canceled" if p.get("canceled") else "")) if failed else "", p.get("encodedDataLength", ""), r["type"], r["url"][-48:]]); reqf.flush()
                    elif meth == "Runtime.consoleAPICalled":
                        txt = " ".join(str(x.get("value", x.get("description", ""))) for x in p["args"]); conf.write(f"{time.time():.3f} {p['type']} {txt[:300]}\n"); conf.flush()
                    elif meth == "Runtime.exceptionThrown":
                        d = p["exceptionDetails"]; conf.write(f"{time.time():.3f} EXC {(d.get('exception', {}).get('description') or d.get('text'))[:300]}\n"); conf.flush()
                    elif "id" in m and m["id"] in pend:
                        v = m.get("result", {}).get("result", {}).get("value")
                        if v:
                            s = json.loads(v); tt = pend.pop(m["id"])
                            st.writerow([f"{tt:.3f}", s.get("ct"), s.get("paused"), s.get("rs"), s.get("mse"), s.get("fill"), s.get("tgt"), s.get("rep"), s.get("est"), s.get("mode"), s.get("rec"), s.get("vis")]); stf.flush()
                now = time.time()
                if (not if_done) and now >= set_if:
                    if_done = True
                    await call("Runtime.evaluate", {"expression": f"setISACInterface({'true' if a.isac_interface=='on' else 'false'},'headless'); 1"})
                if now >= next_s:
                    next_s += 0.5; i = await call("Runtime.evaluate", {"expression": JS_STATE, "returnByValue": True}); pend[i] = now
            if a.final_eval:
                i = await call("Runtime.evaluate", {"expression": a.final_eval, "returnByValue": True})
                tend = time.time() + 5
                while time.time() < tend:
                    m = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                    if m.get("id") == i:
                        json.dump(m.get("result", {}), open(os.path.join(out, "final_eval.json"), "w"), ensure_ascii=False, indent=1); break
        print("done", out, flush=True)
    finally:
        chrome.terminate()
        if stl:
            stl.terminate()
        if probe:
            try:
                probe.wait(timeout=20)
            except Exception:  # noqa: BLE001
                probe.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
