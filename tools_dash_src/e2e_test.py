#!/usr/bin/env python3
"""Headless-Chrome end-to-end test of the real DASH page (real time, software decode) via CDP.

  python3 e2e_test.py --mpd tos --secs 120 [--port 8080] [--chrome-port 9222]

Opens http://127.0.0.1:<port>/dashtest_2sDASH_1080p-public-access.html?mpd=<mpd>, collects console
messages / exceptions, samples the <video> and the ABR state every 2 s, prints a timeline + a verdict.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request

import websockets

JS_STATE = r"""
(function(){
  try{
    var v=document.querySelector('video'); var q=v.getVideoPlaybackQuality?v.getVideoPlaybackQuality():{};
    var rep=(typeof adaptation!=='undefined')?adaptation.currentRepresentation:null;
    var buf=null; try{ buf=_mediaSourceBuffer.fillState.seconds }catch(e){}
    return JSON.stringify({dur:v.duration, t:+v.currentTime.toFixed(2), paused:v.paused, rs:v.readyState,
      vw:v.videoWidth, vh:v.videoHeight, dec:q.totalVideoFrames, drop:q.droppedVideoFrames,
      err:v.error?(v.error.code+':'+v.error.message):null,
      rep:(typeof adaptation!=='undefined')?adaptation.representationID:null,
      repBw:rep?parseInt(rep.bandwidth):null, repRes:rep?(rep.width+'x'+rep.height):null,
      estBps:(typeof myBandwidth!=='undefined')?myBandwidth.getBps():null,
      bufSec:buf, isacMode:(typeof isac!=='undefined')?isac.mode:null, recov:(typeof isac!=='undefined')?isac.recovering:null, bufTarget:(typeof overlayBuffer!=='undefined')?overlayBuffer.bufferSize.maxseconds:null, multires:(typeof DASH_MULTIRES!=='undefined')?DASH_MULTIRES:null,
      buffered:v.buffered.length?+v.buffered.end(v.buffered.length-1).toFixed(2):0});
  }catch(e){return JSON.stringify({jsErr:String(e)})}
})()
"""


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpd", default="tos")
    ap.add_argument("--secs", type=int, default=120)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--chrome-port", type=int, default=9222)
    ap.add_argument("--profile", default="/tmp/e2e_chrome_profile")
    ap.add_argument("--events", default="", help="e.g. 40:doppler_detected,75:normal (setISACMode at given seconds)")
    ap.add_argument("--throttle-mbps", type=float, default=0, help="emulate a download cap (CDP Network)")
    a = ap.parse_args()

    url = f"http://127.0.0.1:{a.port}/dashtest_2sDASH_1080p-public-access.html?mpd={a.mpd}"
    chrome = subprocess.Popen(
        ["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--mute-audio",
         "--autoplay-policy=no-user-gesture-required", f"--remote-debugging-port={a.chrome_port}",
         f"--user-data-dir={a.profile}", "--window-size=1920,1080", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws_url = None
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{a.chrome_port}/json"))
                pages = [t for t in tabs if t.get("type") == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                pass
            time.sleep(0.3)
        if not ws_url:
            print("no chrome page"); return 2
        msgs, excs = [], []
        async with websockets.connect(ws_url, max_size=None) as ws:
            nid = 0

            async def call(method, params=None):
                nonlocal nid
                nid += 1
                await ws.send(json.dumps({"id": nid, "method": method, "params": params or {}}))
                return nid

            await call("Runtime.enable"); await call("Log.enable"); await call("Page.enable")
            if a.throttle_mbps > 0:
                await call("Network.enable")
                await call("Network.emulateNetworkConditions", {"offline": False, "latency": 5, "downloadThroughput": a.throttle_mbps * 1e6 / 8, "uploadThroughput": -1})
            await call("Page.navigate", {"url": url})
            t0 = time.time()
            pending = {}
            states = []
            next_sample = t0 + 4
            events = sorted((float(x.split(':')[0]), x.split(':')[1]) for x in a.events.split(',') if x)
            while time.time() - t0 < a.secs:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    m = json.loads(raw)
                    meth = m.get("method")
                    if meth == "Runtime.consoleAPICalled":
                        txt = " ".join(str(x.get("value", x.get("description", ""))) for x in m["params"]["args"])
                        msgs.append((round(time.time() - t0, 1), m["params"]["type"], txt[:300]))
                    elif meth == "Runtime.exceptionThrown":
                        d = m["params"]["exceptionDetails"]
                        excs.append((round(time.time() - t0, 1), (d.get("exception", {}).get("description") or d.get("text"))[:300]))
                    elif meth == "Log.entryAdded":
                        e = m["params"]["entry"]
                        if e.get("level") in ("error", "warning"):
                            msgs.append((round(time.time() - t0, 1), "log-" + e["level"], (e.get("text", "") + " " + e.get("url", ""))[:300]))
                    elif "id" in m and m["id"] in pending:
                        r = m.get("result", {}).get("result", {}).get("value")
                        if r:
                            states.append((round(time.time() - t0, 1), json.loads(r)))
                except asyncio.TimeoutError:
                    pass
                while events and time.time() - t0 >= events[0][0]:
                    _, mode = events.pop(0)
                    await call('Runtime.evaluate', {'expression': f"setISACMode('{mode}','manual'); updateISACLabels();"})
                if time.time() >= next_sample:
                    next_sample += 2
                    i = await call("Runtime.evaluate", {"expression": JS_STATE, "returnByValue": True})
                    pending[i] = True

        print(f"=== timeline ({a.mpd}) ===")
        last = None
        for t, s in states:
            row = (s.get("repRes"), s.get("rep"), s.get("vw"), s.get("vh"))
            flag = "  <-- change" if row != last else ""
            last = row
            print(f"{t:6.1f}s dur={s.get('dur')} t={s.get('t')} paused={s.get('paused')} buf={s.get('bufSec')}/{s.get('bufTarget')} isac={s.get('isacMode')}{'(rec)' if s.get('recov') else ''} mseEnd={s.get('buffered')} "
                  f"video={s.get('vw')}x{s.get('vh')} dec={s.get('dec')} drop={s.get('drop')} rep#{s.get('rep')} "
                  f"{s.get('repRes')}@{s.get('repBw')} est={s.get('estBps')} err={s.get('err')} {s.get('jsErr','')}{flag}")
        print("=== exceptions ===")
        for e in excs[:20]:
            print(e)
        print("=== console errors/warnings (first 30) ===")
        n = 0
        for t, ty, txt in msgs:
            if ty in ("error", "log-error", "warning", "log-warning", "assert"):
                print(t, ty, txt); n += 1
                if n >= 30: break
        print("=== multires / switching log lines ===")
        for t, ty, txt in msgs:
            if "DASH_MULTIRES" in txt or "SWITCHING STREAM" in txt or "NYI" in txt:
                print(t, txt[:160])
        ok = bool(states) and states[-1][1].get("t", 0) > 5 and not excs and not states[-1][1].get("err")
        print("VERDICT:", "OK" if ok else "PROBLEM")
        return 0 if ok else 1
    finally:
        chrome.terminate()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
