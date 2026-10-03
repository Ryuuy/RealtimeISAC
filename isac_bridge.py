#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ISAC bridge: rt_main's UDP stream -> isac_server (WebSocket for the DASH page) + session file.

## Naming (20261003)

The ISAC mode that used to be called "shadowing" is now **"doppler_detected"** ("we see Doppler
now"); the other mode is still "normal". The DASH page also accepts the legacy string
"shadowing" so an old bridge keeps working. The power-based shadow judgements in the page
(dBFS / power) are a different thing and keep their name.

## What goes where

  rt_main (UDP, same host)        bridge                           consumers
  {"present":...}  verdict flips  -> push_result(mode)             page: adaptation (via mode)
                                  -> {"type":"doppler_event"}      page: detection record, server time
                                  -> SessionLog.add_detection      isac_logs/isac_*.npz
  {"spec": b64,...} 25 Hz         -> {"type":"doppler_spec"}       page: spectrogram chart + snapshot
                                  -> SessionLog.add_spec           isac_logs/isac_*.npz
  {"meta": {...}}  every 5 s      -> {"type":"doppler_meta"}       page: threshold line
  {"dBFS"...} {"power"...}        -> push_dBFS / push_power        page: shadowInputs.js
  (nothing)                       -> {"type":"clock","ts":ms} 1 Hz page: clock sync (see below)

No change to isac_server.py is needed: the new messages are broadcast through its existing
`_broadcast_payload`, and the page's new records (ISAC-interface button, ...) come back
through its existing `video_telemetry` log (stamped with server_recv_ts on arrival).

## One clock

Every timestamp written here is THIS host's time.time()*1000 (rt_main runs on the same host).
The page estimates its offset to this clock from the `ts` carried by every message (see
DASH_html_old-main/dash-js/isacClock.js) and also stamps its own records with a server-time
estimate; snapshots_FORMAT.md describes how to align everything.

## ⚠️ Run with miniconda's python3 (not /usr/bin/python3)

    python3 RealtimeISAC/isac_bridge.py

Two terminals:

    # A: bridge + web server
    python3 RealtimeISAC/isac_bridge.py
    # B: detection
    cd RealtimeISAC/realtime && /usr/bin/python3 rt_main.py --duration 3600 --isac [--live --live-fmax 500]

Web page: ws://<this host>:8765/isac, state: http://<this host>:8765/isac/state

## Robustness

1. Heartbeat: rt_main re-sends the current state every 2 s; push_result() ignores repeats.
2. Stale timeout: no verdict for stale_sec -> fall back to normal (better to miss than to show a
   stale "doppler detected").
3. The session file is written every --save-every seconds (in a thread) and on exit.
"""

import argparse
import asyncio
import base64
import json
import os
import signal
import socket
import sys
import time

DEFAULT_SERVER_DIR = "/home/usrpb210/DASH/DASH_html_old-main/isac_server"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODE_DETECTED = "doppler_detected"
MODE_NORMAL = "normal"


def main() -> int:
    p = argparse.ArgumentParser(description="rt_main verdict/spectrum -> isac_server + session file")
    p.add_argument("--udp-port", type=int, default=9099, help="port rt_main sends to")
    p.add_argument("--udp-host", type=str, default="127.0.0.1")
    p.add_argument("--isac-port", type=int, default=8765, help="isac_server port")
    p.add_argument("--server-dir", type=str, default=DEFAULT_SERVER_DIR,
                   help="directory containing isac_server.py")
    p.add_argument("--stale-sec", type=float, default=6.0,
                   help="fall back to normal after this long without a verdict (0 = never)")
    p.add_argument("--dry", action="store_true",
                   help="do not start isac_server, only print verdicts (and still record)")
    p.add_argument("--log-dir", type=str, default=os.path.join(HERE, "isac_logs"),
                   help="where the session .npz goes (default RealtimeISAC/isac_logs)")
    p.add_argument("--no-log", action="store_true", help="do not write a session file")
    p.add_argument("--save-every", type=float, default=20.0, help="seconds between autosaves")
    p.add_argument("--rotate-min", type=float, default=120.0,
                   help="start a new session part (file) every this many minutes; bounds RAM and "
                        "save time for long runs (0 = never rotate)")
    p.add_argument("--legacy-mode-name", action="store_true",
                   help="push the OLD mode string 'shadowing' instead of 'doppler_detected' to the page "
                        "(only needed while some page copy has not been updated yet; the new page "
                        "accepts both)")
    p.add_argument("--no-spec", action="store_true",
                   help="do not forward spectrogram columns to the page (still recorded)")
    a = p.parse_args()

    # Ctrl+C (SIGINT) and `kill` (SIGTERM) both end in the same clean path: final save of the session
    # file. Without this, a process started in the background (`&`, scripts, nohup) inherits an IGNORED
    # SIGINT, Python then never installs its KeyboardInterrupt handler, and `kill -INT` does nothing.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    srv = None
    push_result = push_dBFS = push_power = None
    get_telemetry = None
    if not a.dry:
        if not os.path.isdir(a.server_dir):
            print(f"isac_server directory not found: {a.server_dir}", file=sys.stderr)
            return 1
        sys.path.insert(0, a.server_dir)
        try:
            import isac_server as srv
            from isac_server import push_result, start_server_in_background
        except ImportError as e:
            print(f"import isac_server failed: {e}\nRun this with MINICONDA's python3 "
                  f"(fastapi/uvicorn), not /usr/bin/python3.", file=sys.stderr)
            return 1
        push_dBFS = getattr(srv, "push_dBFS", None)
        push_power = getattr(srv, "push_power", None)
        get_telemetry = getattr(srv, "get_video_telemetry", None)
        for name, fn in (("push_dBFS", push_dBFS), ("push_power", push_power),
                         ("get_video_telemetry", get_telemetry)):
            if fn is None:
                print(f"[bridge] WARNING: {a.server_dir}/isac_server.py has no {name} (old version) "
                      f"- the matching feature is skipped, everything else keeps working.",
                      file=sys.stderr)
        start_server_in_background(port=a.isac_port)

    def broadcast(payload: dict) -> bool:
        """Send an arbitrary JSON message to every connected page, through isac_server's own
        broadcast coroutine (no isac_server change needed)."""
        if srv is None or getattr(srv, "_loop", None) is None:
            return False
        fn = getattr(srv, "_broadcast_payload", None)
        if fn is None:
            return False
        asyncio.run_coroutine_threadsafe(fn(payload), srv._loop)
        return True

    log = None
    if not a.no_log:
        from isac_session_log import SessionLog
        log = SessionLog(a.log_dir, save_every=a.save_every, bridge_args=vars(a),
                         rotate_min=a.rotate_min)
        print(f"[bridge] session files: {os.path.dirname(log.path)}/isac_<stamp>_pNNN.npz "
              f"(autosave every {a.save_every:.0f}s, new part every {a.rotate_min:g} min)")

    def telemetry():
        try:
            return get_telemetry() if get_telemetry is not None else []
        except Exception:
            return []

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((a.udp_host, a.udp_port))
    sock.settimeout(0.5)
    print(f"[bridge] listening udp://{a.udp_host}:{a.udp_port}"
          f"{'  (dry mode, isac_server not started)' if a.dry else ''}")
    print("[bridge] waiting for rt_main ... Ctrl+C to quit")

    mode = MODE_NORMAL
    last_rx = 0.0
    n_rx = n_rx_dbfs = n_rx_power = n_spec = 0
    last_dbfs_print = last_power_print = 0.0
    last_det_state = None
    last_clock = 0.0

    def emit(new_mode: str, why: str) -> None:
        nonlocal mode
        if new_mode == mode:
            return
        mode = new_mode
        print(f"[{time.strftime('%H:%M:%S')}] -> {('doppler detected' if new_mode == MODE_DETECTED else 'normal'):16s} ({why})",
              flush=True)
        if push_result is not None:
            # pages that were not updated yet only know the old name for the "detected" mode
            push_result("shadowing" if (a.legacy_mode_name and new_mode == MODE_DETECTED) else new_mode)

    def tick() -> None:
        """Periodic work that must run even when rt_main is silent."""
        nonlocal last_clock
        now = time.time()
        if now - last_clock >= 1.0:
            last_clock = now
            broadcast({"type": "clock", "ts": round(now * 1000.0, 3)})
        if log is not None:
            log.maybe_save(telemetry)

    try:
        while True:
            tick()
            try:
                data, _ = sock.recvfrom(8192)
            except socket.timeout:
                if a.stale_sec and last_rx and time.time() - last_rx > a.stale_sec:
                    emit(MODE_NORMAL, f"no verdict for {a.stale_sec:.0f}s, falling back")
                    if log is not None and last_det_state == 1:
                        # synthetic flip, marked with score=-1 so analysis can tell it apart
                        log.add_detection(time.time() * 1000.0, 0, 0.0, -1)
                    last_det_state = 0
                    last_rx = 0.0
                continue
            try:
                msg = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

            # ---- spectrogram column (25 Hz) ----
            if "spec" in msg:
                n_spec += 1
                t_ms = float(msg.get("wallClock", time.time() * 1000.0))
                if log is not None:
                    try:
                        log.add_spec(t_ms, base64.b64decode(msg["spec"]), float(msg.get("ridge", 0.0)),
                                     int(msg.get("hit", 0)), int(msg.get("v", 0)), int(msg.get("E", 0)))
                    except Exception:
                        pass
                if not a.no_spec:
                    broadcast({"type": "doppler_spec", "ts": t_ms, "db": msg["spec"],
                               "ridge": msg.get("ridge"), "hit": msg.get("hit"),
                               "v": msg.get("v"), "E": msg.get("E")})
                continue

            if "meta" in msg:
                if log is not None:
                    log.set_rt_meta(msg["meta"])
                broadcast({"type": "doppler_meta", "meta": msg["meta"]})
                continue

            if "dBFS" in msg:
                n_rx_dbfs += 1
                if log is not None:
                    ch = msg.get("dBFS_channels") or [msg["dBFS"], msg["dBFS"]]
                    log.add_dbfs(float(msg.get("wallClock", time.time() * 1000.0)), float(msg["dBFS"]),
                                 float(ch[0]), float(ch[1]))
                if push_dBFS is not None:
                    push_dBFS(msg["dBFS"], msg.get("wallClock"))
                if time.time() - last_dbfs_print >= 1.0:
                    print(f"[bridge] dBFS={msg['dBFS']:+.1f} channels={msg.get('dBFS_channels')} "
                          f"(total {n_rx_dbfs})", flush=True)
                    last_dbfs_print = time.time()
                continue

            if "power" in msg:
                n_rx_power += 1
                if log is not None:
                    log.add_dlpower(float(msg.get("wallClock", time.time() * 1000.0)), float(msg["power"]))
                if push_power is not None:
                    push_power(msg["power"], msg.get("wallClock"))
                if time.time() - last_power_print >= 1.0:
                    print(f"[bridge] power(downlink)={msg['power']:+.1f}dBFS (total {n_rx_power})",
                          flush=True)
                    last_power_print = time.time()
                continue

            if "present" not in msg:
                continue
            last_rx = time.time()
            n_rx += 1
            state = 1 if msg["present"] else 0
            if state != last_det_state:
                t_ms = float(msg.get("wallClock", time.time() * 1000.0))
                if log is not None:
                    log.add_detection(t_ms, state, float(msg.get("ridge_hz", 0.0)),
                                      int(msg.get("score", 0)))
                broadcast({"type": "doppler_event",
                           "state": MODE_DETECTED if state else MODE_NORMAL, "ts": t_ms,
                           "ridge": msg.get("ridge_hz"), "E": msg.get("score")})
                last_det_state = state
            why = (f"doppler={msg.get('doppler_hz', 0):+.0f}Hz ridge={msg.get('ridge_hz', 0):.0f}Hz "
                   f"E={msg.get('score', '?')}")
            emit(MODE_DETECTED if state else MODE_NORMAL, why)
    except KeyboardInterrupt:
        print(f"\n[bridge] exit ({n_rx} verdict msgs, {n_spec} spectrum columns, "
              f"{n_rx_dbfs} dBFS, {n_rx_power} power)")
        if push_result is not None:
            push_result(MODE_NORMAL)
        if log is not None:
            log.ingest_telemetry(telemetry(), force=True)
            log.save(None, blocking=True)
            print(f"[bridge] session saved: {len(log.paths)} part(s): " + ", ".join(os.path.basename(x) for x in log.paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
