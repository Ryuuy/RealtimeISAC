#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ubuntu-side session recorder used by isac_bridge.py.

Everything is stamped with THIS machine's wall clock (time.time()*1000, ms since the Unix
epoch). rt_main.py runs on the same machine, so spectrogram columns, detection events and the
server-side receive time of the browser's ISAC-interface button presses share ONE clock.
That clock is the "server time" referred to in DASH_html_old-main/snapshots_FORMAT.md.

One session = a series of .npz PARTS, `isac_<stamp>_p001.npz`, `_p002.npz`, ... A new part starts
every `rotate_min` minutes (default 120, so a normal run is ONE file). Reason: re-compressing the whole history every 20 s is O(N^2)
(measured on real spectra: 8 h = 133 MB file, 6.4 s per save, ~400 MB RAM peak); with parts the cost
and memory per part stay bounded (30 min = 8 MB, 0.4 s; 2 h = 33 MB, 1.6 s). Each part is self-contained and covers
[part start, next part start); load several with session_tools.load_isac_log(*paths) (it concatenates).
A part is written every `save_every` seconds while it is the current one, and on exit (atomically).
Detection events: the first event of part k>1 is a carried-over copy of the last state of part k-1,
marked score = -2 (so each part starts with a known state); score = -1 marks a bridge-forced fallback.
`wf_verdict` is always complete, use it if in doubt.

  format_version            int
  meta_json                 JSON string, see below
  --- waterfall (spectrogram) -- one row per column, ~25 Hz ---------------------------------
  wf_t_ms          float64 [T]        server time of the column (end of its 200 ms window)
  wf_db_q          uint8   [T, 256]   quantised dB, dB = db_min + q * (db_max - db_min) / 255
  wf_freqs_hz      float32 [256]      Doppler frequency of each column of wf_db_q (fftshifted)
  wf_db_min, wf_db_max   float64      quantisation range
  wf_ridge_hz      float32 [T]        DC ridge width of that column
  wf_raw_hit       uint8   [T]        1 = ridge width >= threshold (per step, no debounce)
  wf_verdict       uint8   [T]        1 = verdict "doppler detected" at that column
  wf_score         int32   [T]        verdict evidence E (accumulate) or hit count (M-of-N)
  --- Doppler detection events (verdict flips only) -----------------------------------------
  det_t_ms         float64 [N]        server time the verdict flipped (rt_main's clock)
  det_state        uint8   [N]        1 = doppler detected, 0 = back to normal
  det_ridge_hz     float32 [N]
  det_score        int32   [N]
  --- ISAC interface (web page button) ON/OFF events ----------------------------------------
  iface_server_recv_ms   float64 [M]  server time the browser's message ARRIVED (authoritative)
  iface_client_ms        float64 [M]  browser Date.now() when the button was pressed
  iface_client_est_server_ms float64 [M]  the page's own conversion of that to server time
  iface_enabled          uint8   [M]  1 = ISAC interface ON, 0 = OFF
  --- reference / downlink power (rt_main -> bridge, ~50 Hz, server time) -----------------
  dbfs_t_ms        float64 [K]        instantaneous dBFS of the two RX channels of one 20 ms step
  dbfs_mean        float32 [K]        mean of the two channels
  dbfs_ch0         float32 [K]        ch0 = reference channel (dBFS)
  dbfs_ch1         float32 [K]        ch1 = sensing channel   (dBFS)
  dlpower_t_ms     float64 [L]        downlink (Master direction) power, sensing channel
  dlpower_dbfs     float32 [L]        (only while TDD is locked)
  --- operator markers (v3) ---------------------------------------------------------------
  marker_t_ms      float64 [P]        server time the operator pressed Enter in the bridge terminal
  marker_label     str     [P]        text typed before Enter ("mark" if empty), e.g. "cross_mid_fast"
  --- raw page telemetry (segments, playback samples, ...) ---------------------------------
  video_telemetry_json   JSON string: list of the records the page pushed over the WebSocket,
                         each with server_recv_ts (server clock, SECONDS). v2: collected
                         incrementally by the bridge every second (so nothing is lost when
                         isac_server's 50000-record memory wraps), WITHOUT kind=="shadowJudgment"
                         (dBFS/power judgements with placeholder thresholds, ~130 records/s that
                         flooded that memory; the raw dBFS/power values are in dbfs_* / dlpower_*).
"""

import json
import os
import socket
import threading
import time

import numpy as np

FORMAT_VERSION = 3   # v3: + marker_t_ms/marker_label (终端回车打点)。# v2: + dbfs_* / dlpower_* 序列；网页遥测改为增量收集（v1 只含 isac_server 内存里最近 50000 条）
NFFT = 256
FS_OUT = 1000.0


class SessionLog:
    def __init__(self, log_dir: str, save_every: float = 20.0, bridge_args: dict | None = None,
                 rotate_min: float = 120.0):
        os.makedirs(log_dir, exist_ok=True)
        self.t0_ms = time.time() * 1000.0
        self._stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.t0_ms / 1000.0))
        self._log_dir = log_dir
        self.rotate_ms = rotate_min * 60000.0 if rotate_min and rotate_min > 0 else float("inf")
        self.part = 1
        self.part_start_ms = self.t0_ms
        self.path = self._part_path(1)
        self.paths = [self.path]
        self.save_every = save_every
        self._last_save = time.monotonic()
        self._bridge_args = bridge_args or {}
        self._rt_meta: dict = {}
        self._wf_t: list = []
        self._wf_db: list = []          # list of bytes(256)
        self._wf_ridge: list = []
        self._wf_hit: list = []
        self._wf_v: list = []
        self._wf_E: list = []
        self._det: list = []            # (t_ms, state, ridge, score)
        self._dbfs: list = []           # (t_ms, mean, ch0, ch1)
        self._dl: list = []             # (t_ms, dBFS)
        self._marks: list = []          # (t_ms, label)
        self._vt: list = []             # page telemetry records of THIS part (incremental)
        self._vt_last = None            # last record object ingested (identity)
        self._vt_t = 0.0                # monotonic time of the last ingest
        self.vt_gaps = 0                # times the isac_server memory wrapped before we polled
        self._lock = threading.Lock()
        self._saver = None
        self.n_saves = 0

    def _part_path(self, k: int) -> str:
        return os.path.join(self._log_dir, f"isac_{self._stamp}_p{k:03d}.npz")

    def _rotate(self, now_ms: float, video_telemetry: list | None) -> None:
        """Finish the current part (written in a thread) and start the next one with empty buffers."""
        last_state = self._det[-1] if self._det else None
        self.save(video_telemetry, blocking=False, end_ms=now_ms, wait_prev=True)
        with self._lock:
            self._wf_t, self._wf_db, self._wf_ridge = [], [], []
            self._wf_hit, self._wf_v, self._wf_E = [], [], []
            self._dbfs, self._dl, self._vt, self._marks = [], [], [], []
            self._det = [(now_ms, last_state[1], last_state[2], -2)] if last_state else []
        self.part += 1
        self.part_start_ms = now_ms
        self.path = self._part_path(self.part)
        self.paths.append(self.path)

    # ---- ingest ----
    def set_rt_meta(self, meta: dict) -> None:
        self._rt_meta = meta

    def should_rotate(self, now_ms: float) -> bool:
        return now_ms - self.part_start_ms >= self.rotate_ms

    def add_spec(self, t_ms: float, db_q: bytes, ridge: float, hit: int, v: int, score: int) -> None:
        if len(db_q) != NFFT:
            return
        with self._lock:
            self._wf_t.append(t_ms)
            self._wf_db.append(db_q)
            self._wf_ridge.append(ridge)
            self._wf_hit.append(hit)
            self._wf_v.append(v)
            self._wf_E.append(score)

    def add_dbfs(self, t_ms: float, mean: float, ch0: float, ch1: float) -> None:
        with self._lock:
            self._dbfs.append((t_ms, mean, ch0, ch1))

    def add_marker(self, t_ms: float, label: str) -> None:
        with self._lock:
            self._marks.append((t_ms, label))

    def add_dlpower(self, t_ms: float, dbfs: float) -> None:
        with self._lock:
            self._dl.append((t_ms, dbfs))

    def ingest_telemetry(self, recs, force: bool = False) -> None:
        """Append the page-telemetry records that appeared since the last call. `recs` is
        isac_server.get_video_telemetry() (append-only, trimmed at the head when it exceeds 50000).
        Called about once a second; the 50000-record window holds ~6 min even at 130 rec/s, so
        polling every second never misses anything. shadowJudgment records are dropped (see top)."""
        now = time.monotonic()
        if not force and now - self._vt_t < 1.0:
            return
        self._vt_t = now
        if not recs:
            return
        i = len(recs) - 1
        if self._vt_last is not None:
            while i >= 0 and recs[i] is not self._vt_last:
                i -= 1
            if i < 0:
                self.vt_gaps += 1      # our last record was already evicted: we were away > the window
        new = recs[i + 1:] if (self._vt_last is not None and i >= 0) else recs
        self._vt_last = recs[-1]
        keep = [m for m in new if m.get("kind") != "shadowJudgment"]
        if keep:
            with self._lock:
                self._vt.extend(keep)

    def add_detection(self, t_ms: float, state: int, ridge: float, score: int) -> None:
        with self._lock:
            self._det.append((t_ms, state, ridge, score))

    # ---- persistence ----
    def maybe_save(self, video_telemetry_fn) -> None:
        """video_telemetry_fn: callable returning isac_server's page-telemetry list."""
        now_ms = time.time() * 1000.0
        if time.monotonic() - self._vt_t >= 1.0:
            self.ingest_telemetry(video_telemetry_fn())
        if self.should_rotate(now_ms):
            self._rotate(now_ms, None)
        elif time.monotonic() - self._last_save >= self.save_every:
            self.save(None)

    def save(self, video_telemetry: list | None, blocking: bool = False,
             end_ms: float | None = None, wait_prev: bool = False) -> None:
        """Snapshot the lists (fast, under the lock) and compress/write in a thread so the
        bridge loop never stalls the UDP receive. `end_ms` is the end of this part (rotation)."""
        self._last_save = time.monotonic()
        if self._saver is not None and self._saver.is_alive():
            if not (blocking or wait_prev):
                return                  # previous write still running; next tick will catch up
            self._saver.join()
        with self._lock:
            t = np.array(self._wf_t, dtype=np.float64)
            db = (np.frombuffer(b"".join(self._wf_db), dtype=np.uint8).reshape(-1, NFFT).copy()
                  if self._wf_db else np.zeros((0, NFFT), np.uint8))
            ridge = np.array(self._wf_ridge, dtype=np.float32)
            hit = np.array(self._wf_hit, dtype=np.uint8)
            v = np.array(self._wf_v, dtype=np.uint8)
            E = np.array(self._wf_E, dtype=np.int32)
            det = list(self._det)
            dbfs = list(self._dbfs)
            dl = list(self._dl)
            marks = list(self._marks)
            vt = list(self._vt)           # incremental (v2); already limited to this part
        ifc = [m for m in vt if m.get("kind") == "isacInterface"]
        path = self.path
        part = self.part

        def col(rows, i, dt):
            return np.array([r[i] for r in rows], dtype=dt)

        arrays = dict(
            format_version=np.array(FORMAT_VERSION),
            wf_t_ms=t, wf_db_q=db,
            wf_freqs_hz=np.fft.fftshift(np.fft.fftfreq(NFFT, d=1.0 / FS_OUT)).astype(np.float32),
            wf_db_min=np.array(float(self._rt_meta.get("db_min", -90.0))),
            wf_db_max=np.array(float(self._rt_meta.get("db_max", 50.0))),
            wf_ridge_hz=ridge, wf_raw_hit=hit, wf_verdict=v, wf_score=E,
            det_t_ms=col(det, 0, np.float64), det_state=col(det, 1, np.uint8),
            det_ridge_hz=col(det, 2, np.float32), det_score=col(det, 3, np.int32),
            dbfs_t_ms=np.array([r[0] for r in dbfs], dtype=np.float64),
            dbfs_mean=np.array([r[1] for r in dbfs], dtype=np.float32),
            dbfs_ch0=np.array([r[2] for r in dbfs], dtype=np.float32),
            dbfs_ch1=np.array([r[3] for r in dbfs], dtype=np.float32),
            dlpower_t_ms=np.array([r[0] for r in dl], dtype=np.float64),
            dlpower_dbfs=np.array([r[1] for r in dl], dtype=np.float32),
            iface_server_recv_ms=np.array([m.get("server_recv_ts", np.nan) * 1000.0 for m in ifc],
                                          dtype=np.float64),
            iface_client_ms=np.array([m.get("wallClock", np.nan) for m in ifc], dtype=np.float64),
            iface_client_est_server_ms=np.array([m.get("serverTsEst", np.nan) for m in ifc],
                                                dtype=np.float64),
            iface_enabled=np.array([1 if m.get("enabled") else 0 for m in ifc], dtype=np.uint8),
            marker_t_ms=np.array([m[0] for m in marks], dtype=np.float64),
            marker_label=np.array([m[1] for m in marks], dtype=str),
            video_telemetry_json=np.array(json.dumps(vt)),
            meta_json=np.array(json.dumps({
                "format_version": FORMAT_VERSION,
                "doc": "DASH_html_old-main/snapshots_FORMAT.md",
                "start_ms": self.t0_ms,
                "start_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.t0_ms / 1000.0)),
                "part": part, "part_start_ms": self.part_start_ms,
                "part_end_ms": end_ms,           # null = still the current (or last) part
                "session_stamp": self._stamp,
                "saved_ms": time.time() * 1000.0,
                "host": socket.gethostname(),
                "clock": "time.time()*1000 on the host running rt_main.py / isac_bridge.py "
                         "('server time'); NTP-sync this host if you compare with other machines",
                "column_window_sec": float(self._rt_meta.get("window_sec", 0.2)),
                "column_stamp": "end of the 0.2 s analysis window (window centre = stamp - 0.1 s); "
                                "plus ~20-60 ms capture/USB latency not corrected",
                "rt_meta": self._rt_meta,
                "bridge_args": self._bridge_args,
                "n_columns": int(len(t)), "n_detection_events": len(det),
                "n_dbfs": len(dbfs), "n_dlpower": len(dl), "n_telemetry": len(vt),
                "telemetry_gaps": self.vt_gaps,
                "n_interface_events": len(ifc), "n_markers": len(marks),
            }, ensure_ascii=False)),
        )

        def _write():
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **arrays)
            os.replace(tmp, path)
            self.n_saves += 1

        if blocking:
            _write()
        else:
            self._saver = threading.Thread(target=_write, daemon=True)
            self._saver.start()
