#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-time scrolling Doppler spectrogram (matplotlib window, newest column on the right).

Everything drawn on the figure is ASCII/English on purpose: the figure is meant for
publication, and CJK glyphs are not available in the default font (they render as boxes,
and every missing-glyph lookup also made each redraw several times slower).

Layout (top to bottom, shared time axis):
    [decision strip]  Verdict (debounced) / Raw (per-step ridge hit, no debounce)
    [spectrogram   ]  |S(f)|^2 of conj(ch0)*ch1; dashed red line = verdict turns PRESENT,
                      dashed black line = verdict returns to ABSENT
    [DC ridge width]  per-step ridge width, threshold line, red shading = verdict PRESENT

## Why drawing runs in its own process (see LiveWaterfallProc at the bottom)

One redraw costs 10-70 ms on this machine, the DSP step budget is 20 ms. In-process drawing
caused 198 dropped steps out of 4509; a dropped step breaks the phase continuity of the
200-frame ring (single-frame broadband burst, false ridge hits). The GUI therefore lives in
a spawned child process and the DSP side only does a non-blocking put_nowait per step.

## Display latency

The parent stamps every column with its wall clock when it is queued; the child records
(time drawn - time queued) at each redraw and `status()` reports p50/p99. Detection latency
is a separate thing (200 ms analysis window + M-of-N debounce), see rt_config.py.
"""

import os
import time

import numpy as np

from rt_config import RtConfig
from rt_dsp import to_db


def _style(matplotlib):
    rc = matplotlib.rcParams
    rc["font.family"] = "sans-serif"
    rc["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    rc["font.size"] = 10
    rc["axes.titlesize"] = 11
    rc["axes.labelsize"] = 10
    rc["legend.fontsize"] = 8


class LiveWaterfall:
    def __init__(self, freqs: np.ndarray, cfg: RtConfig, span_sec: float = 15.0,
                 fps: float = 12.0, fmax: float | None = None):
        self.cfg = cfg
        self.enabled = False
        self._draw_ms = []
        self._lag_ms = []          # display lag per redraw (ms); full history, ~1 float/redraw
        self._lag_t = []
        self._last_sent = None
        self._have_ridge = False
        self._peak, self._ridge, self._present = 0.0, None, False

        rows = (np.flatnonzero(np.abs(freqs) <= fmax) if fmax is not None
                else np.arange(len(freqs)))
        if rows.size < 2:
            rows = np.arange(len(freqs))
        self._rows = rows
        f_lo, f_hi = float(freqs[rows[0]]), float(freqs[rows[-1]])

        self._ncols = n = max(8, int(round(span_sec / cfg.step_sec)))
        # NaN = no data yet -> background colour, so the picture fills from right to left
        self._buf = np.full((rows.size, n), np.nan, dtype=np.float32)
        self._disp = np.empty_like(self._buf)
        self._rbuf = np.full(n, np.nan, dtype=np.float32)
        self._rdisp = np.empty_like(self._rbuf)
        self._pbuf = np.zeros(n, dtype=np.float32)      # debounced verdict per column
        self._hbuf = np.zeros(n, dtype=np.float32)      # raw (undebounced) ridge hit
        self._p = 0
        self._n_written = 0

        self._every = max(1, int(round(1.0 / (fps * cfg.step_sec))))
        self._k = 0
        self._clim_every = max(1, int(round(3.0 / cfg.step_sec)))
        self._clim = None

        import matplotlib
        if not os.environ.get("DISPLAY"):
            print("WARNING: no DISPLAY, live figure disabled (use --waterfall for the ASCII view)")
            return
        _style(matplotlib)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        plt.ion()
        self._plt = plt
        self.fig = plt.figure(figsize=(10, 6.6), dpi=100)
        self.fig.canvas.manager.set_window_title("RealtimeISAC - Live Doppler")
        gs = self.fig.add_gridspec(3, 2, width_ratios=[60, 1], height_ratios=[0.45, 3, 1],
                                   hspace=0.10, wspace=0.025, left=0.085, right=0.93,
                                   top=0.93, bottom=0.09)
        self.ax_bar = self.fig.add_subplot(gs[0, 0])
        self.ax = self.fig.add_subplot(gs[1, 0], sharex=self.ax_bar)
        self.ax2 = self.fig.add_subplot(gs[2, 0], sharex=self.ax_bar)
        cax = self.fig.add_subplot(gs[1, 1])
        extent = [-span_sec, 0.0, f_lo, f_hi]

        # ---- spectrogram ----
        cmap = matplotlib.colormaps["jet"].copy()
        cmap.set_bad(color="#101018")
        self._im = self.ax.imshow(self._disp, aspect="auto", origin="lower", cmap=cmap,
                                  interpolation="nearest", extent=extent, vmin=-80, vmax=-40)
        self.ax.set_facecolor("#101018")
        self.ax.set_ylabel("Doppler frequency (Hz)")
        self.ax.tick_params(labelbottom=False)
        g = cfg.dc_guard_hz
        for s in (-g, g):
            self.ax.axhline(s, color="w", ls="--", lw=0.6, alpha=0.4)
        cb = self.fig.colorbar(self._im, cax=cax)
        cb.set_label("Relative power (dB)")
        # Verdict transitions as dashed vertical lines inside the spectrogram:
        # red = turns PRESENT, black = back to ABSENT. White halo so both stay visible on
        # jet's dark-blue and dark-red extremes. Segments are rebuilt at every redraw.
        import matplotlib.patheffects as pe
        from matplotlib.collections import LineCollection
        halo = [pe.withStroke(linewidth=3.0, foreground="white")]
        self._span, self._f_lo, self._f_hi = span_sec, f_lo, f_hi
        self._on_lines = LineCollection([], colors="#e00000", linestyles=(0, (5, 3)),
                                        linewidths=1.9, zorder=4)
        self._off_lines = LineCollection([], colors="k", linestyles=(0, (5, 3)),
                                         linewidths=1.9, zorder=4)
        for lc in (self._on_lines, self._off_lines):
            lc.set_path_effects(halo)
            self.ax.add_collection(lc)
        self.ax.set_xlim(-span_sec, 0.0)
        self.ax.set_ylim(f_lo, f_hi)
        from matplotlib.lines import Line2D
        self._leg = self.ax.legend(
            [Line2D([], [], color="#e00000", ls=(0, (5, 3)), lw=1.9,
                    path_effects=halo),
             Line2D([], [], color="k", ls=(0, (5, 3)), lw=1.9, path_effects=halo)],
            ["Presence detected", "Presence cleared"], loc="lower right",
            frameon=True, framealpha=0.85, fontsize=8)
        self._leg.set_zorder(6)
        self._txt = self.ax.text(
            0.008, 0.97, "", transform=self.ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace", color="w",
            bbox=dict(fc="#000000B0", ec="none", pad=3), zorder=5)

        # ---- decision strip (top) ----
        self._strip = np.zeros((2, n, 4), dtype=np.float32)
        self._strip[0, :, :3] = (0.12, 0.47, 0.71)      # Raw: blue
        self._strip[1, :, :3] = (0.84, 0.15, 0.16)      # Verdict: red
        self._strip_im = self.ax_bar.imshow(
            self._strip, aspect="auto", origin="lower", interpolation="nearest",
            extent=[-span_sec, 0.0, 0, 2])
        self.ax_bar.set_ylim(0, 2)
        self.ax_bar.set_yticks([0.5, 1.5])
        self.ax_bar.set_yticklabels(["Raw", "Verdict"], fontsize=8)
        self.ax_bar.tick_params(labelbottom=False, length=0)
        self.ax_bar.set_facecolor("#f2f2f2")
        for sp in self.ax_bar.spines.values():
            sp.set_linewidth(0.5)
        self.ax_bar.set_title("Real-time Doppler spectrogram (cross-channel)", pad=6)

        # ---- DC ridge width ----
        self._xs = np.linspace(-span_sec, 0.0, n)
        self._shade = np.zeros((1, n, 4), dtype=np.float32)
        self._shade[..., :3] = (0.84, 0.15, 0.16)
        self._shade_im = self.ax2.imshow(self._shade, aspect="auto", origin="lower",
                                         interpolation="nearest",
                                         extent=[-span_sec, 0.0, 0, 200], zorder=0)
        (self._rline,) = self.ax2.plot(self._xs, self._rdisp, color="#4c3a8f", lw=1.0)
        thr = self.ax2.axhline(cfg.ridge_thresh_hz, color="k", ls=":", lw=0.9)
        self.ax2.set_ylim(0, 200)
        self.ax2.set_xlim(-span_sec, 0.0)
        self.ax2.set_ylabel("DC ridge width (Hz)")
        self.ax2.set_xlabel("Time (s, 0 = now)")
        self.ax2.grid(alpha=0.3, lw=0.5)
        self._leg2 = self.ax2.legend([self._rline, thr, Patch(fc="#d62728", alpha=0.3)],
                        ["Ridge width", f"Threshold ({cfg.ridge_thresh_hz:g} Hz)", "Present"],
                        loc="upper left", ncol=3, frameon=True, framealpha=0.85)

        self.fig.canvas.mpl_connect("close_event", self._on_close)
        # Every set_data() marks the figure stale and matplotlib queues a draw_idle(); the
        # flush_events() below then ran it = a full Figure.draw (~95 ms) right after each
        # blit, which made blitting pointless. Switch the automatic idle redraw off and do
        # full redraws ourselves only when needed (colour-limit change, window resize).
        self._need_full = False
        # Blit needs every artist that changes per frame to be `animated`: those are skipped
        # by the full canvas.draw() and therefore NOT baked into the saved backgrounds.
        # Without this, each full redraw froze a stale copy of the strip bars / ridge curve /
        # shading into the background, and the live frames were drawn on top of that ghost
        # (looked like one of the top/bottom panels was stuck).
        self._animated = [self._im, self._on_lines, self._off_lines, self._txt, self._leg,
                          self._strip_im, self._shade_im, self._rline, self._leg2]
        for art in self._animated:
            art.set_animated(True)
        self.fig.canvas.draw_idle = lambda *a, **k: None
        self.fig.canvas.mpl_connect("resize_event",
                                    lambda _e: setattr(self, "_need_full", True))
        self._full_redraw()
        self.fig.canvas.flush_events()
        self.enabled = True

    def _on_close(self, _evt) -> None:
        self.enabled = False

    def _full_redraw(self) -> None:
        """Full canvas draw + re-capture of the clean blit backgrounds.

        matplotlib's Axes.draw skips animated artists EXCEPT AxesImage (it always draws
        those), so the three per-frame images would be baked into the background as stale
        ghosts. Hide them for the duration of the draw."""
        imgs = (self._im, self._strip_im, self._shade_im)
        for a in imgs:
            a.set_visible(False)
        try:
            from matplotlib.transforms import Bbox
            cv = self.fig.canvas
            cv.draw()
            # Pad the restore/blit regions by 4 px: line caps and the antialiased edges of
            # the images spill just outside axes.bbox, and unrestored pixels there pile up as
            # a smear along the borders (e.g. the right end of the ridge curve moves each frame).
            self._bbs = []
            for a in (self.ax_bar, self.ax, self.ax2):
                b = a.bbox
                self._bbs.append(Bbox.from_extents(b.x0 - 4, b.y0 - 4, b.x1 + 4, b.y1 + 4))
            self._bgs = [cv.copy_from_bbox(bb) for bb in self._bbs]
        finally:
            for a in imgs:
                a.set_visible(True)

    def update(self, power_lin: np.ndarray, peak_hz: float, status: str, present: bool,
               ridge_hz: float | None = None, raw_hit: bool | None = None,
               t_sent: float | None = None, draw: bool = True) -> None:
        """Called once per DSP step. Writing a column is cheap; the redraw happens every
        `_every` steps. `status` is accepted for interface compatibility and is not drawn
        (the figure text is built here, in English)."""
        if not self.enabled:
            return
        p = self._p
        self._buf[:, p] = to_db(power_lin[self._rows])
        self._pbuf[p] = 1.0 if present else 0.0
        self._hbuf[p] = 1.0 if raw_hit else 0.0
        if ridge_hz is not None:
            self._rbuf[p] = ridge_hz
            self._have_ridge = True
        self._p = (p + 1) % self._ncols
        self._n_written += 1
        self._last_sent = t_sent
        self._peak, self._ridge, self._present = peak_hz, ridge_hz, present
        if not draw:
            return

        self._k += 1
        if self._k % self._every:
            return
        self._redraw()

    def _redraw(self) -> None:
        t0 = time.perf_counter()
        p, n = self._p, self._ncols

        def unroll(src, dst):           # oldest column is at p, newest at p-1
            np.copyto(dst[..., :n - p], src[..., p:])
            np.copyto(dst[..., n - p:], src[..., :p])

        unroll(self._buf, self._disp)
        self._im.set_data(self._disp)
        pd = np.empty(n, dtype=np.float32)
        hd = np.empty(n, dtype=np.float32)
        unroll(self._pbuf, pd)
        unroll(self._hbuf, hd)
        d = np.diff(pd, prepend=pd[0])
        w = self._span / n
        x_on = -self._span + np.flatnonzero(d > 0) * w     # left edge of first PRESENT column
        x_off = -self._span + np.flatnonzero(d < 0) * w    # left edge of first ABSENT column
        self._on_lines.set_segments([[(x, self._f_lo), (x, self._f_hi)] for x in x_on])
        self._off_lines.set_segments([[(x, self._f_lo), (x, self._f_hi)] for x in x_off])
        self._shade[0, :, 3] = 0.22 * pd
        self._shade_im.set_data(self._shade)
        self._strip[0, :, 3] = hd
        self._strip[1, :, 3] = pd
        self._strip_im.set_data(self._strip)
        if self._have_ridge:
            unroll(self._rbuf, self._rdisp)
            self._rline.set_ydata(self._rdisp)

        full = False
        if self._n_written >= 8 and self._k % self._clim_every < self._every:
            v = self._disp[np.isfinite(self._disp)]
            if v.size > 32:
                lo, hi = np.percentile(v, (5, 99))
                if self._clim is None or abs(lo - self._clim[0]) > 1.5 \
                        or abs(hi - self._clim[1]) > 1.5:
                    self._clim = (float(lo), float(hi + 2.0))
                    self._im.set_clim(*self._clim)
                    full = True

        if self._need_full:
            full, self._need_full = True, False
        txt = f"{'PRESENT' if self._present else 'ABSENT '}   peak {self._peak:+6.1f} Hz"
        if self._ridge is not None:
            txt += f"   ridge {self._ridge:5.1f} Hz"
        self._txt.set_text(txt)
        self._txt.set_color("#ff6b6b" if self._present else "w")

        cv = self.fig.canvas
        try:
            if full:
                self._full_redraw()      # clean backgrounds, no per-frame artists baked in
            cv.restore_region(self._bgs[0])
            self.ax_bar.draw_artist(self._strip_im)
            cv.blit(self._bbs[0])
            cv.restore_region(self._bgs[1])
            self.ax.draw_artist(self._im)
            self.ax.draw_artist(self._on_lines)
            self.ax.draw_artist(self._off_lines)
            self.ax.draw_artist(self._leg)
            self.ax.draw_artist(self._txt)
            cv.blit(self._bbs[1])
            cv.restore_region(self._bgs[2])
            self.ax2.draw_artist(self._shade_im)
            if self._have_ridge:
                self.ax2.draw_artist(self._rline)
            self.ax2.draw_artist(self._leg2)
            cv.blit(self._bbs[2])
            cv.flush_events()
        except Exception:
            self.enabled = False       # window was closed
            return

        self._draw_ms.append((time.perf_counter() - t0) * 1e3)
        if self._last_sent is not None:
            self._lag_ms.append((time.time() - self._last_sent) * 1e3)
            self._lag_t.append(time.time())
        for lst in (self._draw_ms,):
            if len(lst) > 512:
                lst.pop(0)

    def status(self) -> str:
        if not self._draw_ms:
            return "live figure: not enabled" if not self.enabled else "live figure: no redraws"
        d = np.array(self._draw_ms)
        s = (f"live figure: {len(self._draw_ms)} redraws  draw p50={np.percentile(d, 50):.1f}ms "
             f"p99={np.percentile(d, 99):.1f}ms  every {self._every} steps")
        if self._lag_ms:
            lag = np.array(self._lag_ms)
            s += (f"  | display lag (queued->drawn) p50={np.percentile(lag, 50):.0f}ms "
                  f"p99={np.percentile(lag, 99):.0f}ms")
            # Does the lag grow with time? compare the first and last 5 s (skip 2 s startup)
            t = np.array(self._lag_t)
            a = lag[(t >= t[0] + 2) & (t < t[0] + 7)]
            b = lag[t >= t[-1] - 5]
            if a.size and b.size:
                s += (f"\n    lag over time: first 5s (after 2s startup) median {np.median(a):.0f}ms"
                      f" -> last 5s median {np.median(b):.0f}ms")
        return s

    def close(self) -> None:
        if getattr(self, "_plt", None) is None:
            return
        self._plt.ioff()
        self.enabled = False


# ======================================================================
# Separate-process wrapper (spawn, not fork: it must start before UHD opens the device and
# must not inherit its handles). Same interface as LiveWaterfall.
# ======================================================================
def _gui_main(q, freqs, cfg, span_sec, fps, fmax):
    from queue import Empty
    lw = LiveWaterfall(freqs, cfg, span_sec, fps, fmax)
    n = 0
    min_dt = 1.0 / max(fps, 1.0)       # target frame interval; draws are skipped, columns are not
    last_draw = 0.0
    while lw.enabled:
        try:
            batch = [q.get(timeout=0.1)]
        except Empty:
            try:
                lw.fig.canvas.flush_events()      # keep the window responsive when idle
            except Exception:
                break
            continue
        # Drain everything already queued: writing a column is ~free, drawing is not. If we
        # drew every k-th column of a backlog, any redraw slower than k steps would make the
        # queue (and the picture's lag) grow without bound, up to maxsize*20ms = 10 s.
        # Draining and drawing once per batch bounds the lag at one redraw time.
        try:
            while True:
                batch.append(q.get_nowait())
        except Empty:
            pass
        stop = False
        for item in batch:
            if item is None:
                stop = True
                break
            power, peak_hz, present, ridge_hz, raw_hit, t_sent = item
            lw.update(power, peak_hz, "", present, ridge_hz=ridge_hz, raw_hit=raw_hit,
                      t_sent=t_sent, draw=False)
            n += 1
        if stop:
            break
        if time.perf_counter() - last_draw >= min_dt:
            lw._k += 1
            lw._redraw()
            last_draw = time.perf_counter()
    print(f"[live figure] closed after {n} columns. {lw.status()}", flush=True)
    lw.close()


class LiveWaterfallProc:
    def __init__(self, freqs: np.ndarray, cfg: RtConfig, span_sec: float = 15.0,
                 fps: float = 12.0, fmax: float | None = None):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self._q = ctx.Queue(maxsize=512)
        self._p = ctx.Process(target=_gui_main, daemon=True,
                              args=(self._q, freqs, cfg, span_sec, fps, fmax))
        self._p.start()
        self.enabled = True
        self.dropped = 0
        self._k = 0

    def update(self, power_lin, peak_hz, status, present, ridge_hz=None,
               raw_hit=None) -> None:
        if not self.enabled:
            return
        self._k += 1
        if self._k % 50 == 0 and not self._p.is_alive():      # window closed
            self.enabled = False
            return
        try:
            self._q.put_nowait((np.array(power_lin, dtype=np.float32), float(peak_hz),
                                bool(present), ridge_hz, raw_hit, time.time()))
        except Exception:
            self.dropped += 1           # queue full: drop the picture, never block the DSP

    def status(self) -> str:
        return f"live figure: separate process, GUI queue-full drops={self.dropped}"

    def close(self) -> None:
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
        self._p.join(timeout=3.0)
        if self._p.is_alive():
            self._p.terminate()
        self.enabled = False
