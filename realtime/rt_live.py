#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时滚动多普勒瀑布图（matplotlib 窗口，数据从右往左填进来）。

## 为什么这东西能跑得起来（设计文档里说"终端打印 > matplotlib"，那是对的，但没说不能画）

关键不在 matplotlib 快不快，在于**画图的节拍和 DSP 的节拍解耦**：

- DSP 照旧 50 步/秒（每 20ms 一帧），每帧只往环形列缓冲里写 **一列 256 个 float**
  （~5µs，一次 to_db + 一次 memcpy），这部分开销可以忽略
- **重绘降频到 ~12 fps**（每 4 步画一次）。人眼看瀑布图 12fps 完全够，
  而每步摊到的画图成本降到 1/4

## 三个必须做对的地方

1. **必须 blit**，不能 `fig.canvas.draw()`。全量重绘要重画坐标轴/刻度/色标，
   40-80ms 一次，直接顶穿 20ms 预算。blit 只重画 AxesImage 本身，实测 3-8ms。
2. **滚动不能每步 np.roll**。改成环形写指针 + 重绘时两次 copyto 拼进预分配的
   显示数组：零分配，而且移位只在重绘时发生（12 次/秒而不是 50 次/秒）。
3. **色标不能每帧自适应**，否则画面一直闪。每 3 秒用分位数重算一次，
   而且只在变化 >1.5dB 时才真的改（改色标要全量重绘 + 重新抓 background）。

## 丢步比 overflow 更要命

生产者线程满了会**主动丢整步**（`src.dropped`）。丢步 = 环形缓冲相位不连续
= **多普勒频率算错**，而且不会有任何报错。所以这里自带降级：重绘耗时超过
每步预算的 40% 就自动砍半帧率，并在收尾时报告。跑完务必看一眼 `主动丢步=0`。
"""

import os
import time

import numpy as np

from rt_config import RtConfig
from rt_dsp import to_db


class LiveWaterfall:
    def __init__(self, freqs: np.ndarray, cfg: RtConfig, span_sec: float = 15.0,
                 fps: float = 12.0, fmax: float | None = None):
        self.cfg = cfg
        self.enabled = False
        self.draw_ms_p99 = 0.0
        self._draw_ms = []
        self._degraded = 0

        rows = (np.flatnonzero(np.abs(freqs) <= fmax) if fmax is not None
                else np.arange(len(freqs)))
        if rows.size < 2:
            rows = np.arange(len(freqs))
        self._rows = rows
        f_lo, f_hi = float(freqs[rows[0]]), float(freqs[rows[-1]])

        self._ncols = max(8, int(round(span_sec / cfg.step_sec)))
        # NaN = 还没数据 -> 画成背景色，于是画面是"从右往左慢慢填满"的
        self._buf = np.full((rows.size, self._ncols), np.nan, dtype=np.float32)
        self._disp = np.empty_like(self._buf)
        self._p = 0
        self._n_written = 0

        self._every = max(1, int(round(1.0 / (fps * cfg.step_sec))))
        self._k = 0
        self._clim_every = max(1, int(round(3.0 / cfg.step_sec)))
        self._clim = None

        import matplotlib
        if not os.environ.get("DISPLAY"):
            print("⚠️  没有 DISPLAY，实时图开不了（改用 --waterfall 的 ASCII 版）")
            return
        import matplotlib.pyplot as plt

        plt.ion()
        self._plt = plt
        self.fig, self.ax = plt.subplots(figsize=(12, 5.5))
        self.fig.canvas.manager.set_window_title("RealtimeISAC — Live Doppler")
        cmap = matplotlib.cm.get_cmap("jet").copy()
        cmap.set_bad(color="#101018")          # NaN（还没填到的区域）的颜色
        self._im = self.ax.imshow(
            self._disp, aspect="auto", origin="lower", cmap=cmap,
            interpolation="nearest", extent=[-span_sec, 0.0, f_lo, f_hi],
            vmin=-80, vmax=-40)
        self.ax.set_xlabel("Time (s, 0 = now)")
        self.ax.set_ylabel("Doppler Frequency (Hz)")
        self.ax.set_facecolor("#101018")
        g = cfg.dc_guard_hz
        for s in (-g, g):
            self.ax.axhline(s, color="w", ls="--", lw=0.7, alpha=0.45)
        self.fig.colorbar(self._im, ax=self.ax, label="Relative Power (dB)")
        self._txt = self.ax.text(
            0.01, 0.97, "", transform=self.ax.transAxes, va="top", ha="left",
            fontsize=10, family="monospace", color="w",
            bbox=dict(fc="#000000A0", ec="none", pad=3))

        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self._bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.fig.canvas.flush_events()
        self.enabled = True

    def _on_close(self, _evt) -> None:
        self.enabled = False

    def update(self, power_lin: np.ndarray, peak_hz: float, status: str,
               present: bool) -> None:
        """每步调用。写一列是廉价的；真正的重绘每 _every 步才做一次。"""
        if not self.enabled:
            return
        self._buf[:, self._p] = to_db(power_lin[self._rows])
        self._p = (self._p + 1) % self._ncols
        self._n_written += 1

        self._k += 1
        if self._k % self._every:
            return

        t0 = time.perf_counter()
        p, n = self._p, self._ncols
        # 环形 -> 顺序：最老的一列在 p（下一个要被覆盖的位置），最新的在 p-1
        np.copyto(self._disp[:, :n - p], self._buf[:, p:])
        np.copyto(self._disp[:, n - p:], self._buf[:, :p])
        self._im.set_data(self._disp)

        full = False
        if self._n_written >= 8 and self._k % self._clim_every < self._every:
            v = self._disp[np.isfinite(self._disp)]
            if v.size > 32:
                lo, hi = np.percentile(v, (5, 99))
                if self._clim is None or abs(lo - self._clim[0]) > 1.5 \
                        or abs(hi - self._clim[1]) > 1.5:
                    self._clim = (float(lo), float(hi + 2.0))
                    self._im.set_clim(*self._clim)
                    full = True     # 色标变了，必须全量重绘并重抓 background

        self._txt.set_text(f"{peak_hz:+7.1f} Hz   {status}")
        self._txt.set_color("#ff5555" if present else "w")

        cv = self.fig.canvas
        try:
            if full:
                cv.draw()
                self._bg = cv.copy_from_bbox(self.ax.bbox)
            else:
                cv.restore_region(self._bg)
                self.ax.draw_artist(self._im)
                self.ax.draw_artist(self._txt)
                cv.blit(self.ax.bbox)
            cv.flush_events()
        except Exception:
            self.enabled = False       # 窗口被关掉了，别再画
            return

        dt = (time.perf_counter() - t0) * 1e3
        self._draw_ms.append(dt)
        if len(self._draw_ms) > 512:
            self._draw_ms.pop(0)
        # 自动降级：画一次的钱超过每步预算的 40% 就砍半帧率，最多砍 3 次
        if dt > 0.4 * self.cfg.step_sec * 1e3 and self._degraded < 3 and not full:
            self._every *= 2
            self._degraded += 1
            print(f"\n⚠️  重绘 {dt:.1f}ms 太贵，帧率降到 "
                  f"{1.0/(self._every*self.cfg.step_sec):.1f} fps 保实时性")

    def status(self) -> str:
        if not self._draw_ms:
            return "实时图: 未启用" if not self.enabled else "实时图: 无重绘样本"
        v = np.array(self._draw_ms)
        return (f"实时图: 重绘 {len(self._draw_ms)} 次  p50={np.percentile(v,50):.1f}ms "
                f"p99={np.percentile(v,99):.1f}ms  "
                f"帧率={1.0/(self._every*self.cfg.step_sec):.1f}fps"
                + (f"  (自动降级 {self._degraded} 次)" if self._degraded else ""))

    def close(self) -> None:
        if getattr(self, "_plt", None) is None:
            return
        self._plt.ioff()
        self.enabled = False
