#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""终端显示。要求是**最快**而不是最好看——实测：单行峰值 3.7µs、ASCII 瀑布行 29.5µs，
而 matplotlib blitting 要 1-5ms、朴素重绘 50-200ms（直接超预算）。

所以这里没有 matplotlib，一行都没有。热循环里绝不能 import 它。
"""

import sys

import numpy as np

RAMP = np.array(list(" .:-=+*#%@"))


class PeakLine:
    """单行覆盖输出：峰值多普勒 + 幅度。最快的显示方式。"""

    def __init__(self, every: int = 1):
        self.every = max(1, every)
        self._n = 0

    def update(self, fd_hz: float, peak_db: float, extra: str = "") -> None:
        self._n += 1
        if self._n % self.every:
            return
        sys.stdout.write(f"\rfd={fd_hz:+7.1f}Hz  peak={peak_db:7.1f}dB  {extra}   ")
        sys.stdout.flush()


class Waterfall:
    """ASCII 瀑布：每帧一行，向下滚动。这就是"视频"，比 matplotlib 快两个数量级。

    dB 范围用滚动分位数自适应，避免一开始全白或全黑。
    """

    def __init__(self, freqs: np.ndarray, width: int = 72,
                 fmin: float = -500.0, fmax: float = 500.0, every: int = 2):
        self.every = max(1, every)
        self._n = 0
        mask = (freqs >= fmin) & (freqs <= fmax)
        self._idx = np.where(mask)[0]
        # 把选中的 bin 均匀映射到 width 个字符列
        self._cols = np.linspace(0, len(self._idx) - 1, width).astype(np.int32)
        self._lo, self._hi = None, None
        self.header = self._make_header(freqs[self._idx], width)

    @staticmethod
    def _make_header(f: np.ndarray, width: int) -> str:
        left, mid, right = f"{f[0]:.0f}", "0", f"{f[-1]:.0f}"
        pad = width - len(left) - len(mid) - len(right)
        return left + " " * (pad // 2) + mid + " " * (pad - pad // 2) + right

    def update(self, spec_db: np.ndarray, extra: str = "") -> None:
        self._n += 1
        if self._n % self.every:
            return
        row = spec_db[self._idx][self._cols]
        lo, hi = np.percentile(row, 10), np.percentile(row, 99)
        # 慢速跟踪显示范围，避免每帧跳动
        self._lo = lo if self._lo is None else 0.9 * self._lo + 0.1 * lo
        self._hi = hi if self._hi is None else 0.9 * self._hi + 0.1 * hi
        span = max(self._hi - self._lo, 1e-6)
        idx = np.clip(((row - self._lo) / span * (len(RAMP) - 1)), 0, len(RAMP) - 1)
        sys.stdout.write("".join(RAMP[idx.astype(np.int8)]) + f" {extra}\n")
        sys.stdout.flush()
