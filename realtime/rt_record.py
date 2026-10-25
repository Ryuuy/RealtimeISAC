#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""debug 录制：把每帧多普勒谱存下来，供离线出图核对。

## 这是对「不落盘」硬约束的**显式例外**，默认关闭

实时链路的设计前提是不存 IQ、不存谱、不存历史。本模块只有 `rt_main --debug`
才会实例化；不加 `--debug` 时连对象都不建，热路径上一行代码都不多跑。

## 为什么攒在内存里、收尾才写一次

热循环的预算是 20ms/步，里面还要留给 USB 取流。每帧 open/write/flush 的抖动
（尤其机械盘或文件系统同步）足够把某一步顶穿预算，进而丢步——**丢步会让环形
缓冲的相位不连续，多普勒频率直接算错**，比 overflow 更隐蔽。所以：

- 缓冲**预分配定长**（按 --duration 算），add() 里只有一次 memcpy，无分配、无 IO
- 收尾时一次 np.savez 落盘
- 缓冲满了就停止录制并置 truncated 标志，**绝不动态扩容**（这台机器内存差，
  扩容 = 一次几十 MB 的 realloc + 拷贝，正好卡在实时循环里）

## 存复数谱而不是功率

实时判决只要 |S|²，但复数谱才画得出相位面板（CAF 那张图的第二栏）。
代价只是每帧 2KB，而验证「多普勒对不对」时相位是关键证据。
"""

import json
import os
import time

import numpy as np

from rt_config import RtConfig


class DopplerRecorder:
    def __init__(self, cfg: RtConfig, out_dir: str, max_sec: float):
        self.cfg = cfg
        self.out_dir = out_dir
        self.n_max = max(1, int(round(max_sec / cfg.step_sec)))
        self.n = 0
        self.truncated = False
        self.t_start_wall = time.time()

        n, nf = self.n_max, cfg.nfft
        self.spec = np.zeros((n, nf), dtype=np.complex64)
        self.t = np.zeros(n, dtype=np.float64)
        self.peak_hz = np.zeros(n, dtype=np.float32)
        self.n_hits = np.zeros(n, dtype=np.int16)
        self.max_run = np.zeros(n, dtype=np.int16)
        self.present = np.zeros(n, dtype=bool)
        self.ready = np.zeros(n, dtype=bool)
        # 每帧各通道的瞬时 dBFS。存这个是为了把"谱塌了"归因清楚：
        # 是射频侧被挡住/TX 停了（功率跟着塌），还是 DSP 侧同步丢了（功率不动）。
        self.pw = np.zeros((n, cfg.num_channels), dtype=np.float32)
        # 能量判据的两条曲线。存下来才能离线复盘"为什么那一刻判成了有人"，
        # 也是下次重标门限的输入。
        self.e_db = np.zeros(n, dtype=np.float32)
        self.floor_db = np.zeros(n, dtype=np.float32)

    @property
    def nbytes(self) -> int:
        return (self.spec.nbytes + self.t.nbytes + self.peak_hz.nbytes
                + self.n_hits.nbytes + self.max_run.nbytes
                + self.present.nbytes + self.ready.nbytes)

    def add(self, t_rel: float, spec, present: bool, n_hits: int, max_run: int,
            peak_hz: float, ready: bool, pw_dbfs=None,
            e_db: float = 0.0, floor_db: float = 0.0) -> None:
        i = self.n
        if i >= self.n_max:
            self.truncated = True
            return
        self.spec[i] = spec              # complex128 -> complex64，一次 memcpy
        self.t[i] = t_rel
        self.peak_hz[i] = peak_hz
        self.n_hits[i] = n_hits
        self.max_run[i] = max_run
        self.present[i] = present
        self.ready[i] = ready
        if pw_dbfs is not None:
            self.pw[i] = pw_dbfs
        self.e_db[i] = e_db
        self.floor_db[i] = floor_db
        self.n = i + 1

    def save(self, freqs: np.ndarray, extra: dict | None = None) -> str | None:
        """一次性落盘，返回文件路径。没录到东西就不建文件。"""
        if self.n == 0:
            return None
        os.makedirs(self.out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.t_start_wall))
        path = os.path.join(self.out_dir, f"doppler_{stamp}.npz")

        cfg = self.cfg
        meta = {
            "sample_rate": cfg.sample_rate, "center_freq": cfg.center_freq,
            "gain": cfg.gain, "serial": cfg.serial,
            "step_sec": cfg.step_sec, "window_sec": cfg.window_sec,
            "nfft": cfg.nfft, "fs_out": cfg.fs_out, "bin_hz": cfg.bin_hz,
            "n_ring": cfg.n_ring, "dc_guard_hz": cfg.dc_guard_hz,
            "tdd_period_sec": cfg.tdd_period_sec,
            "cfar_threshold_db": cfg.cfar_threshold_db, "cfar_train": cfg.cfar_train,
            "cfar_guard": cfg.cfar_guard, "cfar_min_run": cfg.cfar_min_run,
            "use_cfar": cfg.use_cfar,
            "energy_margin_db": cfg.energy_margin_db,
            "energy_fmax_hz": cfg.energy_fmax_hz,
            "energy_ring_sec": cfg.energy_ring_sec, "energy_pct": cfg.energy_pct,
            "energy_rise_db_s": cfg.energy_rise_db_s,
            "debounce": [cfg.debounce_enter, cfg.debounce_exit, cfg.debounce_n],
            "t_start_wall": self.t_start_wall,
            "t_start_str": time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(self.t_start_wall)),
            "n_frames": self.n, "truncated": self.truncated,
        }
        meta.update(extra or {})
        # meta 走 JSON 字符串而不是对象数组：npz 存 dict 要 allow_pickle，
        # 那既不安全也不好在别的工具里读。
        np.savez(path,
                 spec=self.spec[:self.n], t=self.t[:self.n], freqs=freqs,
                 peak_hz=self.peak_hz[:self.n], n_hits=self.n_hits[:self.n],
                 max_run=self.max_run[:self.n],
                 present=self.present[:self.n], ready=self.ready[:self.n],
                 pw=self.pw[:self.n], e_db=self.e_db[:self.n],
                 floor_db=self.floor_db[:self.n],
                 meta_json=np.array(json.dumps(meta, ensure_ascii=False)))
        return path

    def status(self) -> str:
        s = (f"debug录制: {self.n} 帧 ({self.n * self.cfg.step_sec:.1f}s), "
             f"缓冲 {self.nbytes / 1e6:.1f} MB")
        if self.truncated:
            s += "  ⚠️缓冲已满，后续帧被丢弃（用 --debug-max-sec 加大）"
        return s


class HRecorder:
    """--save-h：把 H（1kHz 复数相关系数 dec，多普勒谱就是对它加窗 FFT）本体落盘。

    和 DopplerRecorder 的区别：
    - 存的是 STFT 之前的复数序列，事后能换任何窗长/算法（过零、带符号质心、相位展开...）；
    - 边录边追加写盘（后台线程），Ctrl+C / 崩溃最多丢最后 ~10 s，不像 DopplerRecorder 收尾才写；
    - 带本机 time.time() 时间戳，跟 isac_bridge 的 session / 回车打点 / NTG csv 用同一个钟。
    数据量：H 8 KB/s + 幅度 8 KB/s，≈ 58 MB/小时。

    文件（同一个 stamp 前缀，目录由 --h-dir 指定）：
      h_<stamp>.h.bin     complex64  n_samples          H 的每个样本（1kHz）
      h_<stamp>.amp.bin   float32    n_samples x 2      每帧 ch0/ch1 RMS 幅度（满量程=1）
      h_<stamp>.t.bin     float64    n_steps            每步结束时的 time.time()（秒）
      h_<stamp>.flag.bin  uint8      n_steps            bit0=环已灌满(ready)  bit1=丢步/overflow冻结中
      h_<stamp>.json      元数据（开始时写一次，结束时更新）
    样本 k（第 j 步内第 i 个）的时间 ≈ t[j] - (n_fr-1-i)/fs_out。读取用 debug/load_h.py。
    """

    FLUSH_STEPS = 500          # 10 s 一次

    def __init__(self, cfg: RtConfig, out_dir: str, extra: dict | None = None):
        import queue
        import threading
        os.makedirs(out_dir, exist_ok=True)
        self.cfg = cfg
        self.t_start = time.time()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.t_start))
        self.prefix = os.path.join(out_dir, f"h_{stamp}")
        self.n_fr = cfg.n_frames_per_step
        n = self.FLUSH_STEPS
        self._h = np.zeros(n * self.n_fr, dtype=np.complex64)
        self._a = np.zeros((n * self.n_fr, 2), dtype=np.float32)
        self._t = np.zeros(n, dtype=np.float64)
        self._f = np.zeros(n, dtype=np.uint8)
        self.k = 0
        self.steps = 0
        self.meta = {"fs_out": cfg.fs_out, "step_sec": cfg.step_sec, "n_frames_per_step": self.n_fr,
                     "window_sec": cfg.window_sec, "sample_rate": cfg.sample_rate,
                     "center_freq": cfg.center_freq, "gain": cfg.gain, "serial": cfg.serial,
                     "t_start": self.t_start, "t_start_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t_start))}
        self.meta.update(extra or {})
        self._fh = {k: open(f"{self.prefix}.{k}.bin", "ab") for k in ("h", "amp", "t", "flag")}
        self._q = queue.Queue()
        self._th = threading.Thread(target=self._writer, daemon=True)
        self._th.start()
        self._write_meta(done=False)

    def _write_meta(self, done: bool):
        m = dict(self.meta, n_steps=self.steps, n_samples=self.steps * self.n_fr, complete=done)
        with open(self.prefix + ".json", "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)

    def _writer(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            h, a, t, f = item
            self._fh["h"].write(h.tobytes()); self._fh["amp"].write(a.tobytes())
            self._fh["t"].write(t.tobytes()); self._fh["flag"].write(f.tobytes())
            for fh in self._fh.values():
                fh.flush()

    def add(self, dec, amp, ready: bool, guard: bool) -> None:
        if dec is None:
            return
        j = self.k
        self._h[j * self.n_fr:(j + 1) * self.n_fr] = dec
        self._a[j * self.n_fr:(j + 1) * self.n_fr] = amp
        self._t[j] = time.time()
        self._f[j] = (1 if ready else 0) | (2 if guard else 0)
        self.k += 1
        self.steps += 1
        if self.k >= self.FLUSH_STEPS:
            self._flush()

    def _flush(self):
        n = self.k
        if n:
            self._q.put((self._h[:n * self.n_fr].copy(), self._a[:n * self.n_fr].copy(),
                         self._t[:n].copy(), self._f[:n].copy()))
        self.k = 0

    def close(self, extra: dict | None = None) -> str:
        self._flush()
        self._q.put(None)
        self._th.join(timeout=10)
        for fh in self._fh.values():
            fh.close()
        self.meta.update(extra or {})
        self._write_meta(done=True)
        return self.prefix

    def status(self) -> str:
        return f"save-h: {self.steps} 步 ({self.steps * self.cfg.step_sec:.1f}s) -> {self.prefix}.*"
