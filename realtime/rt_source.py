#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""USRP B210 取流。与旧的 USRPSaveData2Channel.py 的根本区别：**不写任何文件**，
recv 到的数据攒够一个 step 就交给上层处理，处理完就扔。

设备配置段（set_rx_rate/bandwidth/freq/gain/antenna）沿用旧代码里已验证可用的写法。

## 为什么必须用生产者线程（实测结论，别改回单线程）

recv() 是**实时节拍**的：拉 20ms 的数据就得等 20ms（实测吞吐恰好 80MB/s @10MHz）。
单线程里再串一个 4-6ms 的 DSP，一轮就是 25ms > 20ms 预算，**天生落后 25%**，
必然溢出——实测 30 秒跑出 781 次 overflow。

所以 recv 必须独占一个线程持续排空驱动缓冲，DSP 在另一个线程跑。
DSP 只占 20-30% 占空，解耦后余量充足。

## 另一个实测点：recv 会填满大缓冲

`get_max_num_samps()` 只有 2040，但传一个 200,000 样本的缓冲进去，
**一次 recv 就返回 200,000**（UHD 内部会连续收多个包）。
所以直接 recv 进整个 step 缓冲，省掉 ~98 次 Python 调用和一次整块拷贝。
"""

import threading
from queue import Empty, Full, Queue

import numpy as np
import uhd

from rt_config import SC16_DTYPE, RtConfig


class OverflowCounter:
    """UHD 溢出统计。溢出意味着采集侧跟不上（USB 或 CPU），必须记录而不是静默忽略。"""

    def __init__(self):
        self.overflow = 0
        self.late = 0
        self.timeout = 0
        self.other = 0

    def record(self, err) -> None:
        C = uhd.types.RXMetadataErrorCode
        if err == C.none:
            return
        if err == C.overflow:
            self.overflow += 1
        elif err == C.late:
            self.late += 1
        elif err == C.timeout:
            self.timeout += 1
        else:
            self.other += 1

    @property
    def total(self) -> int:
        return self.overflow + self.late + self.timeout + self.other

    def __str__(self) -> str:
        return (f"overflow={self.overflow} late={self.late} "
                f"timeout={self.timeout} other={self.other}")


class UsrpSource:
    """双通道 sc16 取流。用法：

        with UsrpSource(cfg) as src:
            for raw_i16 in src.steps():   # raw_i16: (2, n_step*2) int16 视图
                ...
    """

    def __init__(self, cfg: RtConfig, n_buffers: int = 4):
        self.cfg = cfg
        self.errors = OverflowCounter()
        self.dropped = 0          # 消费者跟不上时主动丢的 step 数
        self.usrp = None
        self.streamer = None
        self._metadata = None
        # 缓冲池。用 SC16_DTYPE 分配（UHD 靠 itemsize=4 认 sc16），
        # 同时给 DSP 一个 int16 连续视图——同一块内存，零拷贝。
        self._pool = [np.zeros((cfg.num_channels, cfg.n_step), dtype=SC16_DTYPE)
                      for _ in range(n_buffers)]
        self._views = {id(b): b.view(np.int16) for b in self._pool}
        self._free: "Queue[np.ndarray]" = Queue()
        self._full: "Queue[np.ndarray]" = Queue()
        for b in self._pool:
            self._free.put(b)
        self._stop = threading.Event()
        self._thread = None

    # ---- 生命周期 ----
    def open(self) -> "UsrpSource":
        cfg = self.cfg
        self.usrp = uhd.usrp.MultiUSRP(f"serial={cfg.serial}")
        for ch in range(cfg.num_channels):
            self.usrp.set_rx_rate(cfg.sample_rate, ch)
            self.usrp.set_rx_bandwidth(cfg.sample_rate, ch)
            self.usrp.set_rx_gain(cfg.gain, ch)
            self.usrp.set_rx_antenna("RX2", ch)
            # AD9361 的后台跟踪校准默认是开的。它周期性更新 DC offset / IQ 平衡
            # 修正量，而修正量随输入信号大小变化 —— 对强信号就是一个周期性的
            # **乘性**扰动，在多普勒谱上现形为固定线谱。每通道独立跑，所以两路
            # 之间没有固定相位关系。相干处理必须关掉它们，用固定增益。
            try:
                self.usrp.set_rx_agc(cfg.rx_agc, ch)
            except Exception:
                pass                      # 有些固件/版本不支持，忽略
            self.usrp.set_rx_dc_offset(cfg.rx_tracking_cal, ch)
            self.usrp.set_rx_iq_balance(cfg.rx_tracking_cal, ch)
        self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(cfg.center_freq))

        st_args = uhd.usrp.StreamArgs("sc16", "sc16")
        st_args.channels = list(range(cfg.num_channels))
        st_args.args = uhd.types.DeviceAddr(f"num_recv_frames={cfg.num_recv_frames}")
        self.streamer = self.usrp.get_rx_stream(st_args)

        self._spb = self.streamer.get_max_num_samps()
        self._sink = np.zeros((cfg.num_channels, cfg.n_step), dtype=SC16_DTYPE)
        self._metadata = uhd.types.RXMetadata()

        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        # 多通道流不能用 stream_now（UHD 会报 "fail to time align"），必须给时间戳
        cmd.stream_now = False
        cmd.time_spec = self.usrp.get_time_now() + uhd.libpyuhd.types.time_spec(0.2)
        self.streamer.issue_stream_cmd(cmd)
        self.streamer.recv(self._sink, self._metadata, timeout=0.5)  # 预热

        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.streamer is not None:
            try:
                self.streamer.issue_stream_cmd(
                    uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
            except Exception:
                pass
        # 显式析构，触发 C++ 侧释放硬件——旧代码踩过不这样做会崩的坑
        self.streamer = None
        self.usrp = None

    def _fill(self, buf) -> bool:
        """把 buf 填满一个 step。返回是否填满（被 stop 打断则 False）。"""
        n_step = self.cfg.n_step
        got = 0
        while got < n_step:
            if self._stop.is_set():
                return False
            n = self.streamer.recv(buf[:, got:] if got else buf,
                                   self._metadata, timeout=1.0)
            self.errors.record(self._metadata.error_code)
            if n > 0:
                got += n
        return True

    def _producer(self) -> None:
        """独占线程：持续排空驱动缓冲，绝不被 DSP 阻塞。"""
        while not self._stop.is_set():
            try:
                buf = self._free.get_nowait()
            except Empty:
                # 消费者跟不上：主动丢一整步（仍要 recv 以免驱动侧溢出）。
                # 有意识地丢，好过让 UHD 静默 overflow。
                self.dropped += 1
                self._fill(self._sink)
                continue
            if not self._fill(buf):
                return
            try:
                self._full.put_nowait(buf)
            except Full:
                self._free.put(buf)

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def actual_rate(self) -> float:
        return self.usrp.get_rx_rate(0)

    @property
    def spb(self) -> int:
        return self._spb

    # ---- 取数 ----
    def steps(self):
        """产出 (2, n_step*2) 的 int16 视图，每个代表 step_sec 的新数据。

        注意：产出的是缓冲池里的**视图**，归还后会被生产者复用覆盖。
        消费者必须在本次迭代内用完，不能存起来——这正是"不存储任何东西"的体现。
        """
        while not self._stop.is_set():
            try:
                buf = self._full.get(timeout=2.0)
            except Empty:
                if self._stop.is_set():
                    return
                continue
            try:
                yield self._views[id(buf)]
            finally:
                self._free.put(buf)
