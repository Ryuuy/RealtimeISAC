#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时多普勒入口。不写任何文件，不存任何历史。

必须用 /usr/bin/python3 跑（默认的 python3 是 miniconda，没有 uhd）：

    cd RealtimeISAC/realtime
    /usr/bin/python3 rt_main.py --duration 30
    /usr/bin/python3 rt_main.py --duration 30 --waterfall
    /usr/bin/python3 rt_main.py --dry-run          # 不碰硬件，用合成数据自检
"""

import argparse
import os
import resource
import sys
import time

import numpy as np

from rt_config import RtConfig
from rt_detect import PowerMonitor, PresenceDetector
from rt_display import PeakLine, Waterfall
from rt_dsp import DopplerEngine, to_db
from rt_live import LiveWaterfall
from rt_publish import Publisher, UdpSink, stdout_sink
from rt_record import DopplerRecorder
from rt_sync import TddSync

DEBUG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "debug")


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class Stats:
    """耗时/RSS 统计。只保留固定长度的环形样本，不无限增长。"""

    def __init__(self, cap: int = 4096):
        self._t = np.zeros(cap, dtype=np.float64)
        self._n = 0
        self._cap = cap
        self.rss_start = rss_mb()

    def add(self, dt_ms: float) -> None:
        self._t[self._n % self._cap] = dt_ms
        self._n += 1

    def summary(self, budget_ms: float) -> str:
        if self._n == 0:
            return "无样本"
        v = self._t[:min(self._n, self._cap)]
        p50, p99, mx = np.percentile(v, 50), np.percentile(v, 99), v.max()
        return (f"步数={self._n}  p50={p50:.3f}ms  p99={p99:.3f}ms  max={mx:.3f}ms  "
                f"预算={budget_ms:.0f}ms  余量={budget_ms/max(p99,1e-9):.0f}x\n"
                f"RSS: {self.rss_start:.0f} -> {rss_mb():.0f} MB "
                f"(增长 {rss_mb()-self.rss_start:+.1f} MB)")


def run(cfg: RtConfig, duration: float, waterfall: bool, dry_run: bool,
        publish: bool = False, debug_dir: str | None = None,
        debug_max_sec: float = 0.0, live: bool = False,
        live_sec: float = 15.0, live_fps: float = 12.0,
        live_fmax: float | None = None, sink=None,
        heartbeat_sec: float = 2.0) -> int:
    print("=== 实时多普勒 + 有无检测 (RealtimeISAC) ===")
    print(cfg.describe())
    print(f"DC保护带 ±{cfg.dc_guard_hz:.0f}Hz  "
          f"能量判据: {'基线(环' + f'{cfg.energy_ring_sec:.0f}s p{cfg.energy_pct:.0f}+棘轮' + f'{cfg.energy_rise_db_s}dB/s) +{cfg.energy_margin_db:.0f}dB' if cfg.use_energy else '屏蔽(debug CFAR用)'}  "
          f"{'OR CFAR' if cfg.use_cfar else 'CFAR关'}(门限+{cfg.cfar_threshold_db:g}dB,"
          f"连续游程>={cfg.cfar_min_run}bin)")
    print(f"去抖 {cfg.debounce_enter}/{cfg.debounce_exit} of {cfg.debounce_n} "
          f"(窗 {cfg.debounce_n*cfg.step_sec:.2f}s)")
    print(f"数据率: {cfg.sample_rate*4*cfg.num_channels/1e6:.0f} MB/s"
          f"{'  <-- 接近 B210 USB3 上限' if cfg.sample_rate*4*cfg.num_channels > 200e6 else ''}")
    print("-" * 68)

    sync = TddSync(cfg)
    engine = DopplerEngine(cfg)
    power_mon = PowerMonitor(cfg)
    detector = PresenceDetector(cfg, engine.dc_mask, engine.freqs)
    if sink is None and publish:
        sink = stdout_sink
    pub = Publisher(sink=sink).start()
    if sink is not None:
        print(f"📡 判决外发: {sink!r}  心跳 {heartbeat_sec:.0f}s")
    stats = Stats()
    peak_line = PeakLine(every=5)
    wf = Waterfall(engine.freqs, every=2) if waterfall else None
    if wf:
        print(wf.header)

    lw = LiveWaterfall(engine.freqs, cfg, live_sec, live_fps, live_fmax) if live else None
    if lw is not None and lw.enabled:
        print(f"📈 实时瀑布图已开：{live_sec:.0f}s 窗口, 目标 {live_fps:.0f}fps"
              f"{f', ±{live_fmax:.0f}Hz' if live_fmax else ''}  (关窗口即停止绘图)")

    # debug 录制：默认不建对象，热路径上没有任何额外开销
    rec = None
    if debug_dir:
        rec = DopplerRecorder(cfg, debug_dir, debug_max_sec or (duration + 5.0))
        print(f"🔴 debug 录制已开启 -> {debug_dir}  "
              f"(上限 {rec.n_max} 帧, 预分配 {rec.nbytes/1e6:.1f} MB)")

    src = None
    last_present = None
    last_beat = 0.0
    try:
        if dry_run:
            stream = _synthetic_steps(cfg, duration)
            errors = None
        else:
            from rt_source import UsrpSource
            src = UsrpSource(cfg).open()
            print(f"设备已开流: 实际采样率={src.actual_rate/1e6:.4f}MHz  spb={src.spb}")
            print("-" * 68)
            stream = src.steps()
            errors = src.errors

        t_run0 = time.monotonic()
        t_end = t_run0 + duration
        for raw in stream:
            t0 = time.perf_counter()
            sync.update(raw)
            power_mon.update(raw)
            engine.ingest(raw)
            power = engine.process(sync.phase, sync.n_integrate)
            fd, pk = engine.peak(power)
            # ring 灌满前谱里还混着初始零值，判决不可信，这段不喂给检测器
            present = detector.update(power, engine.freqs) if engine.ready else False
            # 录制放在计时区**内**：它确实占用每步预算，藏起来只会让 p99 骗人。
            # 实测开销 ~1µs（一次 2KB memcpy），对 4ms 的步长可以忽略。
            if rec is not None:
                rec.add(time.monotonic() - t_run0, engine.last_spectrum, present,
                        detector.n_hits, detector.max_run, detector.peak_hz, engine.ready,
                        power_mon.inst, detector.energy.e_db,
                        detector.energy.floor_db or 0.0)
            tag = "" if engine.ready else "[灌注中]"
            # 实时图也算进计时区：它每 4 步要花几毫秒，是本方案里最贵的一项，
            # 挪到计时区外面只会让 p99 好看而不解决丢步。分项耗时见 lw.status()。
            if lw is not None:
                lw.update(power, fd, f"{detector.status()} {power_mon.status()} {tag}",
                          present)
            dt_ms = (time.perf_counter() - t0) * 1e3
            stats.add(dt_ms)

            # 状态翻转时立即发；另外每 heartbeat_sec 重发一次当前状态。
            # 心跳是给下游用的：桥接进程后启动、或丢了个 UDP 包时，最多一个
            # 心跳周期就能同步回来；下游也能靠"多久没收到"判断本进程还活着没。
            now = time.monotonic()
            if engine.ready and (present != last_present
                                 or now - last_beat >= heartbeat_sec):
                pub.send({"present": bool(present),
                          "doppler_hz": round(detector.peak_hz, 1),
                          "bins": detector.n_hits, "run": detector.max_run,
                          "score": detector.score,
                          "energy_db": round(detector.energy.margin_db, 1),
                          "t": round(time.time(), 3)})
                last_present = present
                last_beat = now

            if wf:
                wf.update(to_db(power), f"{fd:+7.1f}Hz {detector.status()} {tag}")
            else:
                peak_line.update(fd, pk, f"{dt_ms:5.2f}ms {detector.status()} "
                                         f"{power_mon.status()} {tag}")
            if time.monotonic() >= t_end:
                break
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，停止。")
    finally:
        if src is not None:
            src.close()
        pub.stop()
        if lw is not None:
            lw.close()

    print("\n" + "-" * 68)
    print(stats.summary(cfg.step_sec * 1e3))
    print(sync.status())
    print(power_mon.status())
    print(f"判决: {'有人' if detector.present else '无人'}  "
          f"score={detector.score}/{cfg.debounce_n}")
    print(pub.status())
    if lw is not None:
        print(lw.status())
    if rec is not None:
        print(rec.status())
        path = rec.save(engine.freqs, extra={
            "sync_locked": sync.locked, "sync_contrast": round(sync.contrast, 4),
            "sync_duty": round(sync.duty, 4), "sync_phase": sync.phase,
            "sync_n_integrate": sync.n_integrate,
            "power_dbfs": [round(d, 2) for d in power_mon.dbfs if d is not None],
            "dry_run": dry_run,
            "uhd_errors": str(errors) if errors is not None else None,
        })
        if path:
            print(f"💾 已存 {path}")
            print(f"   出图: /usr/bin/python3 {os.path.join(DEBUG_DIR, 'plot_doppler.py')}")
    if not dry_run and errors is not None:
        print(f"UHD 错误: {errors}   主动丢步={src.dropped if src else 0}")
        if errors.overflow:
            print("⚠️  出现 overflow：采集侧跟不上。先排查 USB 控制器/线缆，"
                  "再考虑降采样率——计算侧余量见上方 p99。")
        else:
            print("✅ 无 overflow")
    return 0


def _synthetic_steps(cfg: RtConfig, duration: float, target_period: float = 4.0):
    """不碰硬件的自检数据源：造带 TDD 结构 + 一个多普勒分量的假数据。

    目标按 target_period 的周期**开一半关一半**（默认 2s 有 / 2s 无），不是全程常开。
    这不只是为了好看——全程恒定的单音会在 ~4s 内被 ClutterMap 学进背景然后消失
    （匀速不变的目标本来就是它的设计盲区），自检会假阴性。有开有关才真正走通
    「基线学习 -> 目标出现 -> 检出 -> 目标消失 -> 释放」这条完整链路。
    """
    rng = np.random.default_rng(0)
    n = cfg.n_step
    buf = np.zeros((cfg.num_channels, n * 2), dtype=np.int16)
    on = np.zeros(n, dtype=bool)
    for k in range(cfg.n_frames_per_step):
        on[k * cfg.n_period:k * cfg.n_period + cfg.n_on] = True
    on2 = np.repeat(on, 2)
    t_end = time.monotonic() + duration
    phase = 0.0
    while time.monotonic() < t_end:
        # 两路必须灌**同一份**波形（各自再叠独立噪声），否则 conj(ch0)*ch1 是
        # 噪声乘噪声，下面注入的 120Hz 多普勒根本出不来，频率轴就白验了。
        sig = rng.integers(-8000, 8000, n * 2).astype(np.int16)
        for ch in range(cfg.num_channels):
            buf[ch, on2] = sig[on2] + rng.integers(-400, 400, on2.sum(), dtype=np.int16)
        buf[:, ~on2] = rng.integers(-200, 200, (cfg.num_channels, (~on2).sum()),
                                    dtype=np.int16)
        # 给 ch1 叠一个 120Hz 的多普勒旋转，验证频率轴是否正确
        on = (phase % target_period) >= (target_period / 2)
        if on:
            t = np.arange(n) / cfg.sample_rate + phase
            rot = np.exp(1j * 2 * np.pi * 120.0 * t)
            c1 = (buf[1, 0::2].astype(np.float32)
                  + 1j * buf[1, 1::2].astype(np.float32)) * rot
            buf[1, 0::2] = np.clip(c1.real, -32767, 32767).astype(np.int16)
            buf[1, 1::2] = np.clip(c1.imag, -32767, 32767).astype(np.int16)
        phase += cfg.step_sec
        yield buf
        time.sleep(cfg.step_sec)


def main() -> int:
    p = argparse.ArgumentParser(description="实时多普勒 (不落盘)")
    p.add_argument("--duration", type=float, default=20.0, help="运行秒数")
    p.add_argument("--fs", type=float, default=10.0, help="采样率 MHz")
    p.add_argument("--gain", type=float, default=30.0)
    p.add_argument("--serial", type=str, default="32392D3")
    p.add_argument("--step", type=float, default=0.02, help="步进秒")
    p.add_argument("--window", type=float, default=0.2, help="窗长秒")
    p.add_argument("--waterfall", action="store_true", help="ASCII 瀑布图")
    p.add_argument("--dry-run", action="store_true", help="不碰硬件，用合成数据自检")
    p.add_argument("--publish", action="store_true",
                   help="把有人/无人判决以 JSON 外发（默认打到 stdout，换成你的 IP 推送即可）")
    p.add_argument("--debug", action="store_true",
                   help="录制多普勒谱到 debug/（唯一会落盘的开关，默认关）")
    p.add_argument("--debug-dir", type=str, default=DEBUG_DIR,
                   help=f"录制输出目录（默认 {DEBUG_DIR}）")
    p.add_argument("--debug-max-sec", type=float, default=0.0,
                   help="录制缓冲上限秒数（默认 = duration + 5）")
    p.add_argument("--live", action="store_true",
                   help="弹一个实时滚动多普勒瀑布图（数据从右往左填）")
    p.add_argument("--live-sec", type=float, default=15.0, help="实时图时间窗(秒)")
    p.add_argument("--live-fps", type=float, default=12.0, help="实时图目标刷新率")
    p.add_argument("--live-fmax", type=float, default=None,
                   help="实时图只画 ±这么多 Hz（默认全量程 ±500）")
    p.add_argument("--isac", action="store_true",
                   help="把有人/无人以 UDP 发给 isac_bridge.py（-> 网页 shadowing/normal）")
    p.add_argument("--isac-host", type=str, default="127.0.0.1")
    p.add_argument("--isac-port", type=int, default=9099)
    p.add_argument("--heartbeat", type=float, default=2.0,
                   help="每隔这么多秒重发一次当前状态（0 = 只在翻转时发）")
    p.add_argument("--n-on-us", type=float, default=0.0,
                   help="强制每帧积分多少微秒（0=用实测的 ON 段长度，对比实验用）")
    p.add_argument("--clutter-map", action="store_true",
                   help="打开 CFAR 前的逐bin背景归一化（默认关，实测帮助不大）")
    p.add_argument("--no-tracking-cal", action="store_true",
                   help="关掉 AD9361 的 DC offset / IQ 平衡跟踪校准（默认开；"
                        "关掉实测会制造虚警，只用于对比实验）")
    p.add_argument("--no-energy", action="store_true",
                   help="关掉宽带能量主判据，只看 CFAR 裸跑（默认开；"
                        "仅用于单独 debug CFAR，别用来跑生产）")
    a = p.parse_args()
    cfg = RtConfig(serial=a.serial, sample_rate=a.fs * 1e6, gain=a.gain,
                   step_sec=a.step, window_sec=a.window,
                   force_n_on_us=a.n_on_us,
                   use_clutter_map=a.clutter_map,
                   rx_tracking_cal=not a.no_tracking_cal,
                   use_energy=not a.no_energy)
    sink = UdpSink(a.isac_host, a.isac_port) if a.isac else None
    return run(cfg, a.duration, a.waterfall, a.dry_run, a.publish,
               debug_dir=a.debug_dir if a.debug else None,
               debug_max_sec=a.debug_max_sec, live=a.live, live_sec=a.live_sec,
               live_fps=a.live_fps, live_fmax=a.live_fmax, sink=sink,
               heartbeat_sec=a.heartbeat)


if __name__ == "__main__":
    sys.exit(main())
