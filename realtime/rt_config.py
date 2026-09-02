#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时 ISAC 配置。所有可调参数集中在这里，不要散到各模块去。"""

from dataclasses import dataclass

import numpy as np

# sc16 存盘/传输格式：每复数样点 = 2 个 int16 (re, im)，4 字节。
# UHD 的 recv() 靠缓冲的 itemsize 判断每样点字节数，所以缓冲必须用这个 dtype 分配；
# 处理时再 .view(np.int16) 拿连续视图（同一块内存，零拷贝）。
SC16_DTYPE = np.dtype([('re', np.int16), ('im', np.int16)])
SC16_SCALE = 32767.0


@dataclass
class RtConfig:
    # ---- 设备 ----
    # 注意：这台机器上当前接的是 32392D3，不是旧代码里写死的 321D889。
    serial: str = "32392D3"
    sample_rate: float = 10e6
    center_freq: float = 1.89e9      # 毫米波前端下变频后的 IF，不是真实载波
    gain: float = 30.0
    num_channels: int = 2
    num_recv_frames: int = 1024
    # AD9361 的后台跟踪校准（DC offset / IQ 正交平衡）。
    # ⚠️ 曾经把它设成 False 想压 ±100/±200Hz 的线谱，**实测反而制造虚警，已退回 True**。
    # 原因：关掉 DC 校正后两路各留一个未修正的直流，在 conj(ch0)*ch1 里产生
    # conj(dc0)*s1(t) 和 conj(s0(t))*dc1 两个**跟着信号走**的交叉项——它们不在 0Hz，
    # DC 的 EMA 扣不掉，直接抬高非DC能量，也就直接抬高主判据。
    # 想再试就用 rt_diag.py --spur 先 A/B，别直接改默认。
    rx_tracking_cal: bool = True
    rx_agc: bool = False        # AGC 必须关：相干处理不能让增益自己乱跑

    # ---- 时间参数 ----
    step_sec: float = 0.02           # 每次出谱的间隔（也是每次拉多少新数据）
    window_sec: float = 0.2          # 相干积累窗长 -> 频率分辨率 = 1/window_sec

    # ---- TDD 结构 ----
    # 周期 1ms（真机自相关实测 1000µs，谐波干净）-> 帧率 1kHz
    # -> 多普勒不混叠上限 ±500Hz。
    tdd_period_sec: float = 1e-3
    # ⚠️ 真机实测占空是 71.5%（ON 715µs / OFF 285µs），不是当初以为的 50%。
    # 这个值现在**只用于 dry-run 合成数据和参数打印**——真实门控窗宽由
    # TddSync 每次从数据里量（见 rt_sync 顶部注释里"占空比必须实测"那段）。
    tdd_duty: float = 0.7
    # 盲同步：功率包络先抽取这么多倍再折叠（边沿精度 = decim/fs，10MHz 下 1µs）
    sync_decim: int = 10
    # ON 段两端各内缩这么多秒，避开上下沿的滤波暂态（实测下降沿拖尾 ~80µs）
    sync_edge_guard_sec: float = 10e-6
    # >0 则**强制**每帧积分这么多微秒（仍从实测的 ON 段起点开始切），
    # 0 = 用实测的 ON 段长度。只为做对比实验用，正常别设。
    # 实测结论：400/500/695/1000µs 对 ±100/±200Hz 杂散**毫无影响**（都在 +12~13dB），
    # 因为那是 ch1 收到的信号自己的幅度调制，不是门控切出来的。
    force_n_on_us: float = 0.0
    # ON 段内外的功率对比度低于此值 -> 认为没有 TDD 结构，降级为自由运行。
    # 实测：接上 TX = 1.00，纯接收机噪声 < 0.1，阈值放 0.30 有 3x 余量。
    sync_min_contrast: float = 0.30
    # 每隔多少 step 重新跟一次 TDD 相位（应对 TX/RX 时钟漂移）
    sync_every_n_steps: int = 10

    # ---- 频谱 ----
    nfft: int = 256
    # 时域去 DC 已从 rt_dsp 删除（每 step 块状相减会造 ±50·k Hz 线谱梳，见 rt_dsp
    # process() 里的注释）。这个系数现在**只被 debug/ablate_doppler.py 的对比实验用**，
    # 主链路不再引用；静态杂波靠下面 dc_guard_hz 的频域保护带兜。
    dc_ema_alpha: float = 0.01
    # DC 保护带：静态信道的 conj 乘积在 0Hz 堆一个很大的常量，加窗后会向两侧泄漏。
    # Blackman 主瓣宽度 = 6*fs_out/n_ring = 6*1000/200 = 30Hz，即 DC 能量铺满 ±15Hz。
    # 所以保护带必须 >= 15Hz，取 20Hz 留余量（早先设的 6Hz 挡不住，会把 DC 当成目标）。
    # 代价：看不到 v < 20Hz*λ/2 的极慢运动（25GHz 下约 0.12 m/s），可接受。
    dc_guard_hz: float = 20.0

    # 2026-08-16：TX 自己有一套周期性调制，物理上已经查清楚（不是本地DSP问题），
    # 出现在 50Hz 的整数倍附近，具体哪几条强、多强会随会话变化——8/15 只有
    # ±100/±200Hz 明显；这次(0816)真机实测 ±100/±200Hz 到了 +32~35dB(6~8bin)，
    # ±300/400Hz +22~25dB(5~6bin)，±50/150Hz +9~10dB(3~4bin)，±250/350/450Hz
    # 干净。因为是 TX 发射信号自带的调制（不依赖多径），两路天线收到的几乎
    # 完全同步，归一化成相关系数后逼近 ρ≈1，比人体这种弱相干散射显眼得多——
    # 单条线自己就能占到 6~8 个连续 bin，宽度判据（cfar_min_run）单靠自己
    # 已经防不住，EnergyDetector 的宽带平均功率也会被这几条线直接抬高。
    # 解法：跟 DC guard 一样开陷波口，CFAR 和 EnergyDetector 共用同一个
    # "可用 bin" 掩码（见 rt_dsp.valid_mask()）。代价是这几个频点附近看不到
    # 目标（跟 DC guard 的盲区同类型），当前先按两次实测都出现过的频点收窄，
    # 干净的 250/350/450Hz 先留着能用。
    notch_freqs_hz: tuple[float, ...] = (50.0, 100.0, 150.0, 200.0, 300.0, 400.0)
    notch_guard_hz: float = 20.0     # 每个陷波口的半宽，覆盖实测最宽 ~16Hz 还留余量

    # ---- 宽带能量检测（主判据，2026-08-15 用真值标注重标）----
    # ⚠️ 为什么主判据不是 CFAR：真机实测「有人 vs 无人」的非DC平均功率差 34.3dB，
    # 但 CA-CFAR 的噪声估计同时涨了 34.7dB —— 净剩 -0.4dB。原因是训练区单侧只跨
    # 62Hz，而人走路的微多普勒铺满 ±400Hz，目标能量整个灌进自己的训练区，
    # 门限跟着信号一起涨（扩展目标自遮挡）。CFAR 测的是「谱形尖不尖」，
    # 测不到「能量多不多」——而后者才是 34dB 的强判据。
    # 主判据开关。⚠️ **正常别改**——上面整段注释就是在解释为什么这个才是主力。
    # 2026-08-15 临时关掉（`--no-energy`）只是为了单独 debug CFAR：能量判据一直
    # 命中，会把 frame_hit 焐热，看不出 CFAR 自己在裸跑时的真实表现。
    # debug 完记得改回来 / 别用 --no-energy 跑生产。
    use_energy: bool = True
    energy_fmax_hz: float = 450.0    # 参与能量统计的多普勒上限（避开边缘 bin）
    energy_margin_db: float = 10.0   # 高出基线这么多才算有动静（无人段起伏仅 ±1dB）
    energy_ring_sec: float = 60.0    # 基线统计窗（3000 个 float32 = 12KB）
    energy_pct: float = 10.0         # 基线取环内的低分位数，不取最小值——
                                     # 遮挡会让能量瞬间塌几十 dB，min 会被锁死
    # 棘轮：基线**下降不限速**（新的静默期立刻捕获），**上升限速**。
    # 没有这个限速，人连续待十几秒就会把基线抬上去而丢目标（实测 73.9% vs 95.1%）。
    energy_rise_db_s: float = 0.5
    energy_refresh_steps: int = 10   # 每几步重算一次分位数（基线本来就慢，省算力）

    # ---- CFAR 检测（副判据）----
    # 保留它是为了抓「能量没涨多少但谱上有窄峰」的目标。和能量判据取**或**。
    use_cfar: bool = True
    # 逐 bin 背景图：CFAR 判之前先除掉它，抹平 ±100/±200Hz 那种固定线谱。
    # ⚠️ **默认关**。它确实能压掉线谱（静止场景 CFAR 帧命中 6.8%→1.8%），
    # 但在有真值标注那份数据上把检出从 99.9% 拉到 99.3%、状态多抖 2 次，
    # 而且除以一个自身在抖的 EMA 是徒增方差。用户实测也认为没帮助。
    # 想开就 `--clutter-map`，别改默认。
    use_clutter_map: bool = False
    # EMA 系数：0.005 -> 时间常数 200 步 = 4s。要比人经过的时间尺度慢，
    # 又要能跟上 TX/环境的慢变化。只在判"无人"时更新。
    clutter_alpha: float = 0.005
    cfar_train: int = 16             # 单侧训练单元数
    # 单侧保护单元数。4 -> 8（2026-08-15）：把保护带从 ~15.6Hz 拉宽到 ~31Hz
    # （8 * bin_hz，bin_hz=fs_out/nfft=3.9Hz），给 ±200Hz 那条 Tx 线谱的
    # 裙边多留余量，别让它漏一点能量进训练区把噪声估计搅浑。
    cfar_guard: int = 8
    # 早先 pfa/min_bins 是按统计虚警率标定的（1e-3+2bins -> 1e-4+3bins），
    # 换算下来门限只有 +10.3dB。真机接上 TX 后发现 ±100/±200Hz 有固定线谱，
    # 比本底高 10~14dB，正好压过这条线（见 ClutterMap 类注释）。
    # 2026-08-15 放弃在信号链路上压这条线谱（TDD 占空比/积分时长都试过，无效），
    # 改为直接把 CFAR 门限**改成固定 dB 余量**而不是走 Pfa 反推。
    # 30dB -> 20dB：30dB 太保守（回放验证时把检出率从 99.9%压到 98.1%）。
    # 2026-08-16：20dB 又太保守了——那是按 8/15 那次更强的线谱（10~14dB）标定
    # 的，这次实测（真机 18s 静止录制，用 CfarDetector 自己训练出的本底做基准，
    # 不是全谱中位数）±50/100/200Hz 那几条线现在只比本底高 **中位数3dB、
    # 最强13.5dB**，20dB 门限下 `n_hits`/`max_run` 全程 0，CFAR 名存实亡。
    # 20dB -> 10dB：把 threshold_db 往下扫（6/7/8/9/10/11/12dB）配合下面的
    # `cfar_min_run` 一起看：10dB 是拐点——低于10dB，静止场景里 run>=5 的
    # 帧占比从 0% 迅速爬升（9dB:0.1%，8dB:0.4%，7dB:2.2%，6dB:5.9%）；
    # 10dB 及以上，不管命中落在±50/100/200Hz哪条已知线还是别处，
    # 静止场景全程 run>=5 占比都是 0%。10dB 是"CFAR 从哑火变敏感"和
    # "run>=5 还保持零虚警"两者都满足的下限，配合 min_run 一起用，
    # 别单独再往下调。
    cfar_threshold_db: float = 10.0  # 峰值必须比训练出的本底高这么多 dB 才算命中
    # 2026-08-16：判据从"全谱命中总数"换成"最长连续命中游程"。
    # 旧版 `cfar_min_bins` 数的是**不分位置**的命中 bin 总数——两条互不相邻的
    # 窄线谱（比如 +100Hz 出 2 个 bin、+200Hz 出 3 个 bin）加起来就能凑过阈值，
    # 尽管没有哪一段真的连续够宽。而真人走路的微多普勒是**一整段连续**的
    # （实测能占 13 个相邻 bin），窄线谱本身单独一段顶多 2~4 个（有些甚至
    # 比一个 bin 还窄，物理上只点得亮 1~2 个相邻 bin）。改成"最长连续段够不够
    # 长"直接堵住"多处窄线加起来凑数"这个漏洞，且更贴近"这一段到底宽不宽"
    # 这个物理量。bin_hz≈3.9Hz，5 个 bin ≈ 20Hz，仍留在人体展宽之下。
    cfar_min_run: int = 5
    # M-of-N 去抖：最近 N 帧里有 M 帧命中才判"有人"。用整数位寄存器存，O(1) 内存。
    # ⚠️ 窗重叠 90%，相邻帧强相关，所以这是**平滑去抖**而不是 N 次独立确认。
    # 30/15/4 是拿真值标注扫出来的：检出 99.9% / 虚警 0% / 30秒只翻转 4 次
    # （真实转换就是 3 次）。窗从 0.3s 加长到 0.6s 换来的是：三次遮挡造成的
    # 全谱掉电（各 ~0.35s）不再把状态打翻。代价是检出延迟 0.21s。
    debounce_n: int = 30             # 30 帧 @50fps = 0.6s
    debounce_enter: int = 15         # 进入"有人"所需命中数
    debounce_exit: int = 4           # 低于此值才退回"无人"（迟滞，退出比进入难得多）
    # ---- 功率监控 ----
    power_decim: int = 10            # 算 Rx 功率时的抽取倍数（省算力）
    power_ema_alpha: float = 0.05
    power_low_dbfs: float = -70.0    # 低于此值认为链路异常（天线掉了/TX 停了）
    power_sat_dbfs: float = -3.0     # 高于此值认为削波（增益过高）
    # 2026-08-16：真机测过两次链路强弱剧烈变化（一次走动挡断两基站间同步、
    # 一次是 TX 自己的行为，原因不用深究），Rx 功率能在几十毫秒内跌 20~30dB，
    # 且**不保证会恢复**。一度想在这里（PowerMonitor）拿功率变化去冻结/
    # 保护 EnergyDetector 的基线学习，结果功率跌了不回头时基线被永久冻在
    # 骤降前的高位，判决直接哑火——用 Rx 功率当"是不是遮挡/有没有人"的判据
    # 本身就不可靠，已经放弃这条路，逻辑全删了。
    # 根治办法在 rt_dsp.py：CAF 按两路各自能量归一化成相关系数，链路整体
    # 变强变弱不再改变谱的尺度，PowerMonitor 就只做纯粹的链路健康监控。

    # ---- 派生量 ----
    @property
    def n_step(self) -> int:
        """每个 step 的样本数/通道"""
        return int(round(self.sample_rate * self.step_sec))

    @property
    def n_period(self) -> int:
        """一个 TDD 周期的样本数"""
        return int(round(self.sample_rate * self.tdd_period_sec))

    @property
    def n_on(self) -> int:
        """一个 TDD 周期里 ON 段的样本数"""
        return int(round(self.n_period * self.tdd_duty))

    @property
    def n_frames_per_step(self) -> int:
        """每个 step 包含几个 TDD 帧 = 每 step 产出几个输出点"""
        return int(round(self.step_sec / self.tdd_period_sec))

    @property
    def fs_out(self) -> float:
        """输出流速率 = TDD 帧率"""
        return 1.0 / self.tdd_period_sec

    @property
    def n_ring(self) -> int:
        """环形缓冲长度 = 窗长内的 TDD 帧数"""
        return int(round(self.window_sec * self.fs_out))

    @property
    def bin_hz(self) -> float:
        return self.fs_out / self.nfft

    def describe(self) -> str:
        return (
            f"fs={self.sample_rate/1e6:.1f}MHz  step={self.step_sec*1e3:.0f}ms  "
            f"window={self.window_sec*1e3:.0f}ms\n"
            f"TDD: {self.tdd_period_sec*1e3:.1f}ms 周期 x {self.tdd_duty:.0%} 占空 "
            f"-> 输出率 {self.fs_out:.0f}Hz, {self.n_frames_per_step} 帧/step\n"
            f"每step {self.n_step:,} 样本/通道 -> {self.n_frames_per_step} 个输出点 "
            f"(压缩 {self.n_step//max(self.n_frames_per_step,1):,}x)\n"
            f"ring={self.n_ring}点  FFT={self.nfft}点  "
            f"bin={self.bin_hz:.2f}Hz  跨度=±{self.fs_out/2:.0f}Hz"
        )


def valid_mask(freqs: np.ndarray, cfg: RtConfig) -> np.ndarray:
    """True = 可用 bin：不在 DC 保护带内，也不在已知 TX 杂散陷波口内。

    CFAR 和 EnergyDetector 共用同一个掩码——两边都会被同一组杂散线抬高，
    分开维护迟早会漂开。见 rt_config.py 里 `notch_freqs_hz` 的注释。
    """
    mask = np.abs(freqs) >= cfg.dc_guard_hz
    for f0 in cfg.notch_freqs_hz:
        mask &= np.abs(np.abs(freqs) - f0) >= cfg.notch_guard_hz
    return mask
