#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 2：Rx 功率监控 + CA-CFAR + M-of-N 去抖 -> 「有人 / 无人」。

全部 O(1) 内存，不存任何历史谱：
- 功率：单极点 EMA
- 门限：CFAR 每帧从**空间邻域**（多普勒轴上相邻 bin）现算，天生自适应
- 去抖：最近 N 帧的命中状态压在一个整数的位里

关于卡尔曼：对一个二值判决是过度设计。CFAR 已经在空间上自适应，
再叠一层噪底 EMA 就够，状态只有几个标量。除非有明确证据，否则别上卡尔曼。
"""

import numpy as np

from rt_config import RtConfig, valid_mask
from rt_dsp import to_db


def longest_run(hits: np.ndarray) -> int:
    """最长连续 True 段的长度。O(n)，不额外分配历史状态。

    用来把"这一段命中到底连续多宽"和"全谱一共命中了多少个 bin"分开——
    后者会被互不相邻的几条窄线谱加总凑过阈值，见 PresenceDetector 里
    `cfar_min_run` 的注释。
    """
    if not hits.any():
        return 0
    d = np.diff(np.concatenate(([0], hits.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int((ends - starts).max())


class PowerMonitor:
    """Rx 功率实时监控。用途不是检测目标，而是**发现链路坏了**——
    天线掉了 / TX 停了 / 增益不对 / 削波。这些故障下多普勒结果全是垃圾，
    但不监控就看不出来。

    ⚠️ 2026-08-16 曾经想拿这里的功率去判"链路被挡"、冻结 EnergyDetector 的
    基线学习，真机复测发现 Rx 功率骤降后可能**不会恢复**（用户指出这多半是
    TX 自己的行为，不是"暂时遮挡"），冻结反而让基线永久卡死、判决焐死。
    已经删掉那套逻辑——**根治办法不是在这里堵，是在 rt_dsp.py 里把 CAF
    按两路各自能量做归一化**，让 Doppler 谱本身就不受链路强弱变化影响，
    这样 PowerMonitor 就只用来做它原本该做的事：纯粹的链路健康监控/告警，
    不参与任何检测判决。
    """

    def __init__(self, cfg: RtConfig):
        self.cfg = cfg
        self.dbfs = [None] * cfg.num_channels     # 各通道 EMA 后的 dBFS
        # 本步的**瞬时**值（未平滑）。EMA 是给人看的，瞬时值才看得出遮挡这类
        # 零点几秒的事件——alpha=0.05 的时间常数是 20 步=0.4s，会把它抹平。
        self.inst = [0.0] * cfg.num_channels

    def update(self, raw_i16: np.ndarray) -> None:
        dec = self.cfg.power_decim
        a = self.cfg.power_ema_alpha
        for ch in range(self.cfg.num_channels):
            i = raw_i16[ch, 0::2 * dec].astype(np.int64)
            q = raw_i16[ch, 1::2 * dec].astype(np.int64)
            p = float(np.mean(i * i + q * q)) / (32767.0 ** 2)
            d = float(to_db(p))
            self.inst[ch] = d
            self.dbfs[ch] = d if self.dbfs[ch] is None else (1 - a) * self.dbfs[ch] + a * d

    @property
    def healthy(self) -> bool:
        return all(self.cfg.power_low_dbfs <= d <= self.cfg.power_sat_dbfs
                   for d in self.dbfs if d is not None)

    def status(self) -> str:
        s = " ".join(f"ch{c}={d:6.1f}dBFS" for c, d in enumerate(self.dbfs) if d is not None)
        if not self.dbfs or self.dbfs[0] is None:
            return "功率:--"
        bad = []
        for c, d in enumerate(self.dbfs):
            if d is None:
                continue
            if d < self.cfg.power_low_dbfs:
                bad.append(f"ch{c}过低")
            elif d > self.cfg.power_sat_dbfs:
                bad.append(f"ch{c}削波")
        return s + ("  ⚠️" + ",".join(bad) if bad else "")


class DownlinkPower:
    """20260826 新增。sensing channel（ch1，物理上朝向 Master）在**下行时隙**
    的平均功率，转成 dBFS。

    物理背景（跟这段代码怎么写强相关，不是废话）：两个背靠背角天线，
    reference channel（ch0）朝 Slave，sensing channel（ch1）朝 Master。
    TddSync 只用 ch0 的功率包络判定"上行 TDD 什么时候 ON"（见 rt_sync.py），
    DopplerEngine 的 CAF 也是在这个上行 ON 窗口内积分（见 rt_dsp.py）。
    这里反过来看**上行窗口之外**那一段——对这条 TDD 链路来说那对应 Master
    的下行发射，sensing channel 正对着 Master，这段时间会直接、强地收到
    下行信号，用它的功率变化能不能反映"链路是不是被挡"，是这个类存在的
    目的（跟 PowerMonitor 的"纯链路健康监控"是不同的目的，故意分开）。

    不用 ISAC 预测式的判断思路，也不重新做一遍 TddSync 那套盲同步——
    上行窗口的位置/长度（phase/n_on）TddSync 已经算好了，这里只是换一个
    窗口位置去切同一份数据算功率，是同一类"从 buffer 里抽一段、平方求和"
    的开销，不比 PowerMonitor.update() 贵。

    不维护自己的 carry 缓冲：直接读 DopplerEngine.work（同一份内存的只读
    视图），省内存，也避免两边各自维护一套 carry 逻辑、迟早对不上的风险。
    """

    def __init__(self, cfg: RtConfig):
        self.cfg = cfg
        self.inst = 0.0      # 瞬时 dBFS——遮挡这种零点几秒的事件要看这个，不是 EMA
        self.dbfs = None     # EMA 平滑过的，给人看状态行用
        self.valid = False   # 上行没锁定时，"下行窗口在哪"没意义，这次值不可信

    def update(self, work: np.ndarray, sync) -> None:
        """work: DopplerEngine.work 的返回值，(num_channels, n_work*2) int16。
        sync: 当前的 TddSync 实例（读 .locked/.phase/.n_on）。"""
        cfg = self.cfg
        if not sync.locked:
            self.valid = False
            return

        n_per = cfg.n_period
        n_fr = cfg.n_frames_per_step
        n_on = sync.n_on
        dl_len = n_per - n_on
        if dl_len <= 0:
            # 上行占空 100%（理论上不该发生，真出现了说明 sync 那边有问题），
            # 没有下行窗口可言，不硬凑
            self.valid = False
            return

        # 下行窗口紧跟在上行 ON 窗口后面：[phase+n_on, phase+n_per)。
        # 不用取模——work 里每帧的绝对偏移是 k*n_per + dl_start，这份 buffer
        # 本来就是按连续样本流存的（DopplerEngine._work 的 carry 设计就是
        # 为了让这种"可能跨到下一个周期栅格"的切片不用特殊处理，直接读
        # 连续内存就是对的，见 rt_dsp.py 里 work 属性的说明）。
        dl_start = sync.phase + n_on
        ch1 = work[1].reshape(-1, 2)  # 视图，不拷贝：(n_work_samples, 2) 的 (I,Q)

        total = 0
        for k in range(n_fr):
            s = k * n_per + dl_start
            seg = ch1[s:s + dl_len].astype(np.int64)
            total += int(np.sum(seg * seg))

        mean_power = total / (n_fr * dl_len) / (32767.0 ** 2)
        d = float(to_db(mean_power))
        self.inst = d
        a = cfg.power_ema_alpha
        self.dbfs = d if self.dbfs is None else (1 - a) * self.dbfs + a * d
        self.valid = True

    def status(self) -> str:
        if not self.valid:
            return "下行(Master)=--(上行未锁定)"
        return f"下行(Master)={self.inst:6.1f}dBFS"


class SimpleDownlinkShadowJudge:
    """20260826 新增。task3：一个刻意写得很简单的、纯本地终端可见的二值判决——
    "下行功率比基线低了一截 -> 判遮挡"，不是要跟 Doppler 的 present 判决
    抢答案，是给本地调试用（跑起来在终端上直接看得到"现在判的是遮挡还是
    没遮挡"，不用开着网页盯 shadowJudgment 数组）。

    ⚠️ 真正会被记进论文数据、送到网页的判断，走的是 dash-js/shadowInputs.js
    的 feedPowerShadow()——那边已经有阈值+回滞的判断逻辑了（见两轮之前加的
    那套），这里**不重复实现一套一样的东西跟它打架**：Python 这边只发原始
    dBFS 数值（rt_main.py 通过 push_power() 发出去），真正"要不要标记成
    shadow"的判断只由浏览器那边做一次。这个类纯粹是本地可见性，它的判断
    结果不会被发送、不会被记录，只打印在终端状态行里。

    判法：滑动看最近 baseline_sec 秒里的**最大值**当基线（跟 EnergyDetector
    的思路类似但简化很多——没有百分位数、没有棘轮限速，因为这里的目的是
    "本地一眼看出个大概"，不是精确判据），当前瞬时值比基线低超过
    drop_margin_db 就判 shadow。
    """

    def __init__(self, cfg: RtConfig, baseline_sec: float = 5.0, drop_margin_db: float = 10.0):
        self.cfg = cfg
        self.drop_margin_db = drop_margin_db
        self._ring_len = max(1, int(round(baseline_sec / cfg.step_sec)))
        self._ring = np.full(self._ring_len, np.nan, dtype=np.float64)
        self._i = 0
        self._filled = 0
        self.shadow = False

    def update(self, downlink_power: DownlinkPower) -> None:
        if not downlink_power.valid:
            return
        self._ring[self._i] = downlink_power.inst
        self._i = (self._i + 1) % self._ring_len
        self._filled = min(self._filled + 1, self._ring_len)

        valid_ring = self._ring[:self._filled]
        baseline = float(np.nanmax(valid_ring))
        self.shadow = bool(downlink_power.inst < baseline - self.drop_margin_db)

    def status(self) -> str:
        return f"下行简易判决(本地only)={'遮挡' if self.shadow else '正常'}"


class ClutterMap:
    """逐 bin 的静止背景图（雷达里的 clutter map）。CFAR 前先用它归一化。

    ## 为什么需要它（2026-08-15 真机实测）

    真机在 ±100Hz / ±200Hz 上有**固定线谱**，比局部本底高 10~14dB，
    正好压过 CFAR 的 +10.3dB 门限 -> CFAR 每帧都在这两处报命中 -> 副判据常真。

    来源已查清（`rt_diag.py --spur`）：**ch1 的接收功率本身就有 5ms(200Hz) 和
    10ms(100Hz) 周期的幅度调制**，是 TX 侧的超帧结构。三件事都排除了：
    - 不是 DC 泄漏（DC 到 ±15Hz 就掉到本底了）
    - 不是积分长度（400/500/695/1000µs 上杂散都是 +12~13dB，动都不动）
    - 不是边沿切歪（起点 -100~+200µs 扫过去也不动）
    - 逐帧幅度归一化**会加剧**（100Hz 从 +9.9 涨到 +17.3dB），因为分母自己也在被调制

    ## 做法

    对每个 bin 单独维护一个线性功率的 EMA。CFAR 拿 `power / map` 去判：
    静止线谱在 map 里也一样高，一除就变平；运动目标是**在多普勒轴上游走的**，
    进不了 map，除完反而更突出。

    **只在判"无人"时更新**——不然人在屋里待久了，他自己的多普勒会被学进背景，
    然后就把他抹掉了。
    """

    def __init__(self, cfg: RtConfig, n: int):
        self.cfg = cfg
        self.map = np.zeros(n, dtype=np.float64)
        self._init = False

    def update(self, power: np.ndarray, frozen: bool) -> None:
        if not self._init:
            self.map[:] = power            # 热启动：第一帧直接当背景，别从 0 爬
            self._init = True
            return
        if frozen:
            return
        a = self.cfg.clutter_alpha
        self.map *= (1.0 - a)
        self.map += a * power

    def normalize(self, power: np.ndarray) -> np.ndarray:
        if not self._init:
            return power
        return power / np.maximum(self.map, 1e-30)


class CfarDetector:
    """沿多普勒轴的 CA-CFAR（单元平均），用 cumsum 做到 O(N) 而不是 O(N*M)。

    对每个被测 bin：取两侧各 n_guard 个保护单元之外的 n_train 个训练单元求平均
    作为噪声估计，门限 = alpha * 噪声。保护单元是为了防止目标自身的能量
    漏进训练区把门限抬高（那样就把自己藏起来了）。

    alpha **不是**从统计虚警率反推的（早先版本是 alpha = N*(Pfa^(-1/N)-1)）。
    2026-08-15 改成直接给固定 dB 余量（`cfg.cfar_threshold_db`）：真机上
    ±100/±200Hz 的 Tx 线谱比本底高 10~14dB，任何接近真实的 Pfa 都压不住它，
    而 30dB 的硬余量能干净地把它关在门外，同时仍留在真人目标之下。

    DC 保护带内的 bin **既不参与被测也不参与训练**——否则那个巨大的 DC
    会把整条噪声估计拉高，真目标反而过不了门限。
    """

    def __init__(self, cfg: RtConfig, dc_mask: np.ndarray):
        self.cfg = cfg
        self.dc_mask = dc_mask.copy()          # True = 不在 DC 保护带内
        self.alpha = 10.0 ** (cfg.cfar_threshold_db / 10.0)
        self._g, self._t = cfg.cfar_guard, cfg.cfar_train
        # 只有两侧训练区都完整的 bin 才可判决（边缘的不判，避免门限失真）
        edge = self._g + self._t
        n = len(dc_mask)
        self._valid = np.zeros(n, dtype=bool)
        self._valid[edge:n - edge] = True
        self._valid &= self.dc_mask

    def detect(self, power: np.ndarray):
        """返回 (命中 bin 的布尔数组, 噪声估计, 门限)。power 必须是**线性功率**。"""
        g, t = self._g, self._t
        # DC 带内的能量置零并从计数里排除，避免污染训练区
        p = np.where(self.dc_mask, power, 0.0).astype(np.float64)
        w = np.where(self.dc_mask, 1.0, 0.0)          # 有效单元的权重
        cp = np.concatenate(([0.0], np.cumsum(p)))
        cw = np.concatenate(([0.0], np.cumsum(w)))
        n = len(power)
        idx = np.arange(n)

        def band(lo, hi):
            """求 [lo, hi) 区间的和与有效单元数，越界处 clip。"""
            lo = np.clip(lo, 0, n)
            hi = np.clip(hi, 0, n)
            return cp[hi] - cp[lo], cw[hi] - cw[lo]

        s_lag, n_lag = band(idx - g - t, idx - g)
        s_lead, n_lead = band(idx + g + 1, idx + g + 1 + t)
        cnt = n_lag + n_lead
        noise = np.divide(s_lag + s_lead, cnt, out=np.full(n, np.inf), where=cnt > 0)
        thresh = self.alpha * noise
        hits = self._valid & (power > thresh)
        return hits, noise.astype(np.float32), thresh.astype(np.float32)


class EnergyDetector:
    """宽带非DC能量 vs 自适应基线。**这是主判据。**

    为什么它比 CFAR 强：真机实测有人/无人的非DC平均功率差 **34.3dB**，
    而这个量 CFAR 看不见（见 rt_config 里 energy_* 那段的解释）。

    基线怎么估（三个都是踩出来的）：
    1. **低分位数，不是最小值**。遮挡瞬间能量会塌到基线以下几十 dB，
       min 会被这种离群值永久锁死；p10 天然免疫（遮挡只占 ~1% 的帧）。
    2. **棘轮**：下降不限速，上升限速 rise_db_s。没有限速的话，人在屋里
       连续待十几秒就会把基线抬到目标电平，然后把人丢掉
       （实测检出率 73.9% vs 有棘轮的 95.1%）。
    3. **全帧入环**，不做"只在判无人时才入环"。后者看着更严谨，实际会被
       "录制一开始就有人"这种情况毒化：基线一上来就锁在有人的电平，
       整段全漏（实测踩过）。棘轮已经解决了它想解决的问题。

    ⚠️ 2026-08-16 曾经想在这里加"遮挡期间的帧不入环"（拿 Rx 功率骤降判遮挡），
    真机复测发现 Rx 功率骤降后可能**不会恢复**（多半是 TX 自己的行为，不是
    "暂时遮挡"），冻结基线反而让判决永久卡死，已经删掉。**根治办法挪到了
    rt_dsp.py**：CAF 按两路各自能量归一化成相关系数，链路整体变强变弱
    （不管什么原因）不再改变 `power` 的尺度，这里就不需要再对付它了。

    ⚠️ 同一天又踩到第二个坑：归一化之后 TX 自己那几条杂散线（50/100/200/300/
    400Hz 附近，见 rt_config 里 `notch_freqs_hz` 的注释）在相关系数尺度下
    变得非常强（+22~35dB），`_band` 算的是整个非DC频段的平均功率，这几条线
    直接把平均值抬起来，跟 CFAR 用同一套 `valid_mask()` 陷波口挡掉，别自己
    再单独拿 dc_guard_hz 建 band。
    """

    def __init__(self, cfg: RtConfig, freqs: np.ndarray):
        self.cfg = cfg
        self._band = valid_mask(freqs, cfg) & (np.abs(freqs) <= cfg.energy_fmax_hz)
        n = max(16, int(round(cfg.energy_ring_sec / cfg.step_sec)))
        self._ring = np.full(n, np.nan, dtype=np.float32)
        self._i = 0
        self._k = 0
        self.floor_db = None
        self.e_db = 0.0

    def update(self, power: np.ndarray) -> bool:
        cfg = self.cfg
        e = float(power[self._band].mean())
        self.e_db = float(to_db(e))
        self._ring[self._i] = self.e_db
        self._i = (self._i + 1) % self._ring.size

        # 分位数不必每帧算：基线按设计就是慢变量，每 10 步(0.2s)刷一次足够
        if self.floor_db is None:
            self.floor_db = self.e_db
        elif self._k % cfg.energy_refresh_steps == 0:
            v = self._ring[np.isfinite(self._ring)]
            cand = float(np.percentile(v, cfg.energy_pct))
            if cand < self.floor_db:
                self.floor_db = cand            # 下降不限速
            else:
                rise = cfg.energy_rise_db_s * cfg.step_sec * cfg.energy_refresh_steps
                self.floor_db = min(cand, self.floor_db + rise)
        self._k += 1
        return self.e_db > self.floor_db + cfg.energy_margin_db

    @property
    def margin_db(self) -> float:
        """当前高出基线多少 dB（负数 = 在基线以下，多半是链路被挡）。"""
        return self.e_db - (self.floor_db if self.floor_db is not None else self.e_db)


class PresenceDetector:
    """宽带能量（主）OR CFAR（副） -> M-of-N 去抖 + 迟滞 -> 「有人 / 无人」。

    去抖用一个整数当位寄存器：每帧左移一位塞进新的命中位，
    popcount 就是最近 N 帧的命中数。真 O(1) 内存，没有数组。
    """

    def __init__(self, cfg: RtConfig, dc_mask: np.ndarray, freqs: np.ndarray):
        self.cfg = cfg
        self.cfar = CfarDetector(cfg, dc_mask)
        self.energy = EnergyDetector(cfg, freqs)
        self.clutter = ClutterMap(cfg, len(freqs)) if cfg.use_clutter_map else None
        self._hist = 0
        self._mask = (1 << cfg.debounce_n) - 1
        self.present = False
        self.n_hits = 0            # 本帧过 CFAR 门限的 bin 数（全谱总数，仅供参考/显示）
        self.max_run = 0           # 最长连续命中段长度（bin）——真正用来判决的量
        self.score = 0             # 最近 N 帧的命中帧数
        self.peak_hz = 0.0         # 命中 bin 里最强的那个的多普勒
        self.e_hit = False         # 本帧能量判据是否命中

    def update(self, power: np.ndarray, freqs: np.ndarray) -> bool:
        self.e_hit = self.energy.update(power)
        # CFAR 判在**背景归一化后**的谱上：固定线谱被抹平，游走的目标凸显。
        # 背景只在判"无人"时学，避免把人自己学进去。
        if self.clutter is not None:
            self.clutter.update(power, frozen=self.present)
            p_cfar = self.clutter.normalize(power)
        else:
            p_cfar = power
        hits, _, _ = self.cfar.detect(p_cfar)
        self.n_hits = int(hits.sum())
        self.max_run = longest_run(hits)
        cfar_hit = self.cfg.use_cfar and self.max_run >= self.cfg.cfar_min_run
        # use_energy=False 时主判据被短路掉，只剩 CFAR 裸跑——debug CFAR 用，见 rt_config。
        frame_hit = (self.cfg.use_energy and self.e_hit) or cfar_hit
        if self.n_hits:
            masked = np.where(hits, power, -np.inf)
            self.peak_hz = float(freqs[int(np.argmax(masked))])
        else:
            self.peak_hz = 0.0

        self._hist = ((self._hist << 1) | int(frame_hit)) & self._mask
        self.score = int(self._hist.bit_count())
        # 迟滞：进入和退出用不同门限，避免在边界反复横跳
        if self.present:
            if self.score < self.cfg.debounce_exit:
                self.present = False
        else:
            if self.score >= self.cfg.debounce_enter:
                self.present = True
        return self.present

    def status(self) -> str:
        flag = "有人" if self.present else "无人"
        e_tag = f"E={self.energy.margin_db:+5.1f}dB" + ("" if self.cfg.use_energy else "(屏蔽)")
        return (f"[{flag}] score={self.score:2d}/{self.cfg.debounce_n} "
                f"{e_tag} bins={self.n_hits:2d} run={self.max_run:2d}")
