#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TDD 突发盲同步。

接收端**无法和 TX 同频/同步**，只能从数据里被动找 ON 段。

做法：参考通道功率包络按 TDD 周期折叠平均（把 n_frames 个周期叠起来），
噪声被平均掉、ON/OFF 台阶凸显出来。折叠前先抽取 sync_decim 倍，
边沿精度 = sync_decim/fs（10MHz 下 = 1µs，占 ON 段的 0.14%，可忽略）。

## ⚠️ 占空比必须实测，不能预设（2026-08-15 上机踩到）

早先版本按 cfg.tdd_duty=50% 开一个固定宽度的窗滑动找最大能量。真机实测发现
**真实占空是 71.5%（ON 715µs / OFF 285µs）**，不是 50%。后果不是"锁不上"，
而是更阴险的"看起来锁上了"：500µs 的窗整个落进 715µs 的突发里，窗外还剩 200µs
的 ON，于是 off_mean≈0.4·on_mean，contrast 卡在 (1-0.4)/(1+0.4)=0.43
——高于 0.30 的锁定阈值，于是报"锁定 contrast=0.42"，实际白扔了 30% 的相干积累。

所以现在**不预设任何占空比**：阈值取折叠包络 p10/p90 的几何中点（跨数量级比
算术中点稳），然后找**环形最长连续超阈段**。ON/OFF 实测有 36dB 落差，
这个判据毫无悬念。窗宽由数据决定，contrast 随之变成真正的"有没有结构"指标
（实测：有 TDD=1.00，纯噪声<0.1）。

没有 TX / 没有 TDD 结构时（比如毫米波前端没开），折叠出来的包络是平的，
此时降级为**自由运行**：仍按 TDD 周期切帧，但整个周期都参与积分，不做门控。
这样链路照常跑通，只是拿不到门控带来的 SNR 增益。
"""

import numpy as np

from rt_config import RtConfig


def _longest_run(on: np.ndarray):
    """环形最长连续 True 段 -> (起点, 长度)。全 True / 全 False 返回 (0, 0)。

    向量化：把序列接一倍再找 False->True / True->False 的跳变位置，
    段长就是相邻跳变之差。比 Python 逐点扫快两个数量级。
    """
    n = on.size
    if on.all() or not on.any():
        return 0, 0
    d = np.concatenate([on, on])
    edges = np.flatnonzero(np.diff(np.concatenate(([False], d, [False]))))
    starts, ends = edges[0::2], edges[1::2]
    k = int(np.argmax(ends - starts))
    return int(starts[k] % n), int(min(ends[k] - starts[k], n))


class TddSync:
    def __init__(self, cfg: RtConfig):
        self.cfg = cfg
        self.phase = 0            # ON 段起点在一个 TDD 周期内的样本偏移
        self.locked = False       # 是否检测到可信的 TDD 结构
        self.contrast = 0.0       # ON 段内外的功率对比度，用来判断有没有结构
        self.duty = 0.0           # 实测占空比（不是配置里的 tdd_duty）
        self._n_on = 0            # 实测 ON 段长度（样本），锁定时用它做积分长度
        self._counter = 0

    def update(self, raw_i16: np.ndarray, force: bool = False) -> bool:
        """raw_i16: (2, n_step*2)。返回本次是否真的重新估计了相位。

        不需要每个 step 都做——时钟漂移很慢，每 sync_every_n_steps 跟一次即可。
        """
        if not force and self._counter % self.cfg.sync_every_n_steps != 0:
            self._counter += 1
            return False
        self._counter += 1

        cfg = self.cfg
        dec = cfg.sync_decim
        n_per_dec = cfg.n_period // dec
        n_fr = cfg.n_frames_per_step
        if n_per_dec < 4 or n_fr < 1:
            return False

        # 参考通道(ch0)的 I 分量做功率代理：raw 里每个复数占 2 个 int16，
        # [0::2*dec] 就是每 dec 个样点取一个 I。用 int32 防平方溢出。
        i_comp = raw_i16[0, 0::2 * dec].astype(np.int32)
        need = n_fr * n_per_dec
        if i_comp.size < need:
            return False
        power = i_comp[:need]
        power = power * power
        # 折叠：把 n_fr 个 TDD 周期叠起来平均，噪声 ~1/sqrt(n_fr) 地被压掉
        profile = power.reshape(n_fr, n_per_dec).mean(axis=0)

        # 找 ON 段：阈值 + **环形最长连续段**，窗宽由数据定，不预设占空比。
        # 阈值取 p10/p90 的**几何**中点：ON 与 OFF 差 36dB，算术中点会紧贴 ON
        # 从而把边沿切掉，几何中点落在两者中间，对落差大小不敏感。
        lo = float(np.percentile(profile, 10))
        hi = float(np.percentile(profile, 90))
        thr = np.sqrt(max(lo, 1e-30) * max(hi, 1e-30))
        start, length = _longest_run(profile >= thr)

        # 长度落在 [5%, 95%] 之外都说明包络是平的（纯噪声会给出一堆碎段，
        # 全 ON 说明没有 OFF 段可言）-> 没结构
        if not (0.05 * n_per_dec <= length <= 0.95 * n_per_dec):
            self._fallback()
            return True

        # 对比度：**ON 段内 vs 段外的平均功率之比**，而不是折叠包络的极值张开。
        # 极值法测的是噪声方差不是结构——纯噪声下 1000 个点、CV≈0.2，
        # (max-min)/max 就能到 0.78，会把没开 TX 的情况误判成锁定（实测踩过）。
        idx = (start + np.arange(length)) % n_per_dec
        on_sum = float(profile[idx].sum())
        on_mean = on_sum / length
        off_mean = (float(profile.sum()) - on_sum) / max(n_per_dec - length, 1)
        denom = on_mean + off_mean
        self.contrast = (on_mean - off_mean) / denom if denom > 0 else 0.0

        if self.contrast < cfg.sync_min_contrast:
            # 没有 TDD 结构（没开 TX，或增益/频率不对）-> 自由运行，不做门控
            self._fallback()
            return True

        # 两端各内缩一点，避开上升/下降沿的滤波暂态（实测下降沿有 ~80µs 的拖尾，
        # 比 ON 低 33dB 但比噪底高 7dB，积进去只会引入调幅）。代价 <3% 能量。
        guard = int(round(cfg.sync_edge_guard_sec * cfg.sample_rate))
        n_on = max(1, length * dec - 2 * guard)
        self.phase = (start * dec + guard) % cfg.n_period
        self._n_on = min(n_on, cfg.n_period)
        self.duty = length / n_per_dec
        self.locked = True
        return True

    def _fallback(self) -> None:
        self.locked = False
        self.phase = 0
        self._n_on = 0
        self.duty = 0.0

    @property
    def n_on(self) -> int:
        """20260826 新增：实测的上行 ON 段长度（样本数），不受 force_n_on_us
        影响——跟 n_integrate 不是一回事：n_integrate 是"Doppler 引擎这一步
        实际积分多少样本"，force_n_on_us 非 0 时会被强制覆盖成对比实验要的值；
        这里要的是"上行 ON 窗口真实在哪、多长"这个物理事实，下行功率提取
        （rt_detect.py 的 DownlinkPower）要靠它算"上行窗口之外"是哪一段，
        用被强制覆盖过的值会算错窗口位置。
        """
        return self._n_on

    @property
    def n_integrate(self) -> int:
        """每帧参与积分的样本数：锁定时只积**实测的** ON 段，未锁定时积满整个周期。

        `force_n_on_us` 非 0 时用它覆盖（对比实验用，起点仍是实测的 ON 段起点）。
        """
        if self.cfg.force_n_on_us > 0:
            n = int(round(self.cfg.force_n_on_us * 1e-6 * self.cfg.sample_rate))
            return max(1, min(n, self.cfg.n_period))
        return self._n_on if self.locked else self.cfg.n_period

    def status(self) -> str:
        if not self.locked:
            return f"TDD未锁定(自由运行) 对比度={self.contrast:.2f}"
        # 打**生效**的积分长度，不是实测的 ON 段长度——被 force_n_on_us 覆盖时
        # 两者不一样，打错的那个会让人以为开关没生效
        n = self.n_integrate
        forced = "(强制)" if self.cfg.force_n_on_us > 0 else ""
        return (f"TDD锁定 phase={self.phase} 对比度={self.contrast:.2f} "
                f"占空={self.duty:.1%} 积分={n}样本"
                f"({n/self.cfg.sample_rate*1e6:.0f}µs){forced}")
