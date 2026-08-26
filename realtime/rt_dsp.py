#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多普勒引擎：把每个 TDD 帧塌缩成一个复数，再对这条 1kHz 的流做短 FFT。

核心思想：TDD 每 1ms 才发一次，所以信息的真实速率就是 1kHz。把每帧积分成一个数
不丢任何信息，FFT 因此从 375,000 点降到 256 点（~1500x）。

关键实现细节（都实测过，别随手改回去）：
1. **全程 int16，不转 float**。einsum 必须 dtype=np.int64 累加——int32 会溢出
   （5000 个乘积每个最大 32767^2≈1.07e9，int32 上限 2.1e9），而且不报错。
   只有每 step 的 20 个输出值才转 float，转换量减少 10^4 倍。
2. **einsum('ij,ij->i') 融合乘法与求和**，不物化中间乘积数组，省一整趟内存往返。
3. **carry 缓冲**保证帧栅格跨 step 边界连续——否则每个 step 会丢掉一帧，
   输出率从 1kHz 漂成 950Hz，多普勒频率就是错的。
"""

import numpy as np

from rt_config import RtConfig, valid_mask


class DopplerEngine:
    def __init__(self, cfg: RtConfig):
        self.cfg = cfg
        n_per, n_step = cfg.n_period, cfg.n_step
        # work = [上个 step 的尾巴(1个TDD周期) | 本 step 的新数据]
        # 有了这段 carry，帧栅格才能跨 step 连续，每步稳定产出 n_frames_per_step 帧。
        self._n_work = n_per + n_step
        self._work = np.zeros((cfg.num_channels, self._n_work * 2), dtype=np.int16)
        self._new = self._work[:, n_per * 2:]        # 新数据写这里（视图）
        self._tail_src = self._work[:, -n_per * 2:]  # 本步末尾，下一步的 carry
        self._carry = self._work[:, :n_per * 2]      # carry 区（视图）

        # 帧收集缓冲：(n_frames, n_integrate, 2)。einsum 在连续内存上明显更快，
        # 所以先 gather 成连续再算，而不是直接对 as_strided 的跨步视图做 einsum。
        self._gather = np.empty(
            (cfg.num_channels, cfg.n_frames_per_step, cfg.n_period, 2), dtype=np.int16)

        self._ring = np.zeros(cfg.n_ring, dtype=np.complex64)
        self._win = np.blackman(cfg.n_ring).astype(np.float32)
        self._fftbuf = np.zeros(cfg.nfft, dtype=np.complex64)
        self._dc = np.complex64(0)
        self._dc_n = 0
        self._primed = 0          # 已灌入的帧数，达到 n_ring 前谱不可信
        # 最近一帧的**复数**谱。实时判决只要 |S|^2，这个引用纯给 --debug 录制用
        # （复数谱才画得出相位面板）。不额外分配：S 本来就要算出来。
        self.last_spectrum = None

        self.freqs = np.fft.fftshift(np.fft.fftfreq(cfg.nfft, d=1.0 / cfg.fs_out))
        # 命名仍叫 dc_mask（外部一堆代码在用这个名字），但现在也把已知 TX
        # 杂散陷波口排除在外了，见 rt_config.valid_mask()。
        self._dc_mask = valid_mask(self.freqs, cfg)

    @property
    def ready(self) -> bool:
        """环形缓冲灌满前，谱里还混着初始零值，不该拿去判决。"""
        return self._primed >= self.cfg.n_ring

    @property
    def work(self) -> np.ndarray:
        """20260826 新增：(num_channels, n_work*2) int16，只读——当前 step 的
        新数据 + 上一步末尾的 carry（跟 ingest()/process() 内部用的是同一份
        内存，没有另外拷贝）。这份"当前 step + 上一步尾巴"的连续视图是下行
        功率提取（rt_detect.py 的 DownlinkPower）也需要的同一个东西——不新开
        一份 carry 缓冲重复维护同一套逻辑，直接复用这份，省内存也省一份
        容易跟这边失步的重复代码。

        **只读**：调用方不该写它——写了会打乱 Doppler 处理自己下一步的
        carry 状态，这条链路调得很细（见文件头"关键实现细节...别随手改回去"），
        任何写入都可能引入一个很难查的间歇性 bug。
        """
        return self._work

    def ingest(self, raw_i16: np.ndarray) -> None:
        """把一个 step 的新数据搬进 work 缓冲（并保留上一步的尾巴做 carry）。"""
        # 先把上一步的尾巴挪到 carry 区，再写入新数据——顺序不能反。
        np.copyto(self._carry, self._tail_src)
        np.copyto(self._new, raw_i16)

    def process(self, phase: int, n_integrate: int) -> np.ndarray:
        """算一帧多普勒谱。phase/n_integrate 来自 TddSync。

        返回**线性功率** |S|^2（float32，长度 nfft，已 fftshift，频率轴见 self.freqs）。
        注意不是 dB——CFAR 的单元平均必须在线性功率上做，在 dB 上平均是错的
        （dB 上的均值等于线性上的几何平均，不是 CFAR 要的算术平均）。
        要显示用 to_db() 转换，只有 256 个点，代价可忽略。
        """
        cfg = self.cfg
        n_fr, n_per = cfg.n_frames_per_step, cfg.n_period
        n_int = min(n_integrate, n_per)
        phase = int(phase) % n_per

        # ---- 收集每帧的积分段（跨步 gather 成连续内存）----
        g = self._gather[:, :, :n_int, :]
        for ch in range(cfg.num_channels):
            src = self._work[ch]
            for k in range(n_fr):
                s = (phase + k * n_per) * 2
                np.copyto(g[ch, k], src[s:s + n_int * 2].reshape(n_int, 2))

        # ---- CAF：conj(ch0)*ch1 逐帧积分，全整数，int64 累加 ----
        ar, ai = g[0, ..., 0], g[0, ..., 1]
        br, bi = g[1, ..., 0], g[1, ..., 1]
        re = (np.einsum('ij,ij->i', ar, br, dtype=np.int64)
              + np.einsum('ij,ij->i', ai, bi, dtype=np.int64))
        im = (np.einsum('ij,ij->i', ar, bi, dtype=np.int64)
              - np.einsum('ij,ij->i', ai, br, dtype=np.int64))

        # ---- 归一化：除以本帧两路各自的幅度（||ch0||·||ch1||），不是固定满量程 ----
        # 2026-08-16：原来除的是固定的 `SC16_SCALE²*n_int`（相当于假设两路都是
        # 满量程）。真机实测过两次 TX 链路强弱变化（一次走动挡断两基站间同步、
        # 一次是 TX 自己的行为，具体原因用户说了不用管），Rx 功率能在几十毫秒内
        # 跌 20~30dB，这个固定归一化会让 dec 的幅度跟着链路强弱直接涨跌——
        # 下游所有基于历史基线学出"多大算有人"的判据都会被这个共模变化骗到
        # （之前想用 Rx 功率去挡这个变化，但功率骤降后可能**不会恢复**，
        # 冻结基线反而把判决焐死，已经证明行不通，见 rt_config.py 的注释）。
        #
        # 现在改成除以 sqrt(Σ|ch0|²)·sqrt(Σ|ch1|²)（本帧内两路各自的能量），
        # 这样 dec 就是**相关系数**（Cauchy-Schwarz 保证 |dec|<=1），只反映
        # 两路"有多同步/相干"，跟两路各自绝对功率大小无关——链路整体变强变弱、
        # TX 功率涨跌，只要两路是**同步跌的**（这里确实是，两路来自同一次
        # TX 同步/衰落事件），分子分母同步跌，比值不受影响。不需要额外历史、
        # 不需要功率监控介入，在最靠近物理量的地方把共模增益直接除掉。
        ea = (np.einsum('ij,ij->i', ar, ar, dtype=np.int64)
              + np.einsum('ij,ij->i', ai, ai, dtype=np.int64))
        eb = (np.einsum('ij,ij->i', br, br, dtype=np.int64)
              + np.einsum('ij,ij->i', bi, bi, dtype=np.int64))
        norm = np.sqrt(ea.astype(np.float64)) * np.sqrt(eb.astype(np.float64))
        re_f, im_f = re.astype(np.float64), im.astype(np.float64)
        dec_re = np.divide(re_f, norm, out=np.zeros_like(re_f), where=norm > 0)
        dec_im = np.divide(im_f, norm, out=np.zeros_like(im_f), where=norm > 0)
        dec = (dec_re + 1j * dec_im).astype(np.complex64)

        # ---- DC 抑制：EMA 跟踪均值后相减，O(1) 内存，不存历史 ----
        # ⚠️ 开头这几十步必须用**精确滑动均值**，不能直接上 alpha=0.01 的 EMA。
        # alpha=0.01 的时间常数是 100 步 = 2 秒，而 ring 只要 10 步(0.2s) 就灌满
        # 并报 ready。早先只把第一帧热启动成实测均值仍然不够——那个估计只来自
        # 20 帧，很毛糙，之后要 2 秒才refine。实测后果：开机后 t=1.1~1.6s 残余 DC
        # 泄漏到非DC bin，能量被抬高 12dB，**主判据自己触发，一启动就误报"有人"**。
        #
        # 修法是标准的 EMA 预热：alpha 取 max(设定值, 1/n)。n=1 时 alpha=1（直接
        # 取当前值），之后是精确的累计平均，等 1/n 掉到设定值以下（n=100）再切成
        # 定常 EMA。开头无偏且立刻收敛，稳态行为完全不变。
        #
        # ⚠️ 严格历史序：先用**上一步留下的** self._dc 去减本帧，再拿本帧的均值
        # 刷新 self._dc 给下一帧用。旧写法是反过来（先把本帧均值混进估计，再用
        # 混合后的估计减本帧），本帧会看到掺了自己一点点的估计，不是纯历史平均。
        m = dec.mean()
        dec -= self._dc
        self._dc_n += 1
        a = max(cfg.dc_ema_alpha, 1.0 / self._dc_n)
        self._dc = np.complex64((1.0 - a) * self._dc + a * m)

        # ---- 滑窗：撇掉最旧的 n_fr 个，接上新的 ----
        self._ring[:-n_fr] = self._ring[n_fr:]
        self._ring[-n_fr:] = dec
        self._primed = min(self._primed + n_fr, cfg.n_ring)

        # ---- 加窗 + 补零 + FFT ----
        self._fftbuf[:cfg.n_ring] = self._ring
        self._fftbuf[:cfg.n_ring] *= self._win
        self._fftbuf[cfg.n_ring:] = 0
        S = np.fft.fftshift(np.fft.fft(self._fftbuf))
        self.last_spectrum = S
        return (S.real * S.real + S.imag * S.imag).astype(np.float32)

    def peak(self, power: np.ndarray):
        """返回 (峰值多普勒 Hz, 峰值 dB)，已排除 DC 保护带。"""
        masked = np.where(self._dc_mask, power, -np.inf)
        i = int(np.argmax(masked))
        return float(self.freqs[i]), float(to_db(power[i]))

    @property
    def dc_mask(self) -> np.ndarray:
        """True = 不在 DC 保护带内（可用于检测/找峰值的 bin）。"""
        return self._dc_mask


def to_db(power):
    """线性功率 -> dB。只在显示/打印时用，不要在 CFAR 里用。"""
    return 10.0 * np.log10(np.asarray(power, dtype=np.float64) + 1e-20)
