# ±50·k Hz 多普勒"梳状线"的真正成因：逐 step 块状去 DC（2026-09-02）

> **给工作电脑上接手的 Claude**：这台是实验/分析机，不是主开发机。
> 这次会话在 `experiment_30MHz_static_20260710_155243`（另一台 B210 serial=321D889、
> 2026-07-10 采、30MHz、20s、有一个人在 ~13.75m 处走动）上把实时链路跑了一遍，
> 定位到多普勒谱上 ±50/100/150/200… Hz 那一整排线谱**不是 TX / 硬件 / 50Hz 市电**，
> 而是 `rt_dsp.DopplerEngine.process()` 里那段**每 step 更新一次的 EMA 去 DC**造出来的。
> 本机的另一个 terminal 已经把那段删了（见"已做的改动"），需要你在主开发机上
> `git pull` 后 **复核 + 跑回归 + 决定要不要连带回退之前为这条梳做的对抗性改动**。

---

## 1. 现象

`debug/plot_doppler.py` 出的多普勒谱（`doppler_20260816_160533` 等，也包括本次
155243 的重放）上，除了 0Hz 的 DC 线，还有一整排等间距横线：±50、±100、±150、
±200… 一直到 ±450Hz。之前多次会话（见 `worklog.md` 2026-08-15 / 08-16）把它当成
"TX 侧超帧结构 / ch1 收到的信号自带的幅度调制"，做了一堆对抗：

- `notch_freqs_hz=(50,100,150,200,300,400)` + `notch_guard_hz=20`（CFAR/Energy 共用陷波口）
- `cfar_min_bins` 3→5（后来又改成 `cfar_min_run`）
- `use_clutter_map`（逐 bin 背景归一化，试过又退回默认关）
- `rx_tracking_cal=False`（试过，更糟，退回 True）
- 改积分长度 / 边沿 guard —— 都无效

## 2. 根因：每 step（50Hz）做一次的"块状去 DC"

`process()` 的数据流（**这部分本身没问题**）：

- 每个 TDD 帧（1ms）→ 一个复数 `H[k] = Σ_ON conj(ch0)·ch1 / (‖ch0‖·‖ch1‖)`（零延迟单抽头信道估计 / 相关系数）
- 一个 step（`step_sec=0.02s`）= 20 帧 → 20 个 `H`
- 这 20 个进 `self._ring`（200 长 = 最近 0.2s）
- 整个 ring 一起 Blackman 加窗 + 补零到 256 + 一次 FFT → 一张多普勒谱

**出问题的是夹在中间的去 DC（已删除的那 5 行）**：

```python
m = dec.mean()                      # 这一步 20 个 H 的均值
dec -= self._dc                     # 20 个 H 全部减同一个标量
self._dc_n += 1
a = max(cfg.dc_ema_alpha, 1.0/self._dc_n)
self._dc = (1-a)*self._dc + a*m     # EMA，每 step 才更新一次
```

- `self._dc` 是一个**标量 EMA**，**每 0.02s 才更新一次（= 50Hz）**，且相减时用的是上一步的值。
- 这一步的 20 个 `H` 减的是**同一个数**；下一步减另一个数。
- ring 里 10 个 step 的段各自减了不同标量 → 段与段之间有小台阶 → 这个**周期 20 帧（0.02s）的零阶保持台阶**，FFT 出来就是 step 率 `1/step_sec` 及其所有谐波的梳。

## 3. 证据（都可复现）

### 消融表 —— `debug/ablate_doppler.py`（155243, 12s, 数字=相对本底 dB）

| 配置 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|
| baseline（改动前 rt_dsp） | +8.8 | +10.7 | +8.5 | +7.0 |
| 拿掉逐帧归一化（÷n_int） | +7.6 | +5.3 | +4.2 | +3.7 |
| 拿掉 TDD 门控（积分整个 1ms） | +8.9 | +10.8 | +8.6 | +7.1 |
| **拿掉去 DC** | **+0.4** | **+0.0** | **+0.9** | **+0.8** |
| **去 DC 改成逐帧(1kHz)更新** | **+0.3** | **+0.0** | **+0.9** | **+0.8** |
| **去 DC 改成扣 ring 均值** | **+0.2** | **+0.0** | **+0.9** | **+0.8** |
| 窗换矩形窗 | +4.5 | +3.3 | +2.8 | +2.2 |

→ 只要不是"逐 step 块状去 DC"，梳全部归零。TDD 门控无关。逐帧归一化只是**放大器**
（它除以每 step 都在抖的 `‖ch0‖·‖ch1‖`，让 `dec.mean` step 间抖更多 → 台阶更大），不是根因。

### 梳间距 = `1/step_sec`（决定性）—— `debug/comb_tracks_step_rate_155243.png`

| step_sec | step 率 | 梳出现在 |
|---|---|---|
| 0.02s | 50Hz | 50, 100, 150, 200… 的倍数 |
| 0.025s | 40Hz | 40, 80, 120… 的倍数（50/100/150 变成 0） |
| 0.01s | 100Hz | 100, 200, 300… 的倍数 |

跟 50Hz 市电无关，跟 TX 无关。

### 老 `analyze_caf.py` 没有这条梳 —— `debug/compare_oldcaf_vs_realtime_155243.png`

扫了缓存矩阵所有 delay（0/1/2.75/4.25/8/15），±50·k 全是 +0~1dB。原因：
- `compute_caf_matrix`：**完全没有去 DC**
- `fast_caf_spectrogram`：`scipy.signal.spectrogram` 的 `detrend='constant'` —— 对**每个 0.2s FFT 段**扣一次均值（平滑滑动），不是 50Hz 台阶

老代码的窗跟实时链路一样是 0.2s Blackman 滑动 STFT，**窗不是差别所在**。

### 真实的 TX/市电纹波确实存在，但不产生这条梳

直接量 155243 原始 IQ 的 `|IQ|²` 包络（零处理）：ch0 在 100Hz +12dB、200/300/400Hz +8~10dB
（100Hz = 2×50Hz 东日本市电，像 TX 电源全波整流纹波）。但它是**共模**的：在
`conj(ch0)·ch1`（连不归一化的原始乘积、`dc=none`）里都不产生多普勒梳（100/200/300 全 +0.5）。
相关系数归一化会把共模项进一步约掉。**所以它对最终多普勒谱的贡献可忽略。**

## 4. 已做的改动（本机另一 terminal，待复核）

- `realtime/rt_dsp.py`：删掉 `process()` 里的 EMA 去 DC（`m=... / dec-=self._dc / self._dc=...`）
  和 `__init__` 里的 `self._dc / self._dc_n`。`dec` 原样进 ring。留了注释说明原因。
- `realtime/rt_config.py`：`dc_ema_alpha` 保留（现在只有 `debug/ablate_doppler.py` 的对比实验用），
  加了注释说明主链路不再引用。

## 5. 为什么当初加了去 DC（从 worklog 还原）

2026-08-15：窗 / guard 没调好时，EMA **预热的头 ~1.8s** 有残余 DC 泄漏到非 DC bin，
能量判据被抬高 12dB，**一启动就误报"有人"**（静止场景帧命中 12/31）。当时两条一起上：
① EMA 热启动（`alpha=max(设定值, 1/n)`）② `dc_guard_hz` 从 6Hz 加到 20Hz。
**真正解决问题的是 ②**。① 那条 EMA 留下来当双保险，没人意识到"逐 step 块状相减"会盖一条 step 率的梳。

## 6. 为什么删掉是安全的

- `dc_guard_hz=20`：CFAR / EnergyDetector / 找峰值 全走 `valid_mask`，±20Hz 直接屏蔽。
- Blackman 窗主瓣 ±15Hz，第一旁瓣 −58dB，到 ±50Hz 已 −85dB 以下。静态杂波 DC 就算比噪声
  高 50dB，泄漏到 50Hz 也在噪声下 −35dB，可忽略。
- 人体回波不会正好落在 `k/step_sec` 这些精确频点上，删梳不影响真实检测。

## 7. 工作电脑上接手要做的

1. **`git pull`，复核 `rt_dsp.py` / `rt_config.py` 的改动**（`self._dc` 相关是否清干净，
   `dc_removal_compare.py` 这类老 debug 脚本是否还引用）。
2. **跑回归**：`/usr/bin/python3 debug/replay.py`（用真值标注给判决参数打分）——
   检出率 / 虚警率 / 翻转次数不应退化。注意 `replay.py` 重放的是**已算好的多普勒谱**，
   要真正验证去 DC 改动得重新采一段或用 `debug/replay_raw_iq.py` 在原始 IQ 上跑。
3. **重新评估为这条梳做的对抗性改动**，很可能可以放宽 / 回退：
   - `notch_freqs_hz` —— 梳没了之后 50/150/250/350/450 这些"奇次"口大概率不需要了；
     100/200/300 处还有一点真实市电纹波残留（弱），看实测再定。
   - `cfar_min_run=5` —— 当初是为了把梳的 2~4 bin 挡在门外，梳没了可以重新扫（3? 4?）。
   - `cfar_threshold_db=10` —— 同上，之前 10dB 是踩着梳标的。
   - `use_clutter_map` —— 更没必要了。
4. **确认真实 TX 纹波要不要单独处理**：本次结论是共模、在相关系数里约掉了，
   多普勒谱上 <1dB。如果换个场景（两路强弱差很大、或纹波变差分）又冒出来，
   再考虑，但**不要再往 `process()` 里加时域去 DC**。

## 8. 复现命令

```bash
cd RealtimeISAC/realtime
# 原始 IQ -> 实时 DSP，出多普勒图 + ±50·k 梳强度读数
python ../debug/replay_raw_iq.py <experiment文件夹或.bin>
python ../debug/replay_raw_iq.py --synthetic --fs 30 --syn-tone 137   # 合成对照(无周期调幅)
# 逐项消融
python ../debug/ablate_doppler.py <experiment文件夹> --sec 12
# 梳间距 vs step 率
python ../debug/comb_rootcause.py <experiment文件夹> --sec 18
```

相关文件：`debug/replay_raw_iq.py`、`debug/ablate_doppler.py`、`debug/comb_rootcause.py`、
`debug/compare_doppler_combs.py`、图 `debug/comb_tracks_step_rate_155243.png` /
`debug/ablation_dc_comb_155243.png` / `debug/compare_oldcaf_vs_realtime_155243.png`。
