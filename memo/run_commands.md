# 命令备忘

## ★ 推到 ISAC 网页（有人=shadowing / 没人=normal）—— 现成的，两条命令

完整说明/排障见下面「3. 推到 ISAC 网页」，这里只放最短版本，两个终端：

```bash
# 终端 A —— 起桥接 + web server。注意 python3，不是 /usr/bin/python3
python3 ~/sdr_workspace/RealtimeISAC/isac_bridge.py

# 终端 B —— 起实时检测，把判决推给桥接
cd ~/sdr_workspace/RealtimeISAC/realtime
/usr/bin/python3 rt_main.py --duration 86400 --isac    # 86400s=跑一整天，Ctrl+C随时停
```

网页连 `ws://<本机IP>:8765/isac`（这台机器现在测到的几个 IP，具体用哪个看
网页设备在哪个网段：局域网 `192.168.1.50`、5G核心网 `192.168.70.129`、
无线 `10.2.54.215`）。查状态不用等网页：`curl http://127.0.0.1:8765/isac/state`。

⚠️ **架构方向**：这台机器起的是 **websocket 服务端**，是网页/前端主动连过来
（`ws://本机IP:8765/isac`），不是这台机器去连别的服务器推数据。如果想要的是
"这台机器当客户端往一个远程服务器推"，那是完全不同的另一套，现在没有。

> 两条铁律：① 必须写全 `/usr/bin/python3`（`python3` 是 miniconda，没有 uhd）
> ② 跑 `realtime/` 里的东西必须先 `cd` 进去（模块是平铺 import）

```bash
cd ~/sdr_workspace/RealtimeISAC/realtime
```

## 1. 实时跑（不落盘）

```bash
/usr/bin/python3 rt_main.py --duration 30              # 单行峰值
/usr/bin/python3 rt_main.py --duration 30 --waterfall  # ASCII 瀑布图
```

## 2. 录一段 + 出多普勒图 ← 先跑这个看表现

```bash
/usr/bin/python3 rt_main.py --duration 30 --debug      # 录，存到 debug/
/usr/bin/python3 ../debug/plot_doppler.py              # 画最近一次录的
```

录的时候**分段做动作**，出图时时间轴上一眼能对上，例如：
站着不动 10s → 朝天线走 10s → 走开 10s → 离开房间 10s。
朝天线走应该是**正频率**，走开是**负频率**。

出图有 4 栏，从上到下：

1. **多普勒瀑布** —— 人走路会看到 ±200~400Hz 的对称"拱形"（躯干+四肢微多普勒）
2. **能量判据**（主判据）—— 绿线 E 超过红点线 `基线+10dB` 就算这帧有动静。
   判决为什么在某一刻翻转，看这栏最直接
3. **判决** —— 峰值多普勒（蓝点）+ CFAR 命中数（橙线），红底 = 判"有人"
4. **Rx 功率** —— ch0/ch1 每帧瞬时 dBFS。这栏专门给"谱塌了"归因：
   功率跟着塌 = 被挡住/TX 停了；功率不动而谱塌 = DSP 侧同步丢了

`--debug` 是**唯一**会写文件的开关，不加就一个字节都不落盘。出图脚本常用参数：

```bash
/usr/bin/python3 ../debug/plot_doppler.py --fmax 200   # 只看 ±200Hz
/usr/bin/python3 ../debug/plot_doppler.py --phase      # 多一栏相位
/usr/bin/python3 ../debug/plot_doppler.py --no-show    # 只存 PNG 不弹窗
```

在终端里直接跑会**弹窗**；PNG 同时存在 `debug/` 下同名文件。

## 3. 推到 ISAC 网页（有人=shadowing / 没人=normal）

**要开两个终端**，因为两个解释器依赖互斥（`/usr/bin/python3` 有 uhd 没 fastapi，
miniconda 的 `python3` 反过来），中间用 UDP 连。

**先起哪个都行**，UDP 是发完就走——桥接没起时 rt_main 照跑不误
（实测 `sent=4 dropped=0 errors=0`），桥接晚起最多 2 秒（一个心跳）就同步上。
习惯上还是**先 A 后 B**，这样从第一秒起状态就是对的。

```bash
# 终端 A —— 注意这里用 python3，不是 /usr/bin/python3
python3 ~/sdr_workspace/RealtimeISAC/isac_bridge.py

# 终端 B
cd ~/sdr_workspace/RealtimeISAC/realtime
/usr/bin/python3 rt_main.py --duration 600 --isac     # 600 秒 = 10 分钟
```

`--duration` 的单位是**秒**。想一直跑就给个大数（`--duration 86400`），
Ctrl+C 随时停，停了桥接 6 秒后自动回落 normal。

查状态：`curl http://127.0.0.1:8765/isac/state`，网页连 `ws://<本机IP>:8765/isac`。

- rt_main 每 2 秒重发一次心跳 → 桥接后启动或丢包，最多 2 秒同步回来
- 超过 6 秒收不到判决（rt_main 崩了/Ctrl+C）→ 自动回落 normal，不会挂着假状态
- `isac_bridge.py --dry` 不起 web server，只把收到的判决打出来，联调用
- **`--isac` 和 `--debug` 可以一起加**，边推网页边录数据

## 4. 判决参数回归测试

```bash
/usr/bin/python3 ../debug/replay.py           # 用真值标注给当前参数打分
/usr/bin/python3 ../debug/replay.py --sweep   # 扫参找更优组合
```

改完 `rt_config.py` 里任何判决参数，跑这个看有没有退化。它 import 的就是
`rt_detect` 本身，不存在离线脚本和实时代码漂掉的问题。

## 5. TDD 结构 / 杂散诊断（不落盘）

```bash
/usr/bin/python3 rt_diag.py                  # 抓 100ms，量周期/占空比/ON-OFF 落差
/usr/bin/python3 rt_diag.py --spur           # 抓 1s，查杂散来源 + 量上升沿漂移
```

`--spur` 在**同一份采集**上对比不同积分长度、不同起点、单通道功率、
逐帧归一化，一次把「是不是 DC / 是不是切歪了 / 是不是帧长不够 / 是不是 TX 自己的」
四个假设分开。也会打上升沿的稳定性和 ppm 漂移。

## 6. 不碰硬件自检

```bash
/usr/bin/python3 rt_main.py --dry-run --duration 8 --debug
```

合成数据注入 **120Hz**，输出应读到 `+121.1Hz`（最近 bin）、判决「有人」。
频率轴/门控/CFAR 改坏了，这条命令立刻能看出来。

## 常用开关

`--fs 30`（采样率 MHz，默认 10）　`--gain 40`（默认 30）　`--duration`（秒）　
`--serial 32392D3`　`--publish`（判决 JSON 打 stdout）

对比实验用的：`--n-on-us 400`（强制每帧积分 400µs，默认用实测的 ~695µs）、
`--no-clutter-map`（关掉 CFAR 前的逐bin背景归一化）

## ±100Hz / ±200Hz 那两条线是怎么回事（2026-08-15 查清）

**是 ch1 收到的信号自己的幅度调制**，5ms 周期(200Hz) + 10ms 周期(100Hz)，
TX 侧的超帧结构。`rt_diag.py --spur` 在同一份数据上把四个假设都排除了：

| 假设 | 实测 |
|---|---|
| DC 太大泄漏过去 | ❌ DC 到 ±15Hz 就掉到本底了 |
| 帧长不够 / 切 0.4ms 更好 | ❌ 400/500/695/1000µs 杂散都是 +12~13dB，动都不动 |
| 上升沿切歪了 | ❌ 起点 -100~+200µs 扫过去也不动 |
| 上升沿漂了 | ❌ 1s 内跨度仅 4µs，漂移 4.55ppm，对比度稳定 0.999 |
| 逐帧幅度归一化能除掉 | ❌ **反而加剧**（100Hz +9.9→+17.3dB），分母自己也在被调制 |
| **ch1 单通道功率就有** | ✅ **+12.7dB**（ch0 只有 +2.3dB）|

**采用的解法 = `cfar_min_bins` 从 3 调到 5**（一个数字）。
杂散只顶得起 **2~4 个 bin**，真人走路的微多普勒实测能占到 **13 个**，门限卡 5
正好分开。静止场景虚警 3.0%→**0%**、翻转 2→**0**，代价只有检出 99.9%→98.9%。

### 两个试过但**退回去**的方案（别再拿出来当默认）

**`use_clutter_map`（默认关）** —— 逐 bin 学静止背景、CFAR 在 `power/map` 上判。
确实压得住线谱（CFAR 帧命中 6.8%→1.8%），但检出掉到 99.3% 且状态多抖 2 次，
本质是除以一个自身在抖的 EMA、徒增方差。用户实测也认为没帮助。
想开：`--clutter-map`。它还有个盲区：**匀速不变的目标 ~4s 后会被学进背景而消失**。

**`rx_tracking_cal=False`（已退回 True）** —— 曾想关掉 AD9361 的 DC offset /
IQ 平衡跟踪校准来压线谱，**实测反而制造虚警**。原因：关掉 DC 校正后两路各留
一个未修正的直流，`conj(ch0)*ch1` 里产生 `conj(dc0)*s1(t)` 和 `conj(s0(t))*dc1`
两个**跟着信号走**的交叉项，它们不在 0Hz，DC 的 EMA 扣不掉，直接抬高非DC能量
也就直接抬高主判据。想再试用 `--no-tracking-cal`，但**先用 `rt_diag --spur` A/B**。

## 输出怎么看

- `TDD锁定 对比度=1.00 占空=71.5% 积分=6950样本` → 门控正常（占空是**实测**的）
- `对比度<0.30` → 降级自由运行，TX 或前端没开
- `✅ 无 overflow` + `主动丢步=0` → 采集与计算都跟得上
- `[无人] score=0/30 E=+1.2dB bins=0` → 判决 / 最近 30 帧命中数 / **能量高出基线多少** / CFAR 命中 bin 数
- `ch0=-21.1dBFS ch1=-42.0dBFS` → 两路 Rx 功率（背靠背角天线，差 19dB 正常）

**`E=` 是现在的主判据**：>+10dB 就算有动静。实测无人时它只在 ±1dB 内晃，
有人走动 +31dB，余量极大。CFAR 退成副判据（抓窄峰），两者取或。

## 判决延迟（真值实测，不是估算）

| | 时间 | 来源 |
|---|---|---|
| 检出 | **0.2 ~ 0.5s** | 相干窗填充 0.2s + 去抖需 15/30 帧 |
| 释放 | **~0.9s** | 退出要 score 跌破 4/30，即连续 ~26 帧无命中 |

想更快就把 `debounce_n/enter` 调小，代价是遮挡和落座时状态会来回跳
（`replay.py --sweep` 能直接看到这个折衷）。

## ⚠️ 已知限制

- **落座不动 = 判无人**。多普勒只看运动，人完全静止时非DC能量回到基线，
  物理上就分不出来。实测「落座但偶有小动作」判有人占 52%。
- `bins=5` **不代表 5 个目标**。一个人走路的微多普勒就能占十几个 bin。
  要分人数得对命中 bin 做聚类，现在没做。
