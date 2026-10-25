# 交接提示词：你是 DASH 的 Windows "server" 端 Claude（2026-10-21，写给另一台电脑上的 Claude）

> 先通读本文件再动手。用户(石琦)不在电脑前（去吃饭了），所以**你要自己把第 4 节的任务做完并按第 7 节汇报**，不要中途停下来问"要不要继续"。
> 只有第 5 节列出的"必须停下问"情形才可以问。

## 0. 先读这个：项目最终目的与整个实验的逻辑（决定你"该做什么、不该做什么"）

> 来源：用户的项目记录与我(Ubuntu 侧 Claude)的工作记录。其中标注 **[未证实]** 的是还没有数据支持的假设，不要当结论写进任何东西。

### 0.1 最终目的
证明一件事：**毫米波(25 GHz)链路被人体遮挡会断连约 10 s；如果感知端能在遮挡发生前几秒预警，视频播放器就可以提前补缓冲，从而减少/避免卡顿。**
即论文的应用故事 = "ISAC 预警 → 自适应码率/缓冲策略 → QoE 提升"。你这台 server 是这个故事里的**播放器与视频源**，不是感知端。

### 0.2 实验链路（从物理到指标）
1. **物理链路**：NTG-2501（25 GHz，24.77–25.23 GHz），Master 192.168.1.201 ↔ Slave 192.168.1.101，间距 D = 6 m，波长 λ ≈ 12 mm。人走过链路中间(LOS)会遮挡 → 链路断连，**断连时长被量化到约 9.9 s**（114 次中位 9.92 s，另有 19.9 / 29.9 s 的整数倍；此前 08-27 实验恢复时长 ≈10.6 s）。
2. **感知端**（Ubuntu 上的 USRP B210，被动）：ch0 为参考、ch1 为监测，每个 1 ms TDD 帧算一个复数 H = Σ conj(ch0)·ch1（归一化为相关系数），对 H 序列做 0.2 s 加窗 FFT 得**多普勒谱**。USRP 的 1.89 GHz 只是下变频后的中频，**感知载波就是 25 GHz**，所以感知与链路的菲涅尔区是同一个。
3. **判决（论文要求：纯多普勒，不依赖功率）**：看多普勒谱**围绕 0 Hz 的脊宽**（DC 脊宽，>=90 Hz 命中，证据累积）→ 输出 `normal` / `doppler_detected`。旧的 Energy+CFAR 判据已弃用（`--legacy-judge` 才用）。判据**不能**混入 Rx 功率——这是论文的核心卖点。
4. **接口**：`isac_bridge` → WebSocket → 页面。页面里有"ISAC interface ON/OFF"开关：
   - **OFF**（基线）：播放器完全不听 ISAC，按普通 ABR 跑；
   - **ON**：收到 `doppler_detected` → 强制最低码率 + 把缓冲目标从 4 s 放大到 30 s（尽量在断连前囤满）；恢复 `normal` 后逐步升档。
5. **视频**：为了让"预警才有意义"，缓冲必须**囤不满**，所以做了 tos 4K 阶梯，让码率逼近链路容量（NTG 吞吐中位 ≈12、上限 ≈24.5 Mbit/s；bunny 最高才 4 Mbit/s，缓冲一下就满，预警赢不了 always-30s）。
6. **度量**：卡顿次数/时长、缓冲曲线、码率轨迹、断连前缓冲量；**ON vs OFF** 外加 **always-10 s / always-30 s** 缓冲基线（用户明确要求基线必须包含这两条）。数据由页面的 session 自动保存 + "Save Full Snapshot" 导出（JSON，格式见 `snapshots_FORMAT.md`），Ubuntu 侧用 `analysis/` 脚本对齐 isac_*.npz / NTG csv 分析。

### 0.3 设计链（论文当前骨架，**[未证实]**）
阈值 → 检测距离 R → **提前量 L ≈ (R − r_F)/v** → 能补多少缓冲 → 虚警代价。
补缓冲量 `F = max(0, T_out + m − B + L)`（提前量内播放器也在消耗缓冲；T_out≈9.9 s 起，可到 20 s）。
"1.4 s 提前量"的含义是**人在约 2 m 内**，不是"1.4 s 后必然遮挡"。P(预警后断连) = P(在范围内)×P(穿越|在范围内)×P(断连|穿越)，三项要分开测。

### 0.4 已有证据与诚实的缺口（别替用户夸大）
- ✅ 20:55 人体穿越：32 次断连，多普勒在 30/32 次里**提前约 1.4 s**发现；**Rx 功率几乎没有预警**。
- ⚠️ 20:31 连续扫板（金属板）：多普勒 **0/12** 提前——所以"多普勒一定更早"**不成立**，论文已转向"多特征/可调提前量"的叙述。
- ⚠️ 缺口：单人、单场景；阈值在同一批数据上选；**没有测虚警率 Pfa**；ON/OFF 真机对比**只有 1 次**；周六(10-17)的距离分区实验**没做**。
- 目标期刊现实档位：IEEE Sensors J / TIM / Access；毕业 2027-02。

### 0.5 因此，你的工作永远要服务于"可信、可复现的实验"
- **不要**为了"看起来更好"改判据、改阈值、改 ABR 规则、改缓冲目标 4/30 s、改 ISAC ON 的行为。基线与被测方案必须保持原状，任何改动都会让论文的对比失效。
- **不要**在任何报告里写"证明了…/显著提升…"。只写你**实际观测到**的数字，并说明测试条件（本机回环 / WiFi / 是否限速 / 是否有 NTG）。
- **不要**把家里 WiFi/回环的测速当成 NTG 的结果；它们只用来验证"流程与实时性"，不是论文数据。
- 你这台 server 的正确性标准只有三条：**视频能完整、稳定地被取到**；**页面逻辑与 Ubuntu 上一致**（sha256 对得上）；**不引入额外延迟/抖动**（server 不做限速、不改文件、不缓存失效）。
- 遇到"这个要不要顺手优化一下"的念头：**不要**，写进汇报的"建议"一栏，由用户决定。

## 1. 一句话背景

博士课题：**用 ISAC（USRP B210 做被动多普勒感知）预警毫米波链路（NTG-2501，25 GHz）被人体遮挡，让 DASH 视频播放器提前补缓冲，减少卡顿**。
链路被遮挡会断连，断连时长被"量化"到约 9.9 s（114 次中位 9.92 s，倍数 19.9/29.9 s）；多普勒平均能提前约 1.4 s 发现穿越，Rx 功率本身没有预警。
这台电脑是"老电脑"（两周前开发用的），现在的角色是 **DASH 视频 server（node server7.js，端口 8080）**。

## 2. 整体架构（谁跑在哪台机器上，别搞混）

| 机器 | 角色 | 跑什么 |
|---|---|---|
| **你这台 Windows（192.168.0.100，有线接路由器）** | 视频 server | `node server7.js`（8080）：静态页面 + bunny/tos 视频 + NTG 轮询；**不跑** USRP/rt_main/isac_bridge |
| **用户的 Ubuntu 笔电（WiFi 192.168.0.118；网线口 192.168.1.50）** | 感知端 | `isac_bridge.py`（起 isac_server，WebSocket `:8765/isac`）+ `rt_main.py`（驱动 USRP B210，实时多普勒）。**USRP 在这台，不在你这儿** |
| 浏览器 | 播放器 | 打开 `http://<Windows IP>:8080/?mpd=tos`；页面再 WebSocket 连 Ubuntu 的 isac_server |

- 页面里 ISAC 的 WebSocket 默认地址是 `ws://192.168.1.50:8765/isac`（实验室网线地址）。**在家走 WiFi 时必须用 URL 参数覆盖**：
  `?mpd=tos&isacws=ws://192.168.0.118:8765/isac`（浏览器就在 Ubuntu 上时可用 `ws://127.0.0.1:8765/isac`）。
- IQ / H 数据**全部本机落盘**（Ubuntu 的 `RealtimeISAC/h_logs/`），不经过网络，也不进你这台机器。
- NTG-2501 在学校，家里没有：家里一律 `NTG_ENABLE=0`（server7.js 默认会去轮询 192.168.1.201）。

## 3. 你这台机器的版本与我这边（Ubuntu）的差异

你那边的 ISAC/DASH 大约停在 **10 月 8 日前后**。此后 Ubuntu 侧发生的事（你需要知道，但**大部分不需要你改**）：

**DASH 页面/播放器（需要你确认已安装）**
1. 10-03 / 10-09 / 10-09b / 10-09c 更新：ISAC 接口开关、多普勒瀑布图、NTG 三张图(graph8/9/10)+graph11、`server7.js` 的 NTG 轮询(`/api/ntg`)、`isacClock.js`、`snapshotExport.js`、分析脚本。详见 `UPDATE_20261009_README.txt`、`snapshots_FORMAT.md`。
2. **10-16 新增 tos 4K 阶梯**：`tos_2s_1080p/` 22 档（1080p×14 + 1440p×4 + 4K×4，100 kbit–20 Mbit/s，每档 367 个 2 s 分片，9.2 GiB）+ `tos_ibmff_mixed_1080p*.mpd`。`?mpd=tos` 启用，不加仍是 bunny。
   播放器改了 `DASHttp.js`/`adaptationlogic.js`/`dash.js`（`DASH_MULTIRES`：MPD 含多种分辨率时每片前拼自己档位的 init，允许跨分辨率切档；bunny 单分辨率时行为不变）。
3. **10-19 修了进度条总长来回跳**：`DASHttp.js` 在 `_dashSourceOpen` 里把 `MediaSource.duration` 固定为 MPD 总时长 734 s（以前取"已 append 的末尾"，每加一片就变）。

**RealtimeISAC（Ubuntu 上，你不用跑，仅供对照）**
4. `--save-h`：把 H（1 kHz 复数相关系数，每个 1 ms TDD 帧一个，STFT 之前）+ 每帧 ch0/ch1 RMS 幅度 + 时间戳 边录边写盘（~58 MB/小时）。读取用 `debug/load_h.py`。改动在 `realtime/rt_dsp.py`、`rt_record.py`、`rt_main.py`。
5. `isac_bridge.py`/`isac_session_log.py`：自动起 NTG logger、回车打点、overview 图（10-09 起，家里用 `--no-ntg` 关）。
6. 真机冒烟（10-21，B210，无发射源）：60 s、3000 步，p50 3.4 ms / p99 6.1 ms（预算 20 ms），overflow=0、丢步=0。**但没有 TX，TDD 未锁定，H 是噪声**——只证明"管线实时能跑"，不证明有信号时的结果。

随包提供：`DASH_code_full_20261021.zip`（DASH 非视频代码的完整一致快照）、`RealtimeISAC_code_20261019.zip` + `RealtimeISAC_uncommitted_diff_20261019.patch`（仅供你对照/了解，**不要在这台机器上跑 rt_main**）、`verify_install.js`、`SHA256SUMS.txt`。

## 4. 你要完成的任务（按顺序；每步都要给出证据）

**T0 安全准备**
- 在改任何东西之前，把当前 `DASH_html_old-main` 里**非视频**文件复制一份备份到旁边，如 `DASH_html_old-main_backup_20261021\`（排除 `bunny_2s_1080p`、`tos_2s_1080p`，避免复制十几 GB）。
- 记录当前 Node 版本（`node -v`）和磁盘剩余空间。

**T1 验证安装是否正确**
- 在 `DASH_html_old-main` 目录运行 `node verify_install.js`（只读）。
- 逐条看输出：
  - [1]/[2] 有 ✗ → 视频没装全/解压损坏；把缺的列出来告诉用户，**不要自己编码补视频**；
  - [3] 代码 sha256 不同 → 这个文件和 10-19 版本不一致。先 `fc`/`diff` 对比，**如果是你这边有额外的本地改动，保留并报告，不要覆盖**；如果只是旧版，从 `DASH_code_full_20261021.zip` 取对应文件覆盖；
  - [4] 有缺失 → 说明没装 10-03/10-09 更新，用 `DASH_code_full_20261021.zip` 补（先备份）。
- 对 `SHA256SUMS.txt` 里的 zip 校验（`Get-FileHash -Algorithm SHA256`），如 zip 已删则跳过。

**T2 起 server 并做"服务端可达性"验证（家里：不连 NTG）**
- PowerShell：`$env:NTG_ENABLE=0; node server7.js`（cmd：`set NTG_ENABLE=0 && node server7.js`）。后台常驻，日志写到文件。
- 用 `curl`/`Invoke-WebRequest` 验证返回 200：
  `/`、`/?mpd=tos`、`/tos_ibmff_mixed_1080p-public-access.mpd`、`/tos_2s_1080p/tos_2s_20000kbit/tos_2sinit.mp4`、`/tos_2s_1080p/tos_2s_20000kbit/tos_2s367.m4s`、`/dash-js/DASHttp.js`。
- 检查日志里 `NTG poller: 已禁用（NTG_ENABLE=0）`。

**T3 局域网可达性**
- 确认 Windows 防火墙允许入站 TCP 8080（用户的 Ubuntu/手机要从别的设备访问）。若被拦：**只允许添加一条"入站 TCP 8080、仅专用网络"的规则**，这一项你可以直接做，并在汇报里写清添加命令与撤销命令（`Remove-NetFirewallRule`）。其他任何防火墙/网络改动都要先问用户。
- 从 Windows 自己 `curl http://192.168.0.100:8080/?mpd=tos` 通即可；Ubuntu 侧的访问由用户/Ubuntu 的 Claude 验证。

**T4 无头 Chrome 端到端（如果这台机器装了 Chrome/Edge 且能无头跑；没有就跳过并说明）**
- 打开 `http://127.0.0.1:8080/?mpd=tos`，跑 30–60 s，收集控制台。验收：
  - 控制台出现 `3 distinct resolution(s) in MPD -> DASH_MULTIRES=true`；
  - `video.duration` 恒为 734（**不是**随缓冲增长）；
  - 无 `SourceBuffer`/MSE 异常；ABR 自动升到 4K 档（本机回环速度，应该升到 20 Mbit/s 档）；
  - 已知且无需处理的旧问题：`dash-js/PredTracesFerry/predTrace-LSTM.js` 404（目录大小写，Windows 遗留，无害）、`favicon.ico` 404、播到最后一片的 `webkitSourceEndOfStream is not a function`。

**T5 写一份给用户的"家里 WiFi 测试步骤"**（见第 6 节，你只需核对其中 Windows 侧的命令能用，并把你的 Windows IP 填进去）。

## 5. 约束（违反会出事）

- **绝对不要改这台机器的网络设置**（IP/掩码/网关/DNS/`netsh`/适配器启停）。用户今天刚被"临时改成 192.168.1.x"搞到 Windows 断网。Windows 现在是 DHCP 或 `192.168.0.100/24`、网关 `192.168.0.1`，保持原样。要改网络 → 停下来问。
- 不要 `git add -A`、不要提交/推送任何东西（用户习惯不 commit，除非当次明确要求）。DASH 目录里 `tos_2s_1080p/`（9.2 GiB）没在 .gitignore。
- 不要删除/覆盖 `sessions/`、`snapshots/`、`bunny_2s_1080p/`、`tos_2s_1080p/`。
- 不要重新编码视频、不要改 `DASH_MULTIRES` 逻辑、不要改 `adaptationlogic.js` 的 ABR 规则（论文的对照基线依赖它）。需要改代码才能通过某项验收 → 先备份、改最小、在汇报里列出 diff。
- 只有这些情形才可以停下来问用户：①要改网络/防火墙以外的系统设置；②发现本机有我不知道的本地改动且和 10-19 版本冲突；③磁盘不足；④视频缺失需要重新传。其余一律自己做完并汇报。

## 6. 用户周三晚上/周末前要在"家里 WiFi"做的测试（供你核对 Windows 侧命令，并写进汇报）

目标：①确认 USRP 感知链路**能实时跑**；②在家用 WiFi 测 tos 阶梯的真实网速表现；③周日去学校前把流程跑顺。

| # | 在哪 | 命令 / 操作 | 期望 |
|---|---|---|---|
| a | Windows | `node server7.js`（`NTG_ENABLE=0`） | 8080 监听 |
| b | Ubuntu 终端 A | `python3 ~/sdr_workspace/RealtimeISAC/isac_bridge.py --no-ntg`（注意 `python3`=miniconda） | 起 isac_server :8765 |
| c | Ubuntu 终端 B | `cd ~/sdr_workspace/RealtimeISAC/realtime && /usr/bin/python3 rt_main.py --duration 600 --isac --save-h`（注意 `/usr/bin/python3`） | p99 远小于 20 ms，overflow=0，`h_logs/` 出文件 |
| d | Ubuntu Chrome | `http://192.168.0.100:8080/?mpd=tos&isacws=ws://127.0.0.1:8765/isac` | 视频走**路由器→Ubuntu WiFi**，正好测 WiFi；ISAC 状态条有数据 |
| e | Ubuntu 终端 B（替代 c） | 无发射源时 TDD 不锁定，判决没意义；联调页面用 `rt_main.py --dry-run --duration 300 --isac`（合成 120 Hz 多普勒，会周期性判"有人"） | 页面 ISAC 切 ON 后，检测到多普勒 → 强制最低档 + 缓冲目标放大到 30 s，恢复后逐步升档 |

- 家里测到的网速**不代表** NTG-2501（学校链路中位 ≈12 Mbit/s、上限 ≈24.5 Mbit/s）。要在家模拟 NTG，可用 Chrome DevTools Network 限速到 12 Mbit/s，或用 `~/dash_src/tools/e2e_test.py --throttle-mbps 12`（Ubuntu 上）。
- 不要把 `DASH_html_old-main` 里的 `isac_ws_client.js` 默认地址改成家里的 IP；用 URL 参数覆盖即可。

## 7. 你完成后的汇报格式（写到 `HANDOFF_REPORT_20261021.md`，放在 DASH_html_old-main 同级目录，并在对话里贴同样内容）

1. T0–T4 每项：通过 / 未通过 / 跳过 + 一两行证据（命令输出、关键日志行）。
2. `verify_install.js` 的完整输出。
3. 你改动过的所有文件清单（路径 + 为什么 + 备份在哪）；没改就写"无"。
4. 你的 Windows IP、Node 版本、`server7.js` 的启动命令、是否加了防火墙规则（及撤销命令）。
5. 给用户的一句话结论：这台 server 现在能不能直接用于周末测试；还缺什么。

## 8. 周日（10-25）的学校测试

用户说"下次测试在这周日"，具体方案他还没告诉我；最可能是 10-16 讨论过的**距离分区 + 提前量**实验（2/3/4 m 横向距离穿越，节拍器控速，随机顺序，回车打点区分"起步/踏过 LOS"，并行走对照），Ubuntu 侧会用 `--save-h` 录 H。**这部分不要你做**，只需要保证 server 在学校网络（NTG 开着、Master 192.168.1.201）下能起：届时去掉 `NTG_ENABLE=0` 即可，`NTG_MASTER_IP` 如不同再改。不确定的地方请在汇报里列出，让用户确认，不要自行假设。
