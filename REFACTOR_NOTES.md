# 重构笔记：从 validate_capture_and_caf.py 到实时 ISAC

> ## ⚠️ 2026-08-14 更新：本文的优先级已被推翻，先读 [REALTIME_PROMPT.md](REALTIME_PROMPT.md)
>
> 用户后来明确了新目标：**实时、完全不落盘、采样率降到 1–2 MHz、最终只输出「有人/无人」**。
> 在「没有文件」的前提下：
>
> - 第五节那 7 步计划里的 **`formats/`（D1/D5）、`io/experiment_store`（D2）、离线 `viz`**
>   基本作废——没有存盘格式要抽象，没有 JSON 要读写，没有 GIF 要拼。
> - 仍然成立的只有 **D3（合并两个 CAF 函数）** 的思想：实时内核必须只有一份。
> - 而且实测证明这**不是重构，是重写**——现有代码是彻底 file-first 的批处理管线，
>   能继承的只有 UHD 设备配置块和 sc16 定标数学，处置表见 REALTIME_PROMPT 第 6 节。
>
> **本文以下内容仍然准确**（行号、重复代码清单、拦路石分析都是实测的），
> 作为「现有代码到底长什么样」的存档参考有效；但不要再照着第五节的顺序动手。

现状是 809 行、2 个文件、0 个抽象层。下面把它拆成盒子，标出复用点和实时化的拦路石。
**本文件只是方案，代码一个字符都没动。**

行号全部指向 `current/` 和 `reference/` 里的副本（与根目录原件逐字节一致）。

---

## 一、目标分层

```
realtime_isac/
├── config.py          采集/处理参数（现在散在两个文件的常量块里）
├── formats/           存盘格式：sc16 / fc32 编解码、memmap 打开、格式校验
├── io/                实验目录：建/找/读写 acquisition_parameters.json
├── acquisition/       USRP B210 取流（现在只有「取流即落盘」一种模式）
├── dsp/               ranging / pdp / decimate / caf / quality
├── viz/               谱图、PDP 图、GIF
└── app/               编排：offline 批处理 / realtime 流式
```

## 二、函数 → 盒子 对照表

### `current/validate_capture_and_caf.py`（562 行）

| 行 | 符号 | 归属 | 备注 |
|---|---|---|---|
| 26–48 | 常量块 | `config` | sc16 定标、阈值、Tw/step/DECIM、PDP/GIF delay 范围、采集时长 |
| 51–57 | `range_step_for_sample_rate` | `dsp/ranging` | 纯函数，直接搬 |
| 60–72 | `read_2ch_iq_memmap` | `formats/sc16` | 格式相关，见 D5 |
| 75–86 | `sc16_to_complex64` | `formats/sc16` | 格式相关，见 D5 |
| 89–100 | `check_data_format` | `formats/registry` | 格式相关，见 D5 |
| 103–120 | `find_latest_experiment_folder` | `io/experiment_store` | CWD 耦合，见 R2 |
| 123–128 | `load_acquisition_parameters` | `io/experiment_store` | 与写入端配对，见 D2 |
| 131–133 | `run_step1_acquisition` | `app` | 3 行胶水，重构后消失 |
| 136–155 | `compute_pdp_delay_memmap` | `dsp/pdp` | |
| 158 | `MAGNITUDE_CHECK_CHUNK` | `config` | |
| 161–179 | `check_iq_magnitude_abort_memmap` | `dsp/quality` | 格式相关（直接读 int16 字段），见 D5 |
| 182–187 | `build_decimated_channel` | `dsp/decimate` | |
| 190–222 | `fast_caf_spectrogram` | `dsp/caf` | 与下面重复，见 D3 |
| 225–250 | `fast_caf_spectrogram_at_delay` | `dsp/caf` | 与上面重复，见 D3 |
| 253–260 | `range_label_for_delay` | `dsp/ranging` | 纯函数，直接搬 |
| 263–393 | `run_validation` | `app/offline` + `viz` | 131 行，算法与画图缠在一起，见 D4 |
| 396–514 | `run_validation_gif` | `app/offline` + `viz` | 119 行，同上，见 D4 |
| 517–562 | `main` | `app/entrypoint` | 开关靠改源码，见 D6 |

### `current/USRPSaveData2Channel.py`（247 行）

| 行 | 符号 | 归属 | 备注 |
|---|---|---|---|
| 9–29 | 常量块 | `config` | 含 SC16 定义，与上面重复，见 D1 |
| 31–41 | `create_experiment_folder` | `io/experiment_store` | |
| 43–68 | `save_acquisition_parameters` | `io/experiment_store` | JSON schema 的唯一定义处，见 D2 |
| 70–197 | `acquire_and_save` | `acquisition/usrp_b210` | 128 行，配置+取流+落盘+统计全在一个函数，见 R1 |
| 199–245 | `main` | `app/entrypoint` | |

---

## 三、复用 / 重复清单（D = duplication）

按「值不值得先修」排序。

### D1 — SC16 定义被抄了两份 ⚠️ 高危

- `USRPSaveData2Channel.py:19-20`（写入端）
- `validate_capture_and_caf.py:28-29`（读取端）

两处各写了一遍 `SC16_DTYPE` 和 `SC16_SCALE = 32767.0`。
`validate_capture_and_caf.py:26-27` 的注释自己承认了这件事：

> 定标系数须与采集端 (USRPSaveData2Channel.py 的 SC16_SCALE) 一致，否则幅度/能量算出来是错的。

靠注释维持同步。改一边忘另一边 → 幅度/能量静默算错，不报错。
**这是整个项目里最该先抽出来的东西**，一个 `formats/sc16.py` 就解决。

### D2 — `acquisition_parameters.json` 的 schema 没有单一来源

- 写：`USRPSaveData2Channel.py:48-61`（11 个字段的 dict 字面量）
- 读：`validate_capture_and_caf.py:123-128`，字段名散落在 `:91`(`data_format`)、
  `:280` 和 `:437`(`sample_rate_hz`)

读端还硬编码了两处一模一样的 fallback：`params["sample_rate_hz"] if ... else 20e6`
（`:280` 和 `:437`）——而实际采集是 30 MHz，这个 20e6 的默认值一旦真的生效就是错的。
应该收成一个带校验的 dataclass。

### D3 — 两个 CAF 函数差别只有 delay 从哪来

`fast_caf_spectrogram`(190–222) 和 `fast_caf_spectrogram_at_delay`(225–250)：

- `:200-205` 与 `:229-234` —— delay 正负分支切片，**6 行完全相同**
- `:206-222` 与 `:235-250` —— 对齐/裁剪、`effective_fs`、`conj(ch0)*ch1`、
  `nperseg`/`noverlap`、blackman 窗、`spectrogram`、`fftshift`、转 dB、频率 mask，
  **17 行完全相同**

唯一真差别：前者可选先跑 PDP 自动挑 delay 并额外返回 `Sxx_complex`/`pdp_amp`，
后者用调用方给的 `delay_bin`。抽一个 `caf_at_delay(...)` 内核，前者变成
「PDP 挑 delay → 调内核」的薄壳。**这个内核就是将来实时处理的热点函数**，
必须只有一份，否则优化时会漏改一个。

### D4 — 两个 run_* 函数的前 30 行是同一段

`run_validation:265-301` 与 `run_validation_gif:422-465`，顺序都是：
解析文件夹 → `isdir` 检查 → `2ch_iq_data.bin` 存在性 → `load_acquisition_parameters`
→ `check_data_format` → `sample_rate_hz` fallback → `read_2ch_iq_memmap` → 算 `n_total`
→ `nperseg` 长度守卫 → `check_iq_magnitude_abort_memmap` + 同一段警告文案。

抽成 `open_experiment(folder, max_duration_sec) -> ExperimentHandle`。
剩下的部分（画 1~3 联图 vs 逐 delay 出帧拼 GIF）才是真差异，归 `viz`。

### D5 — `reference/validate_capture_and_caf_fc32test.py` 是整份复制 ⚠️ 最大一处

531 行 vs 562 行，`diff` 只有 131 行不同，其中大部分还是注释和常量。
**真正有逻辑差异的只有 4 个函数**：`read_2ch_iq_memmap`、`sc16_to_complex64`、
`check_data_format`、`check_iq_magnitude_abort_memmap`——也就是「格式相关」那一层。
它自己的文件头注释（`:11-13`）说得很清楚：

> 跟 validate_capture_and_caf.py 的唯一区别：…这几个"数据格式相关"的函数改成认 fc32；
> CAF/PDP/GIF 的算法逻辑一字未动。

也就是说，CAF/PDP/GIF 那 400 多行算法被抄了两份，只为了换一个读盘格式。
`formats/` 抽出来之后，这整个文件塌缩成 **一个 fc32 reader 类 + 一行注册**，
回归测试变成「同一套算法 × 两个 format 后端」。

顺带一提：它连注释里的错别字都一起抄了——`:169` 还写着「先在原始 sc16 memmap 上」，
但它处理的是 fc32。抄文件就会抄成这样。

### D6 — 配置靠改源码，而且有一处两份

跑不同实验要直接编辑源码：`run_acquisition`(`:519`)、`PDP_ON`/`PHASE_ON`(`:520-521`)、
`run_gif`(`:524`)、`experiment_folder`(`:526`)、以及文件头 `:26-48` 的一整块常量。

其中 `ACQUISITION_DURATION` 存在两份且值不一样：

| 位置 | 值 |
|---|---|
| `USRPSaveData2Channel.py:29` `ACQUISITION_DURATION` | `10.0` |
| `validate_capture_and_caf.py:44` `ACQUISITION_DURATION_SEC` | `20.0` |

联跑时 validate 显式传参（`:528` → `USRPSaveData2Channel.py:202-203`），20.0 生效；
但单独跑 `USRPSaveData2Channel.py` 就是 10.0。同一个概念两个数，
后续按文件夹名回溯实验时很容易被坑。

---

## 四、实时化拦路石（R = realtime blocker）

现有设计是**彻底 file-first** 的：采集整段落盘 → memmap → 批处理。要做实时，
下面几条是结构性的，不是调参能解决的。

### R1 — 采集与落盘焊死在一个循环里

`acquire_and_save`（`USRPSaveData2Channel.py:70-197`）在 recv 循环里直接
`f.write(...)`（`:160`），没有任何对外的数据出口。整个函数把设备配置、流控、
落盘、统计打印揉在一起（128 行）。

实时需要的是 `recv → ring buffer → 消费者`，落盘退化成其中一个可选消费者。
这是**第一个要动的地方**，其它实时改造都依赖它。

### R2 — 到处隐式依赖 CWD

`find_latest_experiment_folder`(`:104`) 在 CWD 下 glob；`create_experiment_folder`
(`USRPSaveData2Channel.py:38`) 在 CWD 下建目录；输出目录 `"validate_caf_output"`
硬编码相对路径（`:308`、`:408`）。

在 `RealtimeISAC/` 里换个目录跑，数据就散到另一处去了（README「路径约定」一节）。
路径要参数化，`data/` 才能真正启用。

### R3 — `spectrogram` 是整段批处理

`:213` 和 `:242` 把整条 `prod_stream` 一次性喂给 `scipy.signal.spectrogram`。
实时要改成滑窗增量 STFT：维护 `nperseg` 长的环形缓冲，每 `step` 秒出一列。
参数已经是对的（`Tw=0.2s`、`step=0.1s`、`DECIM=16` → `:33-35`），
只是执行模型要从「一次算完」换成「来一块算一列」。

### R4 — 质量检查要先扫全量

`check_iq_magnitude_abort_memmap`（`:161-179`）在处理前把整段数据扫一遍数饱和点。
实时版本应该在流上滚动统计，超阈值就报警，而不是阻塞在前面。

### R5 — GIF 走磁盘中转

`run_validation_gif` 逐 delay 存临时 PNG（`:490-491`），再读回来拼 GIF（`:498`），
最后删掉（`:508-512`）。离线出片没问题，实时显示要换成直接刷 buffer 的实时瀑布图。

### R6 — `plt.show()` 会阻塞

`:324` 和 `:391`。实时链路里不能有阻塞调用，显示要么独立线程/进程，要么非交互后端。

---

## 五、建议的动手顺序

前 3 步不碰算法，只搬家，风险最低，而且每步都能用现有数据验证结果不变：

1. **`formats/`**（解决 D1 + D5）：sc16/fc32 两个 reader，统一接口。
   验收标准：`reference/` 那个 531 行的文件删掉，回归测试改成同一套算法跑两个 format。
2. **`io/experiment_store`**（解决 D2）：JSON schema 收成 dataclass，读写两端共用；
   顺手把 `20e6` 那个假 fallback 干掉。
3. **`dsp/caf` 内核**（解决 D3 + D4）：合并两个 CAF 函数，抽 `open_experiment`。
   算法与画图分家，`viz` 独立。
4. **`config.py`**（解决 D6）：常量收口，`ACQUISITION_DURATION` 只留一份。
5. **`acquisition/` 拆流**（解决 R1）：`recv → ring buffer`，落盘变成可选消费者。
   ——到这一步才算真正进入实时。
6. **`dsp` 增量化**（解决 R3 + R4）：滑窗 STFT，滚动质量统计。
7. **`viz` 实时化**（解决 R5 + R6）：非阻塞瀑布图。

第 1–4 步是纯重构，`current/` 里的行为应当逐位可复现；第 5 步开始才改变执行模型。
