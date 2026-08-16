# RealtimeISAC

以 `validate_capture_and_caf.py` 为核心，重新起步的实时 ISAC（通感一体）项目。

本文件夹目前**只做整理，不改代码**：`current/` 和 `reference/` 里的 `.py` 与
`sdr_workspace/` 根目录下的原件逐字节一致（`cmp` 校验过）。

**开发实时版本请从 [REALTIME_PROMPT.md](REALTIME_PROMPT.md) 开始**——它包含本机实测的
性能/内存数据、架构决策和分阶段计划。`REFACTOR_NOTES.md` 是更早写的现有代码剖析，
其优先级已被「不落盘 + 实时」这个新前提推翻，现在只作存档参考。

---

## 目录结构

```
RealtimeISAC/
├── README.md              本文件：怎么跑、环境约束、路径约定
├── REALTIME_PROMPT.md     ★ 实时开发提示词（实测数据 + 架构 + 计划）
├── REFACTOR_NOTES.md      现有代码剖析（存档；优先级已被 PROMPT 取代）
├── requirements.txt
├── current/               ← 当前唯一可运行的链路（原样复制）
│   ├── validate_capture_and_caf.py    核心：编排 + PDP + CAF + 出图/GIF
│   └── USRPSaveData2Channel.py        采集：USRP B210 双通道 → sc16 落盘
├── reference/             ← 不在实时链路上，留作回归对照
│   └── validate_capture_and_caf_fc32test.py
└── data/                  ← 采集输出（experiment_*）落地处，当前为空
```

### 为什么 `current/` 里两个文件必须同级

`validate_capture_and_caf.py:132` 是裸的 `import USRPSaveData2Channel`（top-level
module import，不是包内相对导入）。在不改代码的前提下，两个文件**必须待在同一个目录**，
否则 `run_acquisition=True` 时会 `ModuleNotFoundError`。

所以这一层暂时是平铺的——真正的分层（acquisition / formats / dsp / viz / app）
要等重构时才能落到目录上，方案见 REFACTOR_NOTES.md。

### `reference/` 里那个文件

`validate_capture_and_caf_fc32test.py` 是 `validate_capture_and_caf.py` 的整份拷贝
（531 行 vs 562 行），只把 4 个「数据格式相关」函数换成认 fc32，用来拿旧数据回归验证
CAF 算法。它的 `import USRPSaveData2Channel` 在 `run_step1_acquisition()` 里是惰性的，
而它的 `main()` 不采集（写死读 `experiment_30MHz_static_20260227_150212`），
所以放到 `reference/` 单独一层不会破坏它。

它是本项目里最大的一处复制粘贴，重构后应该塌缩成一个格式插件——见 REFACTOR_NOTES 的 D5。

---

## 环境（重要）

**必须用 `/usr/bin/python3` 跑，不能用默认的 `python3`。**

这台机器上 `python3` 解析到 miniconda（`/home/usrpb210/miniconda3/bin/python3`，3.13.5），
它有 numpy/scipy/matplotlib/Pillow 但**没有 uhd**；即使把 PYTHONPATH 指到
`/usr/local/lib/python3.13/site-packages`，也会因为 conda 自带的 libstdc++ 太旧而崩：

```
ImportError: .../miniconda3/lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found
             (required by /usr/local/lib/python3.13/site-packages/uhd/libpyuhd...so)
```

`/usr/bin/python3`（3.12.3，apt）四个科学计算包 + uhd 全都有，是唯一能完整跑通的解释器。

| | miniconda python3 (3.13.5) | /usr/bin/python3 (3.12.3) |
|---|---|---|
| numpy / scipy / matplotlib / Pillow | ✅ | ✅ |
| uhd | ❌ GLIBCXX 冲突 | ✅ |

UHD 版本：`4.8.0.HEAD-0-g308126a4`（源码装到 `/usr/local`）。设备：B210，
`serial=321D889`（写死在 `USRPSaveData2Channel.py:9`）。

---

## 怎么跑

两个脚本都靠**当前工作目录**找/建数据文件夹（见下面「路径约定」），所以先 `cd` 进 `current/`：

```bash
cd RealtimeISAC/current

# 采集 + 分析一条龙（validate 的 main() 里 run_acquisition=True）
/usr/bin/python3 validate_capture_and_caf.py

# 只采集，不分析
/usr/bin/python3 USRPSaveData2Channel.py
```

只想分析已有数据、不重新采集：把 `validate_capture_and_caf.py:519` 的
`run_acquisition` 改成 `False`，它会用 `find_latest_experiment_folder()` 挑最新的
`experiment_*`；或者在 `:526` 写死文件夹名。

主要开关都在 `main()`（`validate_capture_and_caf.py:517-558`）和文件头的常量块
（`:26-48`）里，目前是靠改源码切换的——重构时这些要收进 config，见 REFACTOR_NOTES 的 D6。

### 路径约定（当前行为，尚未改）

| 谁 | 位置 | 行为 |
|---|---|---|
| 采集输出 | `USRPSaveData2Channel.py:31-41` | 在 **CWD** 下建 `experiment_{采样率}MHz_static_{时间戳}/` |
| 找最新数据 | `validate_capture_and_caf.py:104` | 在 **CWD** 下 `glob("experiment_*")` |
| 图/GIF 输出 | `validate_capture_and_caf.py:308, 408` | 写到 **CWD** 下的 `validate_caf_output/` |

也就是说数据会落在你 `cd` 进去的那个目录里，而不是 `RealtimeISAC/data/`。
`data/` 现在是空的占位，等重构把路径参数化之后才会真正启用。

**注意体积**：30 MHz × 2 通道 × sc16 = 120 MB/s，20 秒一次约 2.4 GB。根目录现存的
`experiment_*` 已经有 40+ GB，没有复制进来。

---

## 依赖闭包

`validate_capture_and_caf.py` 的一方依赖只有一个，链路很浅：

```
validate_capture_and_caf.py
├── USRPSaveData2Channel.py        ← 唯一的一方依赖（:132 惰性 import）
│   └── uhd
├── numpy / scipy.signal (spectrogram, windows)
├── matplotlib.pyplot
└── PIL.Image                      ← 只有出 GIF 时才 import（:418）
```

反向没有任何依赖：全仓库除了 `validate_capture_and_caf_fc32test.py`，
没有第三个文件 import 这两个模块。根目录那一堆 `Udoppler_*.py` / `analysis_*.py` /
`check_*.py` 是独立的一次性分析脚本，跟这条链路无关，所以没有带进来。

（`USRPSaveData2Channel_HighRate.py`、`USRPSweepData2Channel*.py` 等是 
`USRPSaveData2Channel.py` 的其它变体，同样不在闭包里，需要的话再单独挑。）
