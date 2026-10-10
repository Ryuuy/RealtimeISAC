# data_archive — 本机 `isac_logs/` 与浏览器快照的完整备份（2026-10-03 ~ 2026-10-10）

这些文件原来**只在实验室电脑上**（`isac_logs/` 被仓库 `.gitignore` 忽略，浏览器快照在 `~/Downloads`）。为了在家也能分析，原样存档于此。

- `isac_logs/`：ISAC 会话日志（`.npz`）、NTG-2501 轮询（`.csv`/`.log`）、`isac_bridge` 退出时自动生成的概览图（`overview_*.png`）。**未加工，原样复制**。
- `dash_snapshots_gz/`：浏览器 "Save Full Snapshot" 的 JSON，**无损 gzip -9**（`gunzip` 后与原件逐字节相同；原件 sha256 见下表），为节省仓库空间。
- 已经在 `experiments/*/raw/` 里**逐字节相同**的文件（`203031`、`205507` 的 npz/csv/log，以及 3 个快照）没有重复存放，表里标了位置。
- 注意：`20261010_180657` 的 CSV/log 在此是**最终完整版**（到 20:30）；`experiments/20261010_knife_edge_plate/raw/` 里是 18:35 的截取版（实验窗口 18:09–18:19:30 在其中）。
- `long3_*.png`（长图）没有归档（可用 `experiments/*/scripts/plot_long3.py` 重新生成）。

## 会话一览

| 会话 | 开始 | USRP 最后样本 | 时长 | >3 s 断连数（ch0） | 断连中位数 | 打点 | 说明 |
|---|---|---|---:|---:|---:|---:|---|
| 20261003_162501 | 10-03 16:25:01 | – | – | – | – | – | 10-03 早期会话（旧格式）（旧格式，无 dbfs） |
| 20261003_164723 | 10-03 16:47:23 | – | – | – | – | – | 10-03（旧格式，无 dbfs） |
| 20261003_170144 | 10-03 17:01:44 | – | – | – | – | – | 10-03（旧格式，无 dbfs） |
| 20261003_173227 | 10-03 17:32:27 | – | – | – | – | – | 10-03（旧格式，无 dbfs） |
| 20261003_181237 | 10-03 18:12:37 | 10-03 18:41:40 | 29 min | 27 | 15.7 s | 0 | 10-03 最长的一次（含多次长断连，分布比后来分散） |
| 20261009_151428 | 10-09 15:14:28 | 10-09 15:24:12 | 10 min | 9 | 9.9 s | 0 | 10-09 第一次带 NTG 同步记录 |
| 20261009_160939 | 10-09 16:09:39 | 10-09 16:13:23 | 4 min | 2 | 9.9 s | 0 | 10-09：2 次断连，用于"断连前特征"分析 |
| 20261009_163214 | 10-09 16:32:14 | 10-09 16:42:02 | 10 min | 12 | 9.9 s | 0 | 10-09：12 次断连 |
| 20261009_173510 | 10-09 17:35:10 | 10-09 17:48:43 | 14 min | 20 | 9.9 s | 0 | 10-09 |
| 20261010_170034 | 10-10 17:00:34 | 10-10 17:32:55 | 32 min | 19 | 9.9 s | 0 | 10-10 17:00 开始；17:19 之后是遮挡实验（17:19 以前可能含旧数据） |
| 20261010_180657 | 10-10 18:06:57 | 10-10 18:20:08 | 13 min | 25 | 9.9 s | 0 | 10-10 刀口步进实验（分析窗口 18:09:00–18:19:30；会话一直录到 20:30，完整 CSV/log 在此） |
| 20261010_203031 | 10-10 20:30:31 | 10-10 20:44:23 | 14 min | 23 | 9.9 s | 0 | 10-10 连续扫板（已在 experiments/20261010_continuous_sweep_trials） |
| 20261010_205507 | 10-10 20:55:07 | 10-10 21:12:40 | 18 min | 32 | 9.9 s | 0 | 10-10 人体穿越（已在 experiments/20261010_205507_human_crossing） |

（"断连"= ch0 reference 通道 < −30 dBFS，只对含 dbfs 的会话有效。）

## 文件与校验和（sha256 为**原始文件**的）

| 文件 | 大小 [B] | sha256 | 状态 |
|---|---:|---|---|
| `isac_20261003_162501.npz` | 322613 | `5715141795dc6935d0ccf3d0e3475321eeb578d8dd97701923bfe77e592360e8` | archived |
| `isac_20261003_164723_p001.npz` | 1421398 | `9e8565ad2e7ef99fd20c4f1dce6ddefb19d964cd47f2a3d2e7f230c7d000f9ad` | archived |
| `isac_20261003_170144_p001.npz` | 617101 | `1c4295afadc123140e2a1032dd325eb4dc21f92058ada4cd4b5f0577bc011259` | archived |
| `isac_20261003_173227_p001.npz` | 4685771 | `d3e696e5fe01430da554da663add71fbd4cb406935ab96e1f9e2c9e9c59c904f` | archived |
| `isac_20261003_181237_p001.npz` | 9008562 | `d0b27fff23e777b3585093b0252c9b1478414c64ed4eb2e531752b52daf317d9` | archived |
| `isac_20261009_151428_p001.npz` | 3379901 | `8d4afb3fcc1c17b65dbbdcbc798d337560639eba3d04bd8e8e04c79ff4ecb2b8` | archived |
| `isac_20261009_160939_p001.npz` | 1221632 | `34e86d670b1f5ca81b4961565d785113cb819cfa1dff16defcea57fdb32ad7b6` | archived |
| `isac_20261009_163214_p001.npz` | 3361563 | `91f5b33b1dbfd38c16bc82e4a64b995458c830454362f8c45d63b442f78cdf2b` | archived |
| `isac_20261009_173510_p001.npz` | 4675614 | `24ba4f961d1bffe215d6fa071fde25b6979bcb8e5abdad00398ed2efb4e0cc69` | archived |
| `isac_20261010_170034_p001.npz` | 10484517 | `4d70cd4346b34f405b2a10d12691bc341ef8eae9102b58ec3c54bcb3402a235c` | archived |
| `isac_20261010_180657_p001.npz` | 4186106 | `be9d809a94369fc0c3aecb2d10f47bba4ffbb6b0fa9a62920a9dd7dfc4c21c4e` | archived |
| `isac_20261010_180657_p002.npz` | 7512 | `403040e41eaad2368f5b6fce59a3ab53fbdaacc37f01ae2b06c16d89f04cc3ed` | archived |
| `isac_20261010_203031_p001.npz` | 4344719 | `89f553c1dc9a99fcec5c6efb824943a334dbf45c2ba744708cb8f0e36d8b479c` | (already stored byte-identical in experiments/20261010_continuous_sweep_trials/raw/) |
| `isac_20261010_205507_p001.npz` | 5560950 | `2f666a127eb390a5c2ce4ac17d8bfc7a2345d41dd87c0fed83d3e12b4af8e48c` | (already stored byte-identical in experiments/20261010_205507_human_crossing/raw/) |
| `ntg_20261009_151428.csv` | 4590383 | `14a06c948a30e6bf15f5e71096e04272f26415686e11bcffd45b4048b1b93ac1` | archived |
| `ntg_20261009_151428.log` | 257268 | `62884ced42fd21f24072fcb539e7fcfc9929900430c5cd71a7e89696be9cea2d` | archived |
| `ntg_20261009_160939.csv` | 349131 | `14d777e37f8b72e36f604d27547c0013cfc69a8e2e517b296911d8669342e66c` | archived |
| `ntg_20261009_160939.log` | 19686 | `cf1b37660f2cfa544346139003f928c5f8c1b275a06f142eea6eea45f782c96a` | archived |
| `ntg_20261009_163214.csv` | 5153410 | `7d1e4233b4a73547247a98aeb44d3f46e4f6454b432a499f66b628b220f65abe` | archived |
| `ntg_20261009_163214.log` | 302259 | `84b4a71f1a290f8d555a6c70815e58c44fecc5f52f722aa566b4bfa879dee4e0` | archived |
| `ntg_20261009_173510.log` | 140 | `252bb0083de7e52aee80abe5655f799b82f961c78ef615f0b017edb8b21c8cd2` | archived |
| `ntg_20261010_170034.csv` | 2685088 | `b8fd22b4bde003f02a3943a3439a894b38541ba0748999f015e305f42ddee52f` | archived |
| `ntg_20261010_170034.log` | 158355 | `c11fae9ed9f12c95984984985bfc12863e72b9f7afa227cf3e69812fb25713c2` | archived |
| `ntg_20261010_180657.csv` | 13325485 | `f4c1470b3d5941783d4a18a6d2e57f649670c7735d835837044aada282e8d8e8` | archived |
| `ntg_20261010_180657.log` | 756944 | `b1576d01a0902097fa3cc9111b41993cf2e997eeadd2a7cb287a38b7a5741426` | archived |
| `ntg_20261010_203031.csv` | 1008924 | `f798651551d0a28b7ceba7aeba6b0844e7ccb6b4e1218678965f2853ff34fa4a` | (already stored byte-identical in experiments/20261010_continuous_sweep_trials/raw/) |
| `ntg_20261010_203031.log` | 62622 | `8248047ff14f8f373e7ed7f0cc6a84a2770528e314bbf38c2a93028358d5b294` | (already stored byte-identical in experiments/20261010_continuous_sweep_trials/raw/) |
| `ntg_20261010_205507.csv` | 1174318 | `3045395f9f53b95fde6d9841c4c1d9322173dd30dbefce45f2efbfdee13898de` | (already stored byte-identical in experiments/20261010_205507_human_crossing/raw/) |
| `ntg_20261010_205507.log` | 74234 | `8c2d24c985f508a5e2515d63312f8ab555cced6891c09d8f553c6e065568bfd5` | (already stored byte-identical in experiments/20261010_205507_human_crossing/raw/) |
| `overview_20261009_151428.png` | 316132 | `e678d7ffaa906df195aea2c15cd2c5aa424e1233bc2a7919ea7ae02784e85cb5` | archived |
| `overview_20261009_160939.png` | 844969 | `b3f3cf4294a928b01341b667930ab467a92b73d6048f9a438783ddec2c8add59` | archived |
| `overview_20261009_163214.png` | 334094 | `65c4d03cac710cfbfaf6a6dae051bc74967dd17e0fdebb6c7ae3571c0e75b21f` | archived |
| `overview_20261009_173510.png` | 729072 | `ff8e2efa206a26cc77b1a70729284b11d0d0167e68d7cd75a6fc9a810f925f82` | archived |
| `overview_20261010_170034.png` | 820045 | `18c031cee39bd1b7d9c4502ef1f8f147ecfa021289ccb35d349a738658941d08` | archived |
| `overview_20261010_180657.png` | 242048 | `4ce4e5a49ff51313818bfb3094a4bb43f81cff2808405deaf34977e5212769f1` | archived |
| `overview_20261010_203031.png` | 882219 | `e2d6d8d7dc4daae56f13a1455ac45c974df0a0ed53f829378d9c1e4459686f28` | archived |
| `overview_20261010_205507.png` | 851911 | `f0de3568e5f795910a42241fb06e71c9c537108bcc861c93b20a64eea7336e4e` | archived |

### 浏览器快照

| 原文件 | 原大小 [B] | gz 大小 [B] | 原件 sha256 | 状态 |
|---|---:|---:|---|---|
| `dashSnapshot_2026-10-03_18-18-35.json` | 502857 | 97205 | `ad1745c9ed325b9e35108ee89a0919238d5be9f6d2621e525c3e4e3cc8e15b2e` | gzip -9 (lossless; gunzip restores the exact original) |
| `dashSnapshot_2026-10-03_18-21-02.json` | 9899511 | 2968915 | `4d7458c4ca721189bfc55afa4d39366d0b78f29cad8567dd13e97f957480ad73` | gzip -9 (lossless; gunzip restores the exact original) |
| `dashSnapshot_2026-10-03_18-40-19.json` | 9414130 | 2752940 | `27a2e2d5824300db84152379fc1a08eb9832be5f74e691b2ec37da7760857e76` | gzip -9 (lossless; gunzip restores the exact original) |
| `dashSnapshot_2026-10-09_15-24-04.json` | 10828913 | 3498840 | `2b37d9a3cadefb93bcdaa4f305d30d55c99ad34fa60dbb14f0fabf1d08628a27` | gzip -9 (lossless; gunzip restores the exact original) |
| `dashSnapshot_2026-10-09_16-13-16.json` | 6941915 | 1370017 | `fb1ecccad8d47cd78d41e4e197d763c192b48e8b8a8e2d21354f75eaef818eef` | gzip -9 (lossless; gunzip restores the exact original) |
| `dashSnapshot_2026-10-09_16-39-01.json` | 9026970 | 2561268 | `741526101ec3e3927e426d5e0711d899c451b56006fb55031f9d083f777d6efc` | gzip -9 (lossless; gunzip restores the exact original) |
| `dashSnapshot_2026-10-09_16-41-48.json` | 11378872 | 3580542 | `ced08fcbe2a5205c61398a89aa50c8eabea338de056a92567aaf72c942792678` | gzip -9 (lossless; gunzip restores the exact original) |
| `dashSnapshot_2026-10-09_20-35-35.json` | 14648480 | 4775829 | `a19b4302156bf4c558c38dac5c13ad658f71d100f8e11092777d52e461b5dc3c` | gzip -9 (lossless; gunzip restores the exact original) |
| `dashSnapshot_2026-10-10_17-11-20.json` | 11679033 | 3599692 | `d1c5db4c6c56e7a15d10f9de29e3f8da3e61bfec43a8dc5866b21d4927b6e7ff` | gzip -9 (lossless; gunzip restores the exact original) |
| `dashSnapshot_2026-10-10_17-18-06.json` | 1121943 | 100104 | `2476d13a8c9ae0fa49f4e79afce6d72bb8447fe725cdbad0a7a386cf3be64d19` | gzip -9 (lossless; gunzip restores the exact original) |
| `dashSnapshot_2026-10-10_17-18-13.json` | 1060153 | 86992 | `9daa6783df7d5ef41ae5850893f84bfa719ae5035fe253b783298d45ebed5c58` | gzip -9 (lossless; gunzip restores the exact original) |
| `dashSnapshot_2026-10-10_17-32-38.json` | 15816359 | 5226349 | `6d60a47e8fac0c5e47fba7d244890979e6fdd7546482df38c5c422935f848578` | gzip -9 (lossless; gunzip restores the exact original) |
| `dashSnapshot_2026-10-10_18-19-55.json` | 15041987 | – | `18aec1728bd8c719b318fa75ac0e64d4e800298cbd466a32f495541da2e7541e` | (already stored byte-identical in experiments/20261010_knife_edge_plate/raw/) |
| `dashSnapshot_2026-10-10_20-44-06.json` | 20020435 | – | `abbfd1441544c7346e56f1f17075ff4aa060795785701d2cd79059851313aed7` | (already stored byte-identical in experiments/20261010_continuous_sweep_trials/raw/) |
| `dashSnapshot_2026-10-10_21-12-11.json` | 22307359 | – | `bc3b2c09cc100dc904dec08406455336908fa334fc00478e5dffebb8b440fb43` | (already stored byte-identical in experiments/20261010_205507_human_crossing/raw/) |

还原快照：`gunzip -k data_archive/dash_snapshots_gz/dashSnapshot_*.json.gz`，然后 `sha256sum` 与上表比对。

## 在家分析的提示

- 用到 `uhd` 的脚本必须用 `/usr/bin/python3`；纯分析（numpy/matplotlib）任意 Python 即可。
- 浏览器快照里的 `ntg.samples` 在 10-10 的几个快照里被页面缓冲截为 40000 条；完整 NTG 记录以 `ntg_*.csv` 为准。
- 10-09 以前的快照在页面修复前，`ntg.samples` 可能有重复/乱序（见 `experiments/*/README.md`）。
