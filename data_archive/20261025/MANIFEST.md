# data_archive/20261025 — 2026-10-25 实验数据（在家分析用）

生成于 2026-10-25 22:07:44（`sync_archive_20261025.py`，可重复运行）。

先读 `../../20261025_测量记录_论文用.md`（结论和限制）和 `walk_schedule.csv`（每一轮走了什么、几点开始、多少次）。

- `h_logs/h_20261025_<时间戳>.*`：dec（复数 complex64，1 kHz）、每帧 ch0/ch1 幅度、时间戳、可用标志。用 `debug/load_h.py` 或 `experiments/20261025_b210_doppler_checks/scripts/cut_events.py` 里的 `load_h()` 读。
- `sessions_raw/`：`isac_*.npz`（多普勒瀑布、检测、ch0/ch1 dBFS）、`ntg_*.csv`（NTG XML，10 Hz，`master_valid`）、`ntgstate_*.csv`（NTG 六状态）、`dataplane_*.csv`（ping 5 Hz）。**NTG 管理口在 21:21:48–~21:36 卡死，csv 里那段 625 s 的“断连”不是链路断，是接口没回应（ping 100% 通）。**
- `dash_snapshots_gz/`：浏览器快照（gzip -9，`gunzip` 后逐字节还原）；`telemetry` 里有 `requests`（每个分片请求含失败/超时）、`ntgStates`、`dataplane`（19:55 之后的快照才有这三项）。
- `probe_runs/`：无头页面 + 探测（requests.csv / state.csv / console.log / ping.log / newc.csv / keep.csv / ntg_states.csv）。
- `events/`：按排定时刻切好的事件窗口（窗口 [S−5, S+间隔−5)，10 s 一次的就是 ±5 s）：`<run>_events.npz`（dec、amp0、amp1）和 `<run>_slots.csv`（每个事件的汇总）。
- **没有存**：原始 IQ（`debug/iq_20261025_*.npz`，每份 1.6–2.4 GB，超过 GitHub 100 MB 限制）。

| 文件 | 大小 [B] | sha256 |
|---|---:|---|
| `WALK_LOG_20261025.md` | 6501 | `2b850bdf6e7e84518bd586c0015d223d1ec44ffa241ed61ef898b1334ce7b363` |
| `dash_snapshots_gz/dashSnapshot_2026-10-25_17-55-11.json.gz` | 69165 | `4a4a4f4967c154fe08d2df1b9ced7081a9d6fa48c273e2c2e6514dcb62c1e01f` |
| `dash_snapshots_gz/dashSnapshot_2026-10-25_18-29-15.json.gz` | 3262912 | `b9a0f940bcb3b308517a92d840a03293593cdf45706f225fce7b94a7254a9563` |
| `dash_snapshots_gz/dashSnapshot_2026-10-25_18-35-16.json.gz` | 185872 | `462c8b990b74f0227e8322105c10f1bcdb9e9ac948361cac63893b17a9ad6616` |
| `dash_snapshots_gz/dashSnapshot_2026-10-25_18-42-39.json.gz` | 2864134 | `0808ad39d932c2c133180970239d7591dd081b51bd2ac86193c8fbd47fe6c9a5` |
| `dash_snapshots_gz/dashSnapshot_2026-10-25_18-49-51.json.gz` | 1335058 | `53a133872c76d77f64a3eff3409868727523ee220672d32b92b1290ac1774e62` |
| `dash_snapshots_gz/dashSnapshot_2026-10-25_19-05-07.json.gz` | 2903150 | `00cee3e8672b98d77bc326b425f34b39276dce2dd93e9c7331f3c6ac764d6c82` |
| `dash_snapshots_gz/dashSnapshot_2026-10-25_19-11-01.json.gz` | 4931504 | `5265b6140200ac8238de0e87dd8f254be881693b175c18ba8b416a8eaa119361` |
| `dash_snapshots_gz/dashSnapshot_2026-10-25_19-27-04.json.gz` | 3225749 | `b0fe26d88180cfc2e5486710b1113b5d17f3d9035ddc9e88da892e379b0a9a18` |
| `dash_snapshots_gz/dashSnapshot_2026-10-25_19-27-54.json.gz` | 3511939 | `1f96ed70939844947b6584f1a1157c72573d497489b2c4c383f8e37ce597734c` |
| `dash_snapshots_gz/dashSnapshot_2026-10-25_20-55-06.json.gz` | 431195 | `f7e75ba857354397d20cc212b699ee7bf2ce92cd10f1877f4b0869888ca549c4` |
| `dash_snapshots_gz/dashSnapshot_2026-10-25_21-01-53.json.gz` | 2963521 | `ce6c192e3fc07296741421fc43a4fdfc052a7b958a1618703f5be56447731fb6` |
| `dash_snapshots_gz/dashSnapshot_2026-10-25_21-31-24.json.gz` | 8510665 | `a2578734172b7920cb1d7827e1668c6b151da16a8d5a7da47d8554d15c5afe3e` |
| `dash_snapshots_gz/dashSnapshot_2026-10-25_22-06-49.json.gz` | 5813715 | `b9abbdbde78ff47082e69131799504c38b29b6e98af9d40fb6f813e80d899d2f` |
| `dash_update_packages/DASH_20261025_code.zip` | 435578 | `8534734f85a1bfadbff3b91e20ecae3a20b390f1f5741964543bb10311b71176` |
| `dash_update_packages/DASH_20261025b_DASHttp_fix.zip` | 9079 | `d23125407565d6ec1bce7351f6bb5f3fd350215ceadc40f3799a28c279570612` |
| `dash_update_packages/DASH_20261025c_requestlog.zip` | 17682 | `a1f7fece6f70444ea7df6706ebd1516d3d84cee90a5425a4cf9857c339d0e4e6` |
| `dash_update_packages/README_20261025_先看这个.txt` | 3287 | `16b9cb410f768455643d61c33009f716867d1097e3722c0a534424913710281a` |
| `dash_update_packages/README_20261025b_DASHttp_fix.txt` | 1019 | `b2b2ab48141ff4ab470ff71eaf71620352003b0f3a8b7c046c667ac2680f8e9a` |
| `dash_update_packages/README_20261025c_requestlog.txt` | 1376 | `4f5d95f36b3624a5a833f7c6a5c17893e1117d3915373422613ce7f39b986911` |
| `dash_update_packages/SHA256SUMS.txt` | 89 | `1bbb5d7fbdae54f1d7ac7ae7d3d16b8ef2f6f83833b112f6cbfca35c8d00e97f` |
| `dash_update_packages/SHA256SUMS_20261025b.txt` | 97 | `dd432d30e77ad8713b73e0905a92c6b196a4b378eb358d52970c65b71407d65f` |
| `dash_update_packages/SHA256SUMS_20261025c.txt` | 96 | `dd51e78819db6706ee8ae722089da25731b3e30b5990ef53c9e0c69635e05e56` |
| `events/F_events.npz` | 4046247 | `7bac01bc80757546a1ef450c7fd85a4853a31187ac31947a2ca01e97bcc8f3a9` |
| `events/F_slots.csv` | 1200 | `9294c7b365e6c9887a7cbfb1aab8230546e2b948f672412d1e2f35b69897dcd0` |
| `events/G_events.npz` | 9411967 | `dd02603a15afaff545bebd06d536354c247afde2431f5084be812a916f7ebe53` |
| `events/G_slots.csv` | 7156 | `1f631c8a6901ea7cd0a3cfb9965ead23c02b70dbb4642a5fe341efcc018d7b10` |
| `events/H_events.npz` | 5852342 | `ffd35b39d6c3b76387e981027bdca71f1efbeec370af84f752713bbc8b98cbfe` |
| `events/H_slots.csv` | 4386 | `f627ee1518c45b6d8e541ff1029adcd4f33557c46cc067f93648d1ca30d11e89` |
| `events/P1_events.npz` | 4355552 | `17cbea561b5e1b8e903a2fc788566ccbf750dae23300974c7282e63609f3db7a` |
| `events/P1_slots.csv` | 1746 | `08f1d338571a3547352dbcae353caf8da7b11461b97a8204818e77fdb5aebb9c` |
| `h_logs/h_20261025_154319.amp.bin` | 160000 | `a779980455a842e0760cb86f4fa513707cc76be67dc03c15bbb2601785ea4da0` |
| `h_logs/h_20261025_154319.flag.bin` | 1000 | `683505f5e4f283d5a981fb10db3e481ea60e893614da3ada06fd85c3fed6285a` |
| `h_logs/h_20261025_154319.h.bin` | 160000 | `9496418b617460d9faefe8c7a70233950e20ccab550c2a18460c0813ce7973b1` |
| `h_logs/h_20261025_154319.json` | 453 | `73e413d15b0b19b76ee3d41cbac99ef26139be86cefda31cc6e60df75bf6b35e` |
| `h_logs/h_20261025_154319.t.bin` | 8000 | `dd235e54e21bc502ee3a4ec451540919efc47bcc31d0c85bbb274875af120c71` |
| `h_logs/h_20261025_181714.amp.bin` | 393760 | `b92ca408f791e2e484076ed58afae93aff91634f48aa3e3b9a3b70ddb8df4b02` |
| `h_logs/h_20261025_181714.flag.bin` | 2461 | `27d214721556e62c28eaa4994ba8a1486ebd2ad8c5cbf82366d0114880cbf9ec` |
| `h_logs/h_20261025_181714.h.bin` | 393760 | `1bbdf46f5ae7611a15930f29f30d8d411586f775eff78d3e40a08ccecbab48ef` |
| `h_logs/h_20261025_181714.json` | 453 | `6eee306ff10a84aaa4b47731f14c0c5abd25f5eb73de84c854ead674bdcecdf4` |
| `h_logs/h_20261025_181714.t.bin` | 19688 | `9b7463e24243a7a18d810b1deaac50d63e66c964cb5910c2778c09631fc24087` |
| `h_logs/h_20261025_181729.json` | 304 | `92f7bd2708601d3854352be3759e8f63b0056b54891914e4563fb122372b355d` |
| `h_logs/h_20261025_181741.json` | 304 | `d85754a4742c7d9264f0b25f04d16dd717ec544d79f7109e53cdc17fb46092ac` |
| `h_logs/h_20261025_181855.amp.bin` | 5011360 | `40ea3a1301ed5b519d85e6ae734b05e293d2b6b7bb96949e03c0ee104afb3534` |
| `h_logs/h_20261025_181855.flag.bin` | 31321 | `8890d5c423ff9c7ae0ef80606a830b1f7cc5cf96eaea86ca5812ad8744c8462a` |
| `h_logs/h_20261025_181855.h.bin` | 5011360 | `6fa8b4f4598c68131cf32487263de2f9b90f747c7c60fff9239375ea0cc3ca3d` |
| `h_logs/h_20261025_181855.json` | 455 | `f795e03f5c4d6e77ccdffec643c7a3e3021cd1e760b1834a7adc3d372505dc61` |
| `h_logs/h_20261025_181855.t.bin` | 250568 | `667e51a99140715482cd2419769d05f97d0d583e9c874d7747fbe90fb496582a` |
| `h_logs/h_20261025_183424.amp.bin` | 28733920 | `746b59fac0e645c7529b699147eb8bd7a32578b123f74781584b1f70a1db588d` |
| `h_logs/h_20261025_183424.flag.bin` | 179587 | `9947363e74475246440003e6601e6fd3956fb8faea6179b802159dee60f142de` |
| `h_logs/h_20261025_183424.h.bin` | 28733920 | `ab34a7d7a1f64a8cb8c6a2adc45cf514e3a4112294318875cbf73dd19dd8dd5d` |
| `h_logs/h_20261025_183424.json` | 459 | `f0ede3df8816753f7df13f7b51d767e82254d77d9466206bb04422bf55e591b6` |
| `h_logs/h_20261025_183424.t.bin` | 1436696 | `c2829b480170cbf93db98da4a07f7b61c15a41adc0670c88e0d19d63e936a9a1` |
| `h_logs/h_20261025_205350.amp.bin` | 13816960 | `05ad5fec4f471d0f126cf52761dc1e1f01b4bc41f028253a57911c25c60f9cda` |
| `h_logs/h_20261025_205350.flag.bin` | 86356 | `c4d2c66e71ad2fa91dd449f4526c4db975257bd0c7be9c812176db9717a729d9` |
| `h_logs/h_20261025_205350.h.bin` | 13816960 | `3a5667c1bcb654df1e7e06c56dc9a911a1bf10cec4e5963cd4c55361911fd517` |
| `h_logs/h_20261025_205350.json` | 455 | `b4741c818b258245713aa98f50b9c202b578cd11f44cc9a0aa592a6bb26a29c3` |
| `h_logs/h_20261025_205350.t.bin` | 690848 | `6c5b9b4636c8ae33e85a06c25957bc769bd3dacbe1529f3bc735a0c58aab3625` |
| `h_logs/h_20261025_212246.amp.bin` | 21144960 | `96beef5720954711032e2edcfc9e7bff75e5aff2de62bf81a3e23f1c96e11f75` |
| `h_logs/h_20261025_212246.flag.bin` | 132156 | `c5dfdc5655f55af5a9b15ba62494ab5352a07f3b964a939fb63271855b5302bf` |
| `h_logs/h_20261025_212246.h.bin` | 21144960 | `618cbd5a1f41a1edd47c5e87d0c87cc0d2e3703940c52217353dfb737c0ce0c4` |
| `h_logs/h_20261025_212246.json` | 456 | `f507e8de68ecea2f94baadb37dd3af590d62f5347bf713602041d2e7316f0f43` |
| `h_logs/h_20261025_212246.t.bin` | 1057248 | `f2e6f488970d1e256ff3865ce34d3e09aee196d6b70b2af26b9113653c274294` |
| `probe_runs/20261025_191922/keep.csv` | 3246 | `4befad25c75bd764a0beb5f57865682c8efe88136eaf71b32c2fd3630a1e29ba` |
| `probe_runs/20261025_191922/newc.csv` | 3129 | `3e8b5d1b2beecfb6dddfb05c596d6fc3dc0fa60dc14034a639864bedb760d25f` |
| `probe_runs/20261025_191922/ping.log` | 39508 | `df2471ebe28671ff75cdf45f0df394b7b56ecc70a493a1d6461698e09ddba14e` |
| `probe_runs/20261025_193213_page/console.log` | 59091 | `5dfe2790428eddb652a34a3fda3ed3bda0aea1a4fa2826c17fde1a9d6e6634ee` |
| `probe_runs/20261025_193213_page/keep.csv` | 10358 | `1c267d7f7419eecccf6b1d01eac327cba1490939e061a303ac44c453e589bc4d` |
| `probe_runs/20261025_193213_page/newc.csv` | 10061 | `8ae0aa8b2bbab73a519d4bc2c95780b942e29ffd1bc5b8fae24c66efd23fdb0d` |
| `probe_runs/20261025_193213_page/ping.log` | 93853 | `3ac72450eb1bc18fe8fe67a3ce17808a60a04d2e19567c469ee1f4666ca7454c` |
| `probe_runs/20261025_193213_page/probe.out` | 122 | `54572f8df0a0047ed295831b93f8656333b6f1677ed0211d272c3739b8d35521` |
| `probe_runs/20261025_193213_page/requests.csv` | 57387 | `b33bc4ff21ddb31154ea8a85aefc83dcf75e30bb180718b631633795e35bab5b` |
| `probe_runs/20261025_193213_page/state.csv` | 32052 | `7a5c761fcf0cb74fc28a57ec643d2bd0bee91fdd7f627304c7696594e40c597c` |
| `probe_runs/20261025_194533_page/console.log` | 41007 | `0ffb22005ce850aa76e889448ae4f634c0894f61da3b5244bb753b137474d16f` |
| `probe_runs/20261025_194533_page/keep.csv` | 9452 | `acbf831b36ffcd195efe8900b4ff9e80e7a42451fcab6a4cde9fb03908b59a7d` |
| `probe_runs/20261025_194533_page/newc.csv` | 8045 | `4dc3f1e493898bf22b37db42b440da0aa9c28a295b5762627f713ee9a9e8c033` |
| `probe_runs/20261025_194533_page/ping.log` | 79790 | `8f64eeef829b3df09f21cd2fb56562d16008a060e4def277c88805b74cfccf03` |
| `probe_runs/20261025_194533_page/probe.out` | 122 | `a13c85bb9ae957575125eceb0be90ede08e6f2af314277dc387fb2478d15a7f7` |
| `probe_runs/20261025_194533_page/requests.csv` | 36776 | `85e9ab993565d1ebeb52757ee29cc3f22749b9011b3604e1e25950da04055e74` |
| `probe_runs/20261025_194533_page/state.csv` | 24729 | `7b833c86176ebf8740ec3a5eac4ca753e386249e001bf3b48cbd186c45f3ad08` |
| `probe_runs/20261025_195711_page/console.log` | 46244 | `627b82c246de9060a00f987c3ebff1a95d0b5a63279d4e3d677f659e3e313dd4` |
| `probe_runs/20261025_195711_page/keep.csv` | 9066 | `008deda50fd6605ad6bf366ed606f79235faf14e68d97139d9908baedea54d0d` |
| `probe_runs/20261025_195711_page/newc.csv` | 8699 | `fa72da4ff5a0cb448c49fbdbc8b10784e52e884d3e86b00bb5cce1db8e4ca235` |
| `probe_runs/20261025_195711_page/ntg_states.csv` | 10512 | `6f4f2904832e34167b507a84a0a576abbc1e5cb89172a6d65579a199c7ccbc6f` |
| `probe_runs/20261025_195711_page/ping.log` | 79237 | `c118db9eccc9d16e9d17e2c45d00a3c5165b84fd1d88de23dc7bea25a441c19e` |
| `probe_runs/20261025_195711_page/probe.out` | 122 | `32d1236fa174e03e35eaed3d2f132e69b92763db05209d85d70dc1811dc5b250` |
| `probe_runs/20261025_195711_page/requests.csv` | 51933 | `6a855b3587256ccb894e6c6340c15750de419c94d304a8b547368076cb8e7956` |
| `probe_runs/20261025_195711_page/state.csv` | 28608 | `722949c395a9f27bb7ec59a461c36af307fc76d3a083dd223074c15c4f007121` |
| `probe_runs/20261025_201120_page/console.log` | 76650 | `c8e6fa4cf7686c779a846b3da9fcf1f6ce3b349052e5c84d64ea13695d9f256d` |
| `probe_runs/20261025_201120_page/keep.csv` | 16951 | `fc26941aa7585d9da1ff1b950b407be01074c5a3aabfebabeb4fed0a901adf5c` |
| `probe_runs/20261025_201120_page/newc.csv` | 15538 | `7e4b3126eb43c7688ef309797d6b1d55112861df8a77d0667f990516171ae222` |
| `probe_runs/20261025_201120_page/ntg_states.csv` | 32654 | `31ffe3159570b6f07ee93d0e650960439cea3a1c7f7ad9d9bf3fc033df17ae6f` |
| `probe_runs/20261025_201120_page/ping.log` | 143917 | `59fabe432fe31c4256cda3f323d6c734a4570c771d3d68ff2a353ea7246316ae` |
| `probe_runs/20261025_201120_page/probe.out` | 122 | `875d28e5e30f372964cc66da94064b54edcfb82de2956c4edcc2bdf55d4773f5` |
| `probe_runs/20261025_201120_page/requests.csv` | 87432 | `fb93b23a5e345d233a21a2dd7fe5676d2496a265c1151976058b1f97ee6d85fc` |
| `probe_runs/20261025_201120_page/state.csv` | 50769 | `f9f086409c9fd90ad76d852e7f6af9c43da97468a291fdb6866093aec19260dc` |
| `sessions_raw/dataplane_20261025_205053.csv` | 390186 | `5bd387b952f7fb2434e4858b01f001435541f558dc951833d056f03890532bf0` |
| `sessions_raw/dataplane_20261025_215043.csv` | 118372 | `f9d44972e821032e55cc13d0b40ed099fe189c76c85d305d989a81e891d3d1c7` |
| `sessions_raw/isac_20261025_181146_p001.npz` | 22154242 | `c14c34bfca16b4ca12f5cf80987de560a3b5d3c8bf2aaaae489cc9ad5061be74` |
| `sessions_raw/isac_20261025_181146_p002.npz` | 139657 | `44ffed6afeeaeb5733f0619d59e654544bad7deefcffeac45fa4e4c3b9638306` |
| `sessions_raw/isac_20261025_205053_p001.npz` | 16600019 | `f9f6b0eef1d8b7d5d1afa2b349664757895c3bab73804c9b1a97a5349d7679ce` |
| `sessions_raw/isac_20261025_215043_p001.npz` | 5555256 | `81d4752d5aec631981205e35c6cf1d4891a56817fcbb9cfc4b170ae2f6e5c860` |
| `sessions_raw/ntg_20261025_181146.csv` | 11723633 | `57498a1c9e6fc122a3afcdc960f38d3662895cbdf0f1793e2e772f791dc943e6` |
| `sessions_raw/ntg_20261025_181146.log` | 809171 | `e80f44e6f2974b038d94c19f008decfd88638454dc2b7b46771f7cfb0e6198ab` |
| `sessions_raw/ntg_20261025_205053.csv` | 3742266 | `0d14a8fba9b1c88b84774a59806e7cdf63b8ee25cbba691096d19a67a240d819` |
| `sessions_raw/ntg_20261025_205053.log` | 259613 | `9c2aa7dd7d02d4403c1dfce9385385cabf3675445887d4c750de089d40046d74` |
| `sessions_raw/ntg_20261025_215043.csv` | 1638056 | `9719607f2d4036a5a0ab24abe829309e554d18796c0f49c3f4b99a97716155dc` |
| `sessions_raw/ntg_20261025_215043.log` | 93031 | `d79ae19bb04c513b4e6e042be9cf8b00b4c6b1d3970dbe0a126ad4517a1e0ae7` |
| `sessions_raw/ntgstate_20261025_205053.csv` | 255051 | `521b72d4953321ef90105a3846f8501064c1cff161b1505674d79297ef16a7de` |
| `sessions_raw/ntgstate_20261025_215043.csv` | 96002 | `0667c60c45e3be7c2ff0556a07e7c5301284b5cba352150c6547cc9db8722540` |
| `walk_schedule.csv` | 3545 | `2e8c33c48a43f20a7bb7b76e2506240780f8ed3b94b954417cd9c51a09d2481c` |
