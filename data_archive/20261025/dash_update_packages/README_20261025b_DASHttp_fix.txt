DASH_20261025b_DASHttp_fix.zip —— 只含 dash-js/DASHttp.js 一个文件（在已经装好 DASH_20261025_code.zip 的基础上再覆盖）

改了什么（DASHttp.js 里三个常量）：
  SEGMENT_FETCH_TIMEOUT_MS         3000 -> 8000     分片请求超时（XHR.timeout：从 send 到整个响应收完的总时长，不是"有数据就续命"）
  SEGMENT_RETRY_INITIAL_DELAY_MS    500 -> 200      失败后重试间隔
  SEGMENT_RETRY_MAX_DELAY_MS       3000 -> 200      INITIAL=MAX，不再 0.5/1/2/3 秒翻倍
原因：链路恢复后第一个能到的请求第一字节要 ~2.6–3.0 s，再下 1–2 s，被 3 s 超时截断，再加上翻倍的重试间隔，
页面比数据面慢 6–13 s 才下到第一个分片。

怎么装（Windows）：把 zip 解压到 DASH_html_old-main（覆盖 dash-js/DASHttp.js），浏览器 Ctrl+F5 强刷。
不需要重启 server，不改 server 任何文件。
检查：F12 控制台在失败重试时会打印  "retry in 200ms"（以前是 500/1000/2000/3000ms）。
