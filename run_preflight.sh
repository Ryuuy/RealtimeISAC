#!/bin/bash
# 实验前 10 s 体检（增益 20 dB），不能和 run_rt.sh 同时跑（都要占 B210）。
cd "$(dirname "$0")" && exec /usr/bin/python3 preflight/preflight.py "$@"
