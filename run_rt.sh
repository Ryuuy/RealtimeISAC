#!/bin/bash
# 终端 B：实时多普勒（增益 20 dB）+ 推给 bridge + 保存每帧 dec/幅度。用 /usr/bin/python3。
# 用法:  ./run_rt.sh [秒数，默认 3600]
cd "$(dirname "$0")/realtime" && exec /usr/bin/python3 rt_main.py --duration "${1:-3600}" --gain 20 --isac --save-h
