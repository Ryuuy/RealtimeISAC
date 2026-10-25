#!/bin/bash
# 终端 A：ISAC bridge + NTG logger（Master 192.168.1.101）+ 回车打点。用 miniconda 的 python3。
cd "$(dirname "$0")" && exec python3 isac_bridge.py "$@"
