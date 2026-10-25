#!/bin/bash
# Wait for the ladder job (PID given as $1), write a status report, then power off.
# Cancel the shutdown with:  touch /home/usrpb210/dash_src/NO_SHUTDOWN   (or kill this script)
ROOT=/home/usrpb210/DASH/DASH_html_old-main
LOG=/home/usrpb210/dash_src/logs/final_status.txt
tail --pid="$1" -f /dev/null
{
  echo "finished: $(date)"
  echo "mpd files:"; ls -l $ROOT/tos_ibmff_mixed_1080p*.mpd 2>&1
  echo "segments per rung (expect 367):"
  for d in $ROOT/tos_2s_1080p/tos_2s_*kbit; do echo "$(basename $d) $(ls $d | grep -c '\.m4s$')"; done | sort -t_ -k3 -n
  echo "bandwidths in MPD:"; grep -o 'width="[0-9]*" height="[0-9]*" startWithSAP="1" bandwidth="[0-9]*"' $ROOT/tos_ibmff_mixed_1080p-public-access.mpd 2>&1
  du -sh $ROOT/tos_2s_1080p
  df -h /home/usrpb210 | tail -1
} > "$LOG" 2>&1
sync
if [ -e /home/usrpb210/dash_src/NO_SHUTDOWN ]; then echo "NO_SHUTDOWN present, staying on" >> "$LOG"; exit 0; fi
echo "powering off in 180 s: $(date)" >> "$LOG"
sleep 180
if [ -e /home/usrpb210/dash_src/NO_SHUTDOWN ]; then echo "cancelled" >> "$LOG"; exit 0; fi
systemctl poweroff >> "$LOG" 2>&1 || systemctl poweroff -i >> "$LOG" 2>&1
