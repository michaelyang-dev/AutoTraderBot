#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f "python3 research/EXP" > /dev/null; do sleep 15; done
while IFS='|' read -r script arg out; do
  [ -z "$script" ] && continue
  echo "=== $(date +%H:%M:%S) START $script $arg"
  nice -n 15 python3 "research/$script" $arg > "research/$out.out" 2>&1 </dev/null
  echo "=== $(date +%H:%M:%S) DONE  $script $arg"
done < research/queue.txt
echo "=== ALL DONE"
