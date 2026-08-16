#!/bin/bash
# SERIAL job runner, low priority. Waits for any in-flight experiment first.
# Five concurrent 26yr jobs load ~30GB+ of pickles (each expands ~6x in RAM as nested dicts)
# and pin five cores -- that is what crashed the laptop twice. One at a time only.
cd "$(dirname "$0")/.."
while pgrep -f "python3 research/EXP" > /dev/null; do sleep 10; done
while IFS='|' read -r script arg out; do
  [ -z "$script" ] && continue
  # MEMORY GUARD. The 26yr universe expands to ~45GB resident (6,300 dates x 3,600 symbols
  # x 29 features held as nested Python dicts -- ~22.7M dicts). On a 64GB machine that is one
  # job and no more. Wait for the previous job's pages to be reclaimed before starting.
  for _i in $(seq 1 30); do
    FREEG=$(vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} END{gsub(/\./,"",f); gsub(/\./,"",i); print int((f+i)*16384/1073741824)}')
    [ "${FREEG:-0}" -ge 20 ] && break
    echo "    (waiting for memory: ${FREEG}GB available)"; sleep 20
  done
  echo "=== $(date +%H:%M:%S) START $script $arg  (avail ${FREEG}GB)"
  nice -n 15 python3 "research/$script" $arg > "research/$out.out" 2>&1 </dev/null
  echo "=== $(date +%H:%M:%S) DONE  $script $arg (exit $?)"
done < research/queue.txt
echo "=== $(date +%H:%M:%S) QUEUE EMPTY"
