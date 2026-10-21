#!/bin/zsh
# Run jobs one at a time, waiting for free memory before each. Usage: scripts/queue.sh <jobfile> <log>
# Each line of <jobfile> is a shell command; failures are logged and the queue moves on.
cd "${0:A:h}/.."
jobs_file=$1; log=$2
wait_for_memory() {
  for i in {1..120}; do
    lvl=$(sysctl -n kern.memorystatus_level)
    pgrep -f "compat\.(train|embed_items|embed_text|baselines)" >/dev/null || { [ $lvl -ge 30 ] && return; }
    sleep 10
  done
  echo "$(date +%T) WARN memory gate timed out (level $lvl)" >> $log
}
while IFS= read -r cmd; do
  [[ -z "$cmd" || "$cmd" == \#* ]] && continue
  wait_for_memory
  echo "$(date +%T) START $cmd   [free mem $(sysctl -n kern.memorystatus_level)%, $(sysctl -n vm.swapusage | awk '{print "swap used "$6}')]" >> $log
  eval "$cmd" >> $log 2>&1
  rc=$?
  echo "$(date +%T) END rc=$rc $cmd" >> $log
done < $jobs_file
echo "$(date +%T) QUEUE DONE" >> $log
