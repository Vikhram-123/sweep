#!/bin/zsh
# Run jobs one at a time. Usage: scripts/queue.sh <jobfile> <log>
# Before each job: wait (no time limit) until no other compat job is running, then wait up to
# 20 min for >= 30% free memory. Each line of <jobfile> is a shell command; failures are logged
# and the queue moves on.
cd "${0:A:h}/.."
jobs_file=$1; log=$2
wait_turn() {
  local n=0
  while pgrep -f "compat\.(train|finetune|embed_items|embed_text|baselines|leakage|analyze)" >/dev/null; do
    sleep 15; n=$((n + 1))
    (( n % 40 == 0 )) && echo "$(date +%T) waiting for another job to finish" >> $log
  done
  for i in {1..120}; do
    [ $(sysctl -n kern.memorystatus_level) -ge 30 ] && return
    sleep 10
  done
  echo "$(date +%T) WARN low free memory ($(sysctl -n kern.memorystatus_level)%), starting anyway" >> $log
}
while IFS= read -r cmd; do
  [[ -z "$cmd" || "$cmd" == \#* ]] && continue
  wait_turn
  echo "$(date +%T) START $cmd   [free mem $(sysctl -n kern.memorystatus_level)%, $(sysctl -n vm.swapusage | awk '{print "swap used "$6}')]" >> $log
  eval "$cmd" >> $log 2>&1
  rc=$?
  echo "$(date +%T) END rc=$rc $cmd" >> $log
done < $jobs_file
echo "$(date +%T) QUEUE DONE" >> $log
