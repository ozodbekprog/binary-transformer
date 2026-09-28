#!/usr/bin/env bash
# ===========================================================================
# start_70b_cluster.sh — Phase 2: 70B-class ternar modelni 4 node pipeline
# bo'ylab ishga tushirish (markaz relay mailbox orqali, NAT-do'st).
#
# Ishlatish (marksaz mashinadan — farm admin):
#   bash scripts/start_70b_cluster.sh start     # barcha node'da bridge+engine
#   bash scripts/start_70b_cluster.sh status    # holat va loglar
#   bash scripts/start_70b_cluster.sh run       # controller generatsiyasi
#   bash scripts/start_70b_cluster.sh stop      # hammasini to'xtatish
#
# Asror'ni keyin qo'shish (u tayyor turibdi):
#   WITH_ASROR=1 bash scripts/start_70b_cluster.sh start
#
# Talab: har node'da ~/bt/bin/{distributed_70b_engine,relay_bridge.py}
# ===========================================================================
set -u

CR="${CR:-python3 $HOME/.local/bin/centerrun}"
TOKEN="${RELAY_TOKEN:-}"
SERVER="${RELAY_SERVER:-https://ozodbeks-macbook-pro.tail91a187.ts.net}"
SEED="${SEED:-42}"
TOKENS="${TOKENS:-4}"
D="${D:-8192}"

# name|layers|final|next
# WITH_ASROR=1 → 4 node (asror-final bilan, kelajakda ishlatish uchun)
# WITH_ASROR=0 → 3 node (asror hozircha ishlamaydi, 2026-09-28)
WITH_ASROR="${WITH_ASROR:-0}"
if [ "$WITH_ASROR" = "1" ]; then
NODES=(
  "Ozodbeks-MBP|0-11|0|ryzen-ai7"
  "ryzen-ai7|12-43|0|gtx1650-pc"
  "gtx1650-pc|44-63|0|asror-fedora"
  "asror-fedora|64-79|1|controller"
)
else
NODES=(
  "Ozodbeks-MBP|0-25|0|ryzen-ai7"
  "ryzen-ai7|26-59|0|gtx1650-pc"
  "gtx1650-pc|60-79|1|controller"
)
fi

LOCAL_HOST="$(hostname)"

run_on() {  # host command timeout
  local host="$1" cmd="$2" tmo="${3:-40}"
  if [ "$host" = "$LOCAL_HOST" ]; then
    bash -c "$cmd"
  else
    $CR "$cmd" "$host" "$tmo" 2>&1 | tail -n +2
  fi
}

start_one() {  # name layers final next
  local name="$1" layers="$2" final="$3" next="$4"
  local role="worker"; [ "$name" = "Ozodbeks-MBP" ] && role="controller"
  local extra=""
  [ "$final" = "1" ] && extra="--final"
  local eargs="--role $role --layers $layers --transport stdio --d $D --seed $SEED --threads 0 $extra --node-name $name"
  [ "$role" = "controller" ] && eargs="$eargs --tokens $TOKENS"
  local inbox="$name"; [ "$name" = "Ozodbeks-MBP" ] && inbox="controller"
  local bargs="--name $inbox --next $next --token $TOKEN --server $SERVER --engine \$HOME/bt/bin/distributed_70b_engine --engine-args \"$eargs\" --log /tmp/d70b_$name.log"
  local cmd="pkill -f 'relay_bridge[.]py --name $name' 2>/dev/null; cd \$HOME/bt/bin && nohup python3 relay_bridge.py $bargs >/tmp/d70b_${name}_bridge.log 2>&1 & sleep 1; echo started-$name"
  echo "== $name ($role $layers -> $next)"
  run_on "$name" "$cmd" 40
}

case "${1:-status}" in
  start)
    # Avval oxirgi node, keyin orqaga qarab (controller eng oxirida)
    for ((i=${#NODES[@]}-1; i>=0; i--)); do
      IFS='|' read -r n l f x <<<"${NODES[$i]}"
      start_one "$n" "$l" "$f" "$x"
      sleep 1
    done
    echo "start: bridge'lar ishga tushdi. Loglar: /tmp/d70b_<node>.log"
    ;;
  status)
    for n in "${NODES[@]}"; do
      IFS='|' read -r name _ _ _ <<<"$n"
      echo "===== $name"
      run_on "$name" "pgrep -f 'relay_bridge[.]py' >/dev/null && echo bridge-ACTIVE || echo bridge-YOQ; tail -3 /tmp/d70b_$name.log 2>/dev/null" 30
    done
    ;;
  run)
    echo "controller generatsiyasi kutilmoqda (log: /tmp/d70b_Ozodbeks-MBP.log)..."
    for _ in $(seq 1 60); do
      out=$(run_on "Ozodbeks-MBP" "grep -h PIPELINE-RESULT /tmp/d70b_Ozodbeks-MBP.log 2>/dev/null | tail -1" 30)
      if [ -n "$out" ]; then echo "$out"; exit 0; fi
      sleep 10
    done
    echo "NATIJA TOPILMADI — loglarni tekshiring" >&2
    exit 1
    ;;
  stop)
    for n in "${NODES[@]}"; do
      IFS='|' read -r name _ _ _ <<<"$n"
      run_on "$name" "pkill -f 'relay_bridge[.]py' 2>/dev/null; pkill -f 'distributed_70b[_]engine' 2>/dev/null; echo stopped-$name" 30
    done
    ;;
  *)
    echo "ishlatish: $0 {start|status|run|stop}" >&2
    exit 2
    ;;
esac
