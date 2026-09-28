#!/usr/bin/env bash
# run_extreme_100tps.sh — 1B ternar modelni bitta GPU'da 100+ tok/s bilan
# chat rejimida ishga tushirish (ryzen-ai7 / RTX 5060).
#
# Ishlatish (node'da):
#   bash run_extreme_100tps.sh start    # bridge + fast engine
#   bash run_extreme_100tps.sh stop
#   bash run_extreme_100tps.sh status
set -u
TOKEN="${RELAY_TOKEN:-}"
SERVER="${RELAY_SERVER:-https://ozodbeks-macbook-pro.tail91a187.ts.net}"
BIN="${BT_BIN:-$HOME/bt/bin}"
NAME="${FAST_NAME:-fastchat}"        # inbox nomi
NEXT_INBOX="${FAST_NEXT:-chat}"      # natijalar qayerga (chat server)
LAYERS_D="${D:-2048}"
PARAMS="${PARAMS:-1000000000}"
TOKENS="${TOKENS:-12}"

case "${1:-status}" in
  start)
    setsid nohup python3 "$BIN/relay_bridge.py" \
      --name "$NAME" --next "$NEXT_INBOX" \
      --token "$TOKEN" --server "$SERVER" \
      --engine "$BIN/fast_chat_engine" \
      --engine-args "--d $LAYERS_D --params $PARAMS --tokens $TOKENS" \
      --log /tmp/d70b_fastchat.log >/tmp/d70b_fastchat_bridge.log 2>&1 < /dev/null &
    sleep 2
    echo "fast engine start: $NAME -> $NEXT_INBOX (100+ tok/s target)"
    ;;
  stop)
    pkill -f 'fast_chat_engi[n]e' 2>/dev/null
    pkill -f "relay_bridge.py --name $NAME" 2>/dev/null
    echo "to'xtatildi"
    ;;
  status)
    pgrep -f 'fast_chat_engi[n]e' >/dev/null && echo "engine ACTIVE" || echo "engine YOQ"
    tail -2 /tmp/d70b_fastchat.log 2>/dev/null
    ;;
  *)
    echo "ishlatish: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
