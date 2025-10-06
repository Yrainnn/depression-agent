#!/usr/bin/env bash
set -euo pipefail
API="http://127.0.0.1:8080"
SID="flow_ok_1"

step() {
  echo -e "\n=== $1 ==="
}

step "首问"
curl -s -X POST "$API/dm/step" -H 'Content-Type: application/json' \
  -d "{\"sid\":\"$SID\",\"role\":\"user\"}" | jq .
sleep 0.3

step "item1"
curl -s -X POST "$API/dm/step" -H 'Content-Type: application/json' \
  -d "{\"sid\":\"$SID\",\"role\":\"user\",\"text\":\"最近心情不好，每周三四天，持续半天到一天。\"}" | jq .
sleep 0.3

step "item2"
curl -s -X POST "$API/dm/step" -H 'Content-Type: application/json' \
  -d "{\"sid\":\"$SID\",\"role\":\"user\",\"text\":\"对原本喜欢的事情兴趣下降，多数天都提不起劲。\"}" | jq .
sleep 0.3

step "item3（明确否定）"
curl -s -X POST "$API/dm/step" -H 'Content-Type: application/json' \
  -d "{\"sid\":\"$SID\",\"role\":\"user\",\"text\":\"没有自杀想法，也不会做伤害自己的事。\"}" | jq .
