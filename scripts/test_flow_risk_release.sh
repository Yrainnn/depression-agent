#!/usr/bin/env bash
set -euo pipefail
API="http://127.0.0.1:8080"
SID="flow_risk_release_1"

step() {
  echo -e "\n=== $1 ==="
}

step "首问"
curl -s -X POST "$API/dm/step" -H 'Content-Type: application/json' \
  -d "{\"sid\":\"$SID\",\"role\":\"user\"}" | jq .
sleep 0.3

step "触发高风险（示例）"
curl -s -X POST "$API/dm/step" -H 'Content-Type: application/json' \
  -d "{\"sid\":\"$SID\",\"role\":\"user\",\"text\":\"有时候会想自杀。\"}" | jq .
sleep 0.3

step "解除风险（明确否定+说明已安全）"
curl -s -X POST "$API/dm/step" -H 'Content-Type: application/json' \
  -d "{\"sid\":\"$SID\",\"role\":\"user\",\"text\":\"我现在已经安全，不需要紧急帮助，没有自杀想法。\"}" | jq .
sleep 0.3

step "继续下一题（验证能前进）"
curl -s -X POST "$API/dm/step" -H 'Content-Type: application/json' \
  -d "{\"sid\":\"$SID\",\"role\":\"user\",\"text\":\"最近入睡困难，上床后要40分钟才能睡着。\"}" | jq .
