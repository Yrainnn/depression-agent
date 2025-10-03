#!/usr/bin/env bash
set -euo pipefail

python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8080 --reload
