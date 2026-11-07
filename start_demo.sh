#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${1:---replay}"
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"
case "$MODE" in
  --replay) export DEMO_REPLAY=1 ;;
  --live) export DEMO_REPLAY=0 ;;
  *) printf 'Usage: %s [--replay|--live]\n' "$0" >&2; exit 2 ;;
esac

PYTHON_CMD="${PYTHON:-python3}"
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  printf 'Python 3.11 or newer is required. Set PYTHON to its executable if needed.\n' >&2
  exit 1
fi
if ! "$PYTHON_CMD" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  printf 'Python 3.11 or newer is required for the pinned dependency set.\n' >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_CMD" -m venv .venv
fi
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.lock.txt

.venv/bin/python - "$API_PORT" "$UI_PORT" <<'PY'
import socket
import sys

ports = [int(value) for value in sys.argv[1:]]
if len(set(ports)) != len(ports):
    raise SystemExit("API_PORT and UI_PORT must be different ports.")
for port in ports:
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))
    except OSError:
        raise SystemExit(f"Local port {port} is already in use. Stop the existing service or choose API_PORT/UI_PORT.")
    finally:
        probe.close()
PY

if ! command -v java >/dev/null 2>&1; then
  printf 'Java 11 or newer is required for the money-mining demo. Install Java, then retry.\n' >&2
  exit 1
fi
if [[ ! -f tools/spmf.jar ]]; then
  printf 'Missing tools/spmf.jar. Download it using the instructions in tools/README.txt, then retry.\n' >&2
  exit 1
fi

if [[ ! -f data/processed/clean_transactions.csv ]]; then
  if [[ ! -f data/raw/online_retail.xlsx && ! -f 'data/raw/Online Retail.xlsx' \
        && ! -f data/raw/online_retail.csv && ! -f data/raw/OnlineRetail.csv ]]; then
    printf 'Missing UCI Online Retail data. Download it from https://archive.ics.uci.edu/dataset/352/online+retail\n' >&2
    printf 'Save the workbook as data/raw/online_retail.xlsx, then run this script again.\n' >&2
    exit 1
  fi
  .venv/bin/python -m src.prepare_data
fi

if [[ "$MODE" == "--live" ]]; then
  if ! .venv/bin/python - <<'PY'
from dotenv import load_dotenv
from src.core.errors import LLMUnavailable
from src.llm.client import create_llm_client

load_dotenv(".env")
try:
    client = create_llm_client()
    print(f"Live model ready: {client.model}")
except LLMUnavailable as error:
    raise SystemExit(str(error))
PY
  then
    printf 'Live mode needs the configured model available. Check LLM_PROVIDER/OLLAMA_MODEL in .env, or use ./start_demo.sh for replay mode.\n' >&2
    exit 1
  fi
fi

export API_URL="http://127.0.0.1:${API_PORT}"
API_LOG="$(mktemp -t pattern-mining-api.XXXXXX)"
.venv/bin/python -m uvicorn api.main:create_app --factory --host 127.0.0.1 --port "$API_PORT" --log-level warning >"$API_LOG" 2>&1 &
API_PID=$!
cleanup() {
  kill "$API_PID" 2>/dev/null || true
  wait "$API_PID" 2>/dev/null || true
  rm -f "$API_LOG"
}
trap cleanup EXIT INT TERM

ready=0
for _ in {1..60}; do
  if curl --silent --fail "$API_URL/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    printf 'The API failed to start:\n' >&2
    cat "$API_LOG" >&2
    exit 1
  fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then
  printf 'The API did not become ready in 60 seconds:\n' >&2
  cat "$API_LOG" >&2
  exit 1
fi

printf 'Starting Streamlit at http://127.0.0.1:%s (%s mode). Press Ctrl+C to stop both services.\n' "$UI_PORT" "${MODE#--}"
.venv/bin/python -m streamlit run ui/app.py --server.address 127.0.0.1 --server.port "$UI_PORT"
