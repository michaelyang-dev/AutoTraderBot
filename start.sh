#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
#  start.sh — Launch ML signal server + JS trading bot together
#  Usage: ./start.sh
#  Stop:  Ctrl+C  (kills both processes cleanly)
# ══════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/.venv"

# ── Sanity checks ─────────────────────────────────────────────
if [ ! -f "$VENV/bin/python3" ]; then
  echo "ERROR: Python venv not found at $VENV"
  echo "       Run: python3 -m venv .venv && .venv/bin/pip install -r ml_service/requirements.txt"
  exit 1
fi

if [ ! -f "$SCRIPT_DIR/ml_service/signal_server.py" ]; then
  echo "ERROR: ml_service/signal_server.py not found"
  exit 1
fi

# ── Activate venv ─────────────────────────────────────────────
source "$VENV/bin/activate"

# ── Cleanup on exit ───────────────────────────────────────────
ML_PID=""
cleanup() {
  echo ""
  echo "Shutting down..."
  if [ -n "$ML_PID" ] && kill -0 "$ML_PID" 2>/dev/null; then
    echo "  Stopping ML signal server (PID $ML_PID)..."
    kill "$ML_PID" 2>/dev/null
    wait "$ML_PID" 2>/dev/null || true
  fi
  echo "  Done."
  exit 0
}
trap cleanup INT TERM

# ── Start ML signal server in background ──────────────────────
echo "Starting ML signal server on http://localhost:5001 ..."
python3 "$SCRIPT_DIR/ml_service/signal_server.py" &
ML_PID=$!

# ── Wait for ML server to be ready ────────────────────────────
echo "Waiting for ML server to initialize..."
sleep 3

# Confirm it's still running
if ! kill -0 "$ML_PID" 2>/dev/null; then
  echo "ERROR: ML signal server failed to start. Check ml_service/signal_server.py"
  exit 1
fi
echo "  ML server running (PID $ML_PID)"

# ── Start JS trading bot in foreground ────────────────────────
echo "Starting JS trading bot (npm start)..."
echo "Press Ctrl+C to stop both servers."
echo "─────────────────────────────────────────────────────────"

npm start

# npm start exited on its own — clean up ML server
cleanup
