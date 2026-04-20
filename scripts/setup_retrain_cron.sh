#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
#  Setup Monthly Retrain + Weekly Dry-Run Cron Jobs
#
#  Monthly retrain:  1st of each month at 2:00 AM (live deploy)
#  Weekly dry-run:   Every Sunday at 3:00 AM (smoke test, no deploy)
#
#  Usage:
#    chmod +x scripts/setup_retrain_cron.sh
#    ./scripts/setup_retrain_cron.sh          # show cron entries (preview)
#    ./scripts/setup_retrain_cron.sh --install # actually install cron
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
# Adjust these paths for your AWS deployment
REPO_DIR="${REPO_DIR:-/home/ubuntu/AutoTraderBot}"
VENV_PYTHON="${REPO_DIR}/.venv/bin/python"
LOG_DIR="${REPO_DIR}/logs"
RETRAIN_SCRIPT="${REPO_DIR}/ml_service/retrain.py"

# ── Cron entries ─────────────────────────────────────────────────────
# Monthly production retrain: 1st of month at 2:00 AM
MONTHLY_ENTRY="0 2 1 * * cd ${REPO_DIR} && ${VENV_PYTHON} ${RETRAIN_SCRIPT} >> ${LOG_DIR}/retrain_cron.log 2>&1"

# Weekly dry-run smoke test: Sunday at 3:00 AM (skip data update for speed)
WEEKLY_ENTRY="0 3 * * 0 cd ${REPO_DIR} && ${VENV_PYTHON} ${RETRAIN_SCRIPT} --dry-run --skip-data >> ${LOG_DIR}/retrain_dryrun.log 2>&1"

# Log cleanup: 1st of month at 4:00 AM, delete logs older than 12 months
CLEANUP_ENTRY="0 4 1 * * find ${LOG_DIR} -name 'retrain_*.log' -mtime +365 -delete"

echo "══════════════════════════════════════════════════════════"
echo "  Retrain Cron Setup"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "  1. Monthly retrain (1st of month, 2 AM):"
echo "     ${MONTHLY_ENTRY}"
echo ""
echo "  2. Weekly dry-run (Sunday, 3 AM):"
echo "     ${WEEKLY_ENTRY}"
echo ""
echo "  3. Log cleanup (1st of month, 4 AM):"
echo "     ${CLEANUP_ENTRY}"
echo ""
echo "  Python: ${VENV_PYTHON}"
echo "  Logs:   ${LOG_DIR}/retrain_cron.log"
echo "          ${LOG_DIR}/retrain_dryrun.log"
echo ""

# ── Ensure log directory exists ──────────────────────────────────────
mkdir -p "${LOG_DIR}"

if [[ "${1:-}" == "--install" ]]; then
    # Remove any existing retrain entries
    if crontab -l 2>/dev/null | grep -q "retrain"; then
        echo "  Removing existing retrain cron entries ..."
        crontab -l 2>/dev/null | grep -v "retrain" | crontab -
    fi

    # Install all three entries
    (crontab -l 2>/dev/null; echo "${MONTHLY_ENTRY}"; echo "${WEEKLY_ENTRY}"; echo "${CLEANUP_ENTRY}") | crontab -
    echo "  INSTALLED 3 cron entries. Verify with: crontab -l"
    echo ""
    echo "  To remove all: crontab -l | grep -v retrain | crontab -"
    echo "  To test:       ${VENV_PYTHON} ${RETRAIN_SCRIPT} --dry-run --skip-data"
else
    echo "  PREVIEW — not installing."
    echo "  To install: $0 --install"
    echo ""
    echo "  Manual test commands:"
    echo "    # Quick test (small data, no deploy):"
    echo "    cd ${REPO_DIR}"
    echo "    ${VENV_PYTHON} ${RETRAIN_SCRIPT} --test --dry-run"
    echo ""
    echo "    # Full dry run (real data, no deploy):"
    echo "    ${VENV_PYTHON} ${RETRAIN_SCRIPT} --dry-run"
    echo ""
    echo "    # Full retrain + deploy:"
    echo "    ${VENV_PYTHON} ${RETRAIN_SCRIPT}"
fi
