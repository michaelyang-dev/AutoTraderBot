#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
#  Setup Monthly Retrain Cron Job
#  Schedule: 1st of each month at 2:00 AM ET
#
#  Usage:
#    chmod +x scripts/setup_retrain_cron.sh
#    ./scripts/setup_retrain_cron.sh          # show cron entry (dry run)
#    ./scripts/setup_retrain_cron.sh --install # actually install cron
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
# Adjust these paths for your AWS deployment
REPO_DIR="${REPO_DIR:-/home/ubuntu/AutoTraderBot}"
VENV_PYTHON="${REPO_DIR}/.venv/bin/python"
LOG_DIR="${REPO_DIR}/logs"
RETRAIN_SCRIPT="${REPO_DIR}/ml_service/retrain.py"

# Cron schedule: minute hour day-of-month month day-of-week
# 0 2 1 * * = 2:00 AM on the 1st of every month
CRON_SCHEDULE="0 2 1 * *"

# ── Cron entry ───────────────────────────────────────────────────────
CRON_ENTRY="${CRON_SCHEDULE} cd ${REPO_DIR} && ${VENV_PYTHON} ${RETRAIN_SCRIPT} >> ${LOG_DIR}/retrain_cron.log 2>&1"

echo "══════════════════════════════════════════════════════════"
echo "  Monthly Retrain Cron Setup"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "  Schedule:  1st of month at 2:00 AM (server time)"
echo "  Script:    ${RETRAIN_SCRIPT}"
echo "  Log:       ${LOG_DIR}/retrain_cron.log"
echo "  Python:    ${VENV_PYTHON}"
echo ""
echo "  Cron entry:"
echo "  ${CRON_ENTRY}"
echo ""

# ── Ensure log directory exists ──────────────────────────────────────
mkdir -p "${LOG_DIR}"

if [[ "${1:-}" == "--install" ]]; then
    # Check if already installed
    if crontab -l 2>/dev/null | grep -q "retrain.py"; then
        echo "  WARNING: retrain cron entry already exists!"
        echo "  Current crontab:"
        crontab -l 2>/dev/null | grep "retrain"
        echo ""
        read -p "  Replace existing entry? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "  Aborted."
            exit 0
        fi
        # Remove old entry
        crontab -l 2>/dev/null | grep -v "retrain.py" | crontab -
    fi

    # Install new entry
    (crontab -l 2>/dev/null; echo "${CRON_ENTRY}") | crontab -
    echo "  INSTALLED. Verify with: crontab -l"
    echo ""
    echo "  To remove: crontab -e  (and delete the retrain line)"
    echo "  To test:   ${VENV_PYTHON} ${RETRAIN_SCRIPT} --dry-run --skip-data"
else
    echo "  DRY RUN — not installing."
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

# ── Log cleanup cron (optional) ──────────────────────────────────────
echo ""
echo "  Optional: auto-delete logs older than 12 months"
echo "  Add to crontab manually:"
echo "  0 3 1 * * find ${LOG_DIR} -name 'retrain_*.log' -mtime +365 -delete"
