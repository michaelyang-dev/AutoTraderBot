#!/bin/bash
cd /home/ubuntu/AutoTraderBot/ml_service
# 2026-09-22: export .env before python starts — signal_server imports the strategy module (which reads
# MOM_EQUAL_WEIGHT / PROD_BULL_WEIGHTS at import) BEFORE its own load_dotenv(), so a late load is a silent no-op.
export $(cat /home/ubuntu/AutoTraderBot/.env | xargs)
exec /home/ubuntu/AutoTraderBot/ml_service/venv/bin/python3 signal_server.py
