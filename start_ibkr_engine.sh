#!/bin/bash
cd /home/ubuntu/AutoTraderBot/ml_service
export $(cat /home/ubuntu/AutoTraderBot/.env | xargs)
exec /home/ubuntu/AutoTraderBot/ml_service/venv/bin/python3 ibkr_engine.py
