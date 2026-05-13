#!/bin/bash
# Restart signal server after data refresh
pkill -f "signal_server.py" 2>/dev/null
sleep 5
cd "/Users/michaelslyanggmail.com/Downloads/auto-trader 2/ml_service"
nohup /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -u signal_server.py >> /tmp/signal_server_v11.log 2>&1 &
echo "Signal server restarted at $(date)"
