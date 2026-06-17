#!/bin/bash
cd ~/AutoTraderBot
git pull origin main
npm install --production
pm2 restart all
echo "✅ Updated and restarted"
