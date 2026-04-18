#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
#  AutoTrader Bot — AWS EC2 Setup Script
#  Target: Ubuntu 22.04 LTS (t2.micro or larger)
#
#  This script sets up a fresh EC2 instance to run the trading bot.
#  Run it once after SSH-ing into your new instance:
#
#    chmod +x aws-setup.sh
#    ./aws-setup.sh
#
#  After the script finishes:
#    1. Edit .env with your real API keys
#    2. Upload your ML model file (see instructions below)
#    3. Start the bot with: pm2 start ecosystem.config.js
#
#  WHAT THIS SCRIPT DOES:
#    Step 1 — Install system packages (Node.js 20, Python 3.11, git)
#    Step 2 — Clone the repo from GitHub
#    Step 3 — Install npm dependencies
#    Step 4 — Create Python venv and install ML dependencies
#    Step 5 — Create .env file with placeholder API keys
#    Step 6 — Remind to upload the ML model (not in git)
#    Step 7 — Install PM2 globally for process management
#    Step 8 — Generate PM2 ecosystem config
#    Step 9 — Configure PM2 to start on system boot
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

APP_DIR="$HOME/AutoTraderBot"
REPO_URL="https://github.com/michaelyang-dev/AutoTraderBot.git"

echo ""
echo "══════════════════════════════════════════"
echo "  AutoTrader Bot — EC2 Setup"
echo "══════════════════════════════════════════"
echo ""

# ── Step 1: Install system packages ──────────────────────────────────
echo "📦 Step 1/9: Installing system packages..."

sudo apt-get update -y
sudo apt-get install -y curl git build-essential software-properties-common

# Node.js 20 LTS via NodeSource
if ! command -v node &> /dev/null || [[ "$(node -v)" != v20* ]]; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi
echo "   Node.js $(node -v)"
echo "   npm $(npm -v)"

# Python 3.11
if ! command -v python3.11 &> /dev/null; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -y
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi
echo "   Python $(python3.11 --version)"

# pip for Python 3.11
if ! python3.11 -m pip --version &> /dev/null; then
    curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.11
fi
echo "   pip $(python3.11 -m pip --version | awk '{print $2}')"

echo "✅ System packages installed."
echo ""

# ── Step 2: Clone the repo ──────────────────────────────────────────
echo "📂 Step 2/9: Cloning repository..."

if [ -d "$APP_DIR" ]; then
    echo "   Directory $APP_DIR already exists — pulling latest..."
    cd "$APP_DIR"
    git pull origin main
else
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

echo "✅ Repository ready at $APP_DIR"
echo ""

# ── Step 3: Install npm dependencies ────────────────────────────────
echo "📦 Step 3/9: Installing npm dependencies..."

cd "$APP_DIR"
npm install --production

echo "✅ npm dependencies installed."
echo ""

# ── Step 4: Python virtual environment + ML packages ────────────────
echo "🐍 Step 4/9: Setting up Python virtual environment..."

cd "$APP_DIR"
python3.11 -m venv ml_service/venv
source ml_service/venv/bin/activate
pip install --upgrade pip
pip install -r ml_service/requirements.txt
deactivate

echo "✅ Python venv created at ml_service/venv with all packages."
echo ""

# ── Step 5: Create .env file ────────────────────────────────────────
echo "🔑 Step 5/9: Creating .env template..."

ENV_FILE="$APP_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    echo "   .env already exists — skipping (won't overwrite your keys)."
else
    cat > "$ENV_FILE" << 'ENVEOF'
# ══════════════════════════════════════════
#  AutoTrader Bot — Environment Variables
#  Fill in your API keys below.
# ══════════════════════════════════════════

# Alpaca Paper Trading API (https://app.alpaca.markets → Paper Trading → API Keys)
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_SECRET_KEY=your_paper_secret_key_here

# Financial Modeling Prep (https://financialmodelingprep.com → API Key)
# Optional — earnings calendar will be disabled without this
FMP_API_KEY=your_fmp_api_key_here

# Email Notifications (optional — Gmail SMTP)
# Generate an App Password: Google Account → Security → 2-Step Verification → App Passwords
EMAIL_USER=
EMAIL_APP_PASSWORD=
ENVEOF
    echo "   Created $ENV_FILE — edit this with your real API keys!"
fi
echo ""

# ── Step 6: ML model file reminder ──────────────────────────────────
echo "🤖 Step 6/9: ML model file check..."

MODEL_FILE="$APP_DIR/ml_service/data/model.lgb"
if [ -f "$MODEL_FILE" ]; then
    echo "   ✅ ML model found at $MODEL_FILE"
else
    echo ""
    echo "   ⚠️  ML model NOT found at: $MODEL_FILE"
    echo ""
    echo "   The model file is not in git. Upload it from your local machine:"
    echo ""
    echo "   scp -i your-key.pem ml_service/data/model.lgb ubuntu@<EC2-IP>:~/AutoTraderBot/ml_service/data/model.lgb"
    echo ""
    echo "   The bot will still run without it (falls back to consensus engine),"
    echo "   but ML-driven trades won't work until the model is uploaded."
    echo ""
fi

# Ensure the data directory exists for the model and trade journal DB
mkdir -p "$APP_DIR/ml_service/data"

echo ""

# ── Step 7: Install PM2 ────────────────────────────────────────────
echo "⚙️  Step 7/9: Installing PM2..."

if ! command -v pm2 &> /dev/null; then
    sudo npm install -g pm2
fi
echo "   PM2 $(pm2 -v)"
echo "✅ PM2 installed."
echo ""

# ── Step 8: Create PM2 ecosystem config ─────────────────────────────
echo "📋 Step 8/9: Creating PM2 ecosystem config..."

cat > "$APP_DIR/ecosystem.config.js" << 'PM2EOF'
// PM2 Ecosystem Configuration — AutoTrader Bot
// Start:   pm2 start ecosystem.config.js
// Stop:    pm2 stop all
// Logs:    pm2 logs
// Monitor: pm2 monit

module.exports = {
  apps: [
    {
      name: "ml-signal-server",
      script: "ml_service/venv/bin/python3",
      args: "ml_service/signal_server.py",
      cwd: __dirname,
      interpreter: "none",           // don't wrap with node
      env: {
        PYTHONUNBUFFERED: "1",        // flush print() immediately to logs
      },
      watch: false,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,            // wait 5s between restarts
      log_date_format: "YYYY-MM-DD HH:mm:ss",
    },
    {
      name: "express-server",
      script: "server/index.js",
      cwd: __dirname,
      node_args: "--max-old-space-size=512",
      env: {
        NODE_ENV: "production",
        PORT: "3001",
      },
      watch: false,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 10000,     // wait 10s between restarts (gives ML server time to load)
      log_date_format: "YYYY-MM-DD HH:mm:ss",
    },
  ],
};
PM2EOF

echo "✅ ecosystem.config.js created."
echo ""

# ── Step 9: PM2 startup on boot ─────────────────────────────────────
echo "🔄 Step 9/9: Configuring PM2 startup on boot..."

# Generate the startup script (must run the output command as root)
pm2 startup systemd -u "$USER" --hp "$HOME" | tail -1 | sudo bash

echo "✅ PM2 will auto-start on system reboot."
echo ""

# ══════════════════════════════════════════════════════════════════════
#  DONE
# ══════════════════════════════════════════════════════════════════════

echo "══════════════════════════════════════════"
echo "  Setup Complete!"
echo "══════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Edit your API keys:"
echo "     nano $APP_DIR/.env"
echo ""
echo "  2. Upload your ML model (from your local machine):"
echo "     scp -i your-key.pem ml_service/data/model.lgb ubuntu@<EC2-IP>:~/AutoTraderBot/ml_service/data/model.lgb"
echo ""
echo "  3. Start the bot:"
echo "     cd $APP_DIR"
echo "     pm2 start ecosystem.config.js"
echo "     pm2 save"
echo ""
echo "  4. Check status:"
echo "     pm2 status"
echo "     pm2 logs"
echo ""
echo "  5. Access the dashboard:"
echo "     http://<EC2-PUBLIC-IP>:3001/api/health"
echo "     http://<EC2-PUBLIC-IP>:3001/api/trading-state"
echo ""
echo "  Useful PM2 commands:"
echo "     pm2 logs express-server    — Express/trading engine logs"
echo "     pm2 logs ml-signal-server  — ML signal server logs"
echo "     pm2 monit                  — live CPU/memory dashboard"
echo "     pm2 restart all            — restart both services"
echo "     pm2 stop all               — stop everything"
echo ""
