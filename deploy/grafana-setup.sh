#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
#  AutoTrader — Grafana Dashboard Setup
#  Target: Ubuntu 22.04 (EC2 t3.large)
#
#  Installs Grafana OSS, configures the Infinity datasource plugin,
#  and provisions dashboards that query the trading bot's API.
#
#  Usage:
#    chmod +x deploy/grafana-setup.sh
#    ./deploy/grafana-setup.sh
#
#  After setup:
#    - Grafana runs on port 3000
#    - Default login: admin / autotrader (change on first login)
#    - Dashboards auto-provisioned under "AutoTrader" folder
#    - Access: http://<EC2_IP>:3000
#
#  Prerequisites:
#    - trading-bot running on port 3001 (serves /api/grafana/* endpoints)
#    - EC2 security group allows inbound TCP 3000
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

APP_DIR="${HOME}/AutoTraderBot"
GRAFANA_DASH_DIR="/etc/grafana/dashboards"
GRAFANA_PROV_DIR="/etc/grafana/provisioning"

echo ""
echo "══════════════════════════════════════════"
echo "  AutoTrader — Grafana Setup"
echo "══════════════════════════════════════════"
echo ""

# ── Step 1: Install Grafana OSS ─────────────────────────────────────
echo "Step 1/5: Installing Grafana OSS..."

if ! command -v grafana-server &> /dev/null; then
    # Add Grafana APT repository
    sudo apt-get install -y apt-transport-https software-properties-common wget
    sudo mkdir -p /etc/apt/keyrings/
    wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null
    echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
    sudo apt-get update
    sudo apt-get install -y grafana
    echo "  Grafana installed successfully"
else
    echo "  Grafana already installed, skipping"
fi

# ── Step 2: Install Infinity datasource plugin ──────────────────────
echo ""
echo "Step 2/5: Installing Grafana Infinity plugin..."

if ! sudo grafana-cli plugins ls 2>/dev/null | grep -q "yesoreyeram-infinity-datasource"; then
    sudo grafana-cli plugins install yesoreyeram-infinity-datasource
    echo "  Infinity plugin installed"
else
    echo "  Infinity plugin already installed, skipping"
fi

# ── Step 3: Configure Grafana ────────────────────────────────────────
echo ""
echo "Step 3/5: Configuring Grafana..."

# Set default admin password
sudo sed -i 's/^;admin_password = admin/admin_password = autotrader/' /etc/grafana/grafana.ini 2>/dev/null || true
# If the line doesn't have a semicolon prefix
sudo sed -i 's/^admin_password = admin$/admin_password = autotrader/' /etc/grafana/grafana.ini 2>/dev/null || true

# Allow embedding (for iframe if needed)
sudo sed -i 's/^;allow_embedding = false/allow_embedding = true/' /etc/grafana/grafana.ini 2>/dev/null || true

# Set org name
sudo sed -i 's/^;org_name = Main Org./org_name = AutoTrader/' /etc/grafana/grafana.ini 2>/dev/null || true

# Allow unsigned plugins (Infinity)
if ! grep -q "allow_loading_unsigned_plugins" /etc/grafana/grafana.ini; then
    sudo bash -c 'echo "" >> /etc/grafana/grafana.ini'
    sudo bash -c 'echo "[plugins]" >> /etc/grafana/grafana.ini'
    sudo bash -c 'echo "allow_loading_unsigned_plugins = yesoreyeram-infinity-datasource" >> /etc/grafana/grafana.ini'
fi

echo "  Grafana configured"

# ── Step 4: Provision datasources and dashboards ─────────────────────
echo ""
echo "Step 4/5: Provisioning datasources and dashboards..."

# Copy datasource provisioning
sudo cp "${APP_DIR}/deploy/grafana/provisioning/datasources/autotrader.yml" \
        "${GRAFANA_PROV_DIR}/datasources/autotrader.yml"

# Copy dashboard provisioning config
sudo cp "${APP_DIR}/deploy/grafana/provisioning/dashboards/autotrader.yml" \
        "${GRAFANA_PROV_DIR}/dashboards/autotrader.yml"

# Copy dashboard JSON files
sudo mkdir -p "${GRAFANA_DASH_DIR}"
sudo cp "${APP_DIR}/deploy/grafana/dashboards/"*.json "${GRAFANA_DASH_DIR}/"

# Fix permissions
sudo chown -R grafana:grafana "${GRAFANA_PROV_DIR}" "${GRAFANA_DASH_DIR}"

echo "  Datasources and dashboards provisioned"

# ── Step 5: Start Grafana ────────────────────────────────────────────
echo ""
echo "Step 5/5: Starting Grafana..."

sudo systemctl daemon-reload
sudo systemctl enable grafana-server
sudo systemctl restart grafana-server

# Wait for Grafana to come up
echo "  Waiting for Grafana to start..."
for i in $(seq 1 15); do
    if curl -s http://localhost:3000/api/health | grep -q "ok"; then
        echo "  Grafana is running!"
        break
    fi
    sleep 2
done

echo ""
echo "══════════════════════════════════════════"
echo "  Grafana Setup Complete!"
echo "══════════════════════════════════════════"
echo ""
echo "  URL:      http://$(curl -s http://checkip.amazonaws.com 2>/dev/null || echo '<EC2_IP>'):3000"
echo "  Login:    admin / autotrader"
echo ""
echo "  Dashboards:"
echo "    - AutoTrader - Executive Summary"
echo "    - AutoTrader - Risk & System Health"
echo ""
echo "  IMPORTANT: Open port 3000 in your EC2 security group:"
echo "    aws ec2 authorize-security-group-ingress \\"
echo "      --group-id <sg-id> --protocol tcp --port 3000 --cidr <your-ip>/32"
echo ""
echo "  To update dashboards after code changes:"
echo "    sudo cp ~/AutoTraderBot/deploy/grafana/dashboards/*.json /etc/grafana/dashboards/"
echo "    sudo systemctl restart grafana-server"
echo ""
