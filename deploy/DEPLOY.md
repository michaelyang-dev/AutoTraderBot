# AutoTrader Bot — AWS EC2 Deployment Guide

## 1. Launch an EC2 Instance

### AMI
- **Ubuntu Server 22.04 LTS** (HVM, SSD Volume Type)
- AMI ID: `ami-0f5ee92e2d63afc18` (us-east-1) — or search "Ubuntu 22.04" in the AMI catalog for your region

### Instance Type
- **t2.micro** (free tier eligible) — 1 vCPU, 1 GB RAM
- This is sufficient for the bot. The ML model is lightweight (LightGBM) and the Express server uses minimal memory.
- If you run into memory issues with the ML model loading, upgrade to **t2.small** (2 GB RAM).

### Storage
- **20 GB** gp3 SSD (default 8 GB is too small for Node modules + Python venv + trade journal DB)

### Key Pair
- Create or select an existing key pair (`.pem` file)
- Save it securely — you'll need it to SSH in

### Security Group
Create a security group with these inbound rules:

| Type       | Protocol | Port  | Source        | Purpose                    |
|------------|----------|-------|---------------|----------------------------|
| SSH        | TCP      | 22    | Your IP       | SSH access                 |
| Custom TCP | TCP      | 3001  | Your IP       | Express API + dashboard    |
| Custom TCP | TCP      | 5001  | 127.0.0.1/32  | ML server (localhost only) |

> **Security note:** Restrict port 3001 to **your IP only** (`x.x.x.x/32`), not `0.0.0.0/0`. The bot has no authentication — anyone with access can see your portfolio and trading state. If you need remote access from changing IPs, add a simple auth layer or use SSH tunneling instead (see below).

### Network
- Use default VPC
- Auto-assign public IP: **Enable**

---

## 2. SSH Into the Instance

```bash
# Set correct permissions on your key file
chmod 400 your-key.pem

# SSH in (replace <EC2-IP> with your instance's public IP)
ssh -i your-key.pem ubuntu@<EC2-IP>
```

---

## 3. Run the Setup Script

Upload and run the setup script:

```bash
# Option A: If the repo is already pushed to GitHub with the deploy/ folder
git clone https://github.com/michaelyang-dev/AutoTraderBot.git
cd AutoTraderBot
chmod +x deploy/aws-setup.sh
./deploy/aws-setup.sh
```

```bash
# Option B: Upload the script directly from your local machine
# (run this from your local terminal, not on EC2)
scp -i your-key.pem deploy/aws-setup.sh ubuntu@<EC2-IP>:~/aws-setup.sh

# Then on EC2:
chmod +x ~/aws-setup.sh
~/aws-setup.sh
```

The script takes about 3-5 minutes and installs everything automatically.

---

## 4. Configure API Keys

```bash
cd ~/AutoTraderBot
nano .env
```

Replace the placeholders with your real keys:
```
ALPACA_API_KEY=PK...your real key...
ALPACA_SECRET_KEY=...your real secret...
FMP_API_KEY=...your FMP key (optional)...
```

Save and exit (`Ctrl+X`, then `Y`, then `Enter`).

---

## 5. Upload the ML Model

The trained LightGBM model (`model.lgb`) is not stored in git. Upload it from your local machine:

```bash
# Run this from your LOCAL machine (not EC2)
scp -i your-key.pem ml_service/data/model.lgb ubuntu@<EC2-IP>:~/AutoTraderBot/ml_service/data/model.lgb
```

> The bot will still run without the model — it falls back to the consensus engine (5-strategy voting). ML-driven trades require the model file.

---

## 6. Start the Bot

```bash
cd ~/AutoTraderBot
pm2 start ecosystem.config.js
pm2 save    # saves the process list so PM2 restarts them on reboot
```

Check that both services are running:
```bash
pm2 status
```

Expected output:
```
┌─────────────────┬────┬──────┬───────┐
│ Name            │ id │ mode │ status│
├─────────────────┼────┼──────┼───────┤
│ ml-signal-server│ 0  │ fork │ online│
│ express-server  │ 1  │ fork │ online│
└─────────────────┴────┴──────┴───────┘
```

---

## 7. Verify Everything Works

```bash
# Check Express server health
curl http://localhost:3001/api/health

# Check trading engine state
curl http://localhost:3001/api/trading-state | python3 -m json.tool | head -20

# Check ML signal server
curl http://localhost:5001/health

# View live logs
pm2 logs

# View just the trading engine logs
pm2 logs express-server --lines 50
```

From your local machine (if port 3001 is open in the security group):
```bash
curl http://<EC2-IP>:3001/api/health
curl http://<EC2-IP>:3001/api/trading-state
```

---

## 8. Useful PM2 Commands

```bash
pm2 status                     # process list
pm2 logs                       # tail all logs
pm2 logs express-server        # tail Express logs only
pm2 logs ml-signal-server      # tail ML server logs only
pm2 monit                      # live CPU/memory monitor
pm2 restart all                # restart both services
pm2 restart express-server     # restart just Express
pm2 stop all                   # stop everything
pm2 delete all                 # remove from PM2 (need to pm2 start again)
```

---

## 9. SSH Tunnel (Secure Alternative to Opening Port 3001)

Instead of opening port 3001 to the internet, you can use an SSH tunnel:

```bash
# Run on your LOCAL machine — forwards localhost:3001 to EC2's port 3001
ssh -i your-key.pem -L 3001:localhost:3001 ubuntu@<EC2-IP>
```

Then access the dashboard at `http://localhost:3001` in your browser. No security group changes needed — only port 22 (SSH) is exposed.

---

## 10. Updating the Bot

```bash
cd ~/AutoTraderBot
pm2 stop all
git pull origin main
npm install --production
source ml_service/venv/bin/activate && pip install -r ml_service/requirements.txt && deactivate
pm2 start ecosystem.config.js
pm2 save
```

---

## Troubleshooting

### Bot won't start
```bash
# Check logs for errors
pm2 logs express-server --lines 100

# Common issue: missing .env keys
cat .env   # make sure keys are filled in, not placeholders
```

### ML server won't start
```bash
pm2 logs ml-signal-server --lines 100

# Common issue: missing model file
ls -la ml_service/data/model.lgb

# Common issue: Python package missing
source ml_service/venv/bin/activate
python3 -c "import lightgbm; print('OK')"
deactivate
```

### Out of memory (t2.micro)
```bash
# Check memory usage
free -m

# Add swap space (1 GB)
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Port 3001 not accessible
- Check security group allows inbound TCP 3001 from your IP
- Check the Express server is actually running: `pm2 status`
- Check it's listening: `ss -tlnp | grep 3001`
