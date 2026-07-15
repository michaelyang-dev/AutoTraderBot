# Deployment & Disaster Recovery

> The old step-by-step EC2 setup guide and `deploy/aws-setup.sh` were **removed 2026-07-15**.
> They described the retired **April 2026, Alpaca-paper-only** system (PM2 apps `ml-signal-server`/
> `express-server`, LightGBM model uploads) and no longer matched the live IBKR system — a
> from-scratch rebuild off them would have produced a broken, paper-only box with no live trading.

## Where the system is documented

- **Live layout / services / crons / Telegram controls:** [`docs/LIVE_SYSTEM.md`](../docs/LIVE_SYSTEM.md) — source of truth.
- **PM2 services:** [`ecosystem.config.js`](../ecosystem.config.js) — `ibkr-engine`, `signal-server`, `trading-engine`, `edgar-monitor`.

## Why there is no "rebuild from git"

Much of the live system is **not in git and can't be**: ~17 GB of data (WRDS, enhanced_data, NAV
history, journals), all secrets (IBKR, Telegram, Alpaca, FMP, Massive, Ortex — in `.env`), the IB
Gateway docker container + its 2FA login, and the system crontab. The repo holds the *code* only, so
the code alone cannot restore a working box. **Disaster recovery is a machine image, not a script.**

## Disaster recovery — restore the AWS Backup snapshot

DR is an **AWS Backup** plan (created 2026-07-15): weekly (Sat 00:30 ET), 35-day retention, vault
**Default**, protecting instance **`i-092b64baffc90b6d0`** (`54.158.238.15`). Each recovery point is a
full-instance image.

To recover after a lost / corrupted instance:
1. **AWS Backup → Backup vaults → Default** → newest recovery point for the instance → **Restore** →
   launch a new EC2 instance from it (same type/AZ, keep the key pair).
2. Re-associate the Elastic IP / update whatever points at `54.158.238.15` to the new instance.
3. **IB Gateway** needs its **2FA re-approved** on first connect (IBKR mobile / IB Key). Do **not**
   log into IBKR web / Client Portal — it kills the Gateway session.
4. `pm2 list` should show `ibkr-engine` / `signal-server` / `trading-engine` online (PM2 resurrects
   from its saved dump); `crontab -l` restores with the image.
5. Sanity-check: signals on `:5001`, `/pnl` + `/status` in Telegram, and that
   `ml_service/data/ibkr_rebal_state.json` (rebalance counter/date) is intact.

## Routine updates (existing box)

```bash
cd ~/AutoTraderBot
git pull origin main
npm install --production                                        # only if package.json changed
ml_service/venv/bin/pip install -r ml_service/requirements.txt  # only if requirements changed
pm2 restart <service> && pm2 save                               # restart only what changed
```

> Avoid `pm2 restart ibkr-engine` during market hours without cause — on reconnect it cancels pending orders.
