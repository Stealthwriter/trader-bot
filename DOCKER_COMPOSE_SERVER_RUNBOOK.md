# Docker Compose Server Runbook (trader.aivntg.com)

This runbook is the end-to-end checklist for deploying this repo on a VPS with:

- `mt5linux`
- `chart-server`
- `bot`
- `caddy` (HTTPS reverse proxy)

It assumes your public domain is `trader.aivntg.com`.

## 1) What is running

Current `docker-compose.yml` services:

- `mt5linux`: MT5 runtime + noVNC UI
- `chart-server`: FastAPI app (`chart_server.py`) on internal port `8000`
- `bot`: trading bot (`bot.py`)
- `caddy`: public web entrypoint on ports `80/443`, reverse-proxying to `chart-server:8000`

Current Caddy site:

```caddyfile
trader.aivntg.com {
    encode zstd gzip
    reverse_proxy chart-server:8000
}
```

## 2) Required inbound ports (server + cloud firewall)

Open these ports on both your cloud firewall and OS firewall:

- `22/tcp`: SSH
- `80/tcp`: required for HTTP and ACME validation
- `443/tcp`: HTTPS
- `443/udp`: optional but recommended for HTTP/3
- `6081/tcp`: MT5 noVNC UI (restrict to your own IP if possible)
- `18812/tcp`: mt5linux RPC (only needed if remote clients must access MT5 directly)

Security recommendations:

- Restrict `6081` to your own public IP (`x.x.x.x/32`).
- Restrict `18812` to trusted IPs, or close it if not needed externally.

Example `ufw` rules:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 443/udp
sudo ufw allow from <YOUR_PUBLIC_IP>/32 to any port 6081 proto tcp
# Optional: only if an external bot/client needs direct MT5 RPC
sudo ufw allow from <TRUSTED_BOT_IP>/32 to any port 18812 proto tcp
sudo ufw status
```

## 3) Cloudflare setup (mandatory for this domain)

In Cloudflare for `aivntg.com`:

1. DNS:
   - Create `A` record:
     - Name: `trader`
     - Value: `<YOUR_SERVER_PUBLIC_IP>`
     - Proxy status: Proxied (orange cloud)

2. SSL/TLS -> Overview:
   - Set encryption mode to **Full (strict)**.
   - Do **not** use `Flexible` (can cause redirect loops).

3. Redirect rules / Page rules:
   - Remove any self-redirect that rewrites `https://trader.aivntg.com` to itself.
   - If you use "Always Use HTTPS", that is fine with `Full (strict)`.

## 4) Install Docker + Compose plugin (Ubuntu)

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

## 5) Pull project on server

First time:

```bash
cd ~
git clone <YOUR_GITHUB_REPO_URL> trader-bot
cd trader-bot
```

Existing deployment update:

```bash
cd /path/to/trader-bot
git fetch origin
git checkout main
git pull --ff-only origin main
```

## 6) Start / update services

```bash
cd /path/to/trader-bot
docker compose up -d --build
docker compose ps
```

Note: you usually do **not** need `docker compose down` before update.

## 7) First-time MT5 login (important)

1. Open `http://<SERVER_PUBLIC_IP>:6081`
2. Enter VNC password (from `docker-compose.yml`)
3. Complete MT5 broker login in UI and save credentials

Until this is done, MT5 initialize can fail/retry in `bot` and `chart-server`.

## 8) Verify deployment

Public endpoint:

- `https://trader.aivntg.com/`
- `https://trader.aivntg.com/api/candles?limit=100`

Service status:

```bash
docker compose ps
```

## 9) Logs for each service

```bash
docker compose logs -f mt5linux
docker compose logs -f chart-server
docker compose logs -f caddy
docker compose logs -f bot
```

Useful variants:

```bash
docker compose logs --tail=100 chart-server
docker compose logs --since=10m caddy
docker compose logs -f
```

## 10) Caddy config reload (when only Caddyfile changes)

```bash
docker compose exec -w /etc/caddy caddy caddy reload
```

Or recreate just Caddy:

```bash
docker compose up -d caddy
```

## 11) Troubleshooting

### A) Browser says ERR_TOO_MANY_REDIRECTS

Most common cause is Cloudflare SSL mode/rules mismatch.

Check:

- Cloudflare SSL mode is `Full (strict)` (not `Flexible`)
- No redirect rule loops `https://trader.aivntg.com` back to itself
- Clear browser cache/cookies for the domain after fixing rules

Quick header check:

```bash
curl -I https://trader.aivntg.com
```

### B) Caddy does not serve HTTPS

Check:

- `trader.aivntg.com` DNS points to your server
- Ports `80` and `443` are reachable externally
- Caddy logs:

```bash
docker compose logs -f caddy
```

### C) chart-server or bot keeps retrying MT5 init

Check:

- MT5 UI login completed at least once on port `6081`
- `mt5linux` container healthy:

```bash
docker compose logs -f mt5linux
docker compose logs -f chart-server
docker compose logs -f bot
```

## 12) Optional hardening checklist

- Change `VNC_PASSWORD` in `docker-compose.yml` to a strong value.
- Restrict `6081` and `18812` by source IP.
- Keep system packages and Docker engine updated.
