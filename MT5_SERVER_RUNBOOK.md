# MT5 Server Runbook (Google Cloud VPS)

Use this file every time you create a new VPS or configure a new MT5 account.

## 0) What this setup does

- Runs MT5 inside Docker on Linux (Wine + noVNC).
- Exposes MT5 UI in browser on port `6081`.
- Exposes `mt5linux` RPC server on port `18812` for your Python bot.

## 1) Open ports in Google Cloud

Open these firewall rules in GCP:

- `6081/tcp`: noVNC browser access (for MT5 UI/login).
- `18812/tcp`: mt5linux API (only if bot is outside this VPS).

Security recommendations:

- Allow `6081` only from your own public IP (`x.x.x.x/32`).
- Keep `18812` private if bot runs on same VPS.
- If opening `18812`, allow only trusted bot server IP(s).

If Ubuntu firewall (`ufw`) is enabled:

```bash
sudo ufw allow 6081/tcp
sudo ufw allow 18812/tcp
sudo ufw status
```

## 2) Install Docker on VPS (Ubuntu)

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

## 3) Create and start MT5 container

```bash
mkdir -p ~/mt5linux
cd ~/mt5linux
```

Create `docker-compose.yml`:

```yaml
services:
  mt5linux:
    image: lprett/mt5linux:mt5-installed
    container_name: mt5linux
    environment:
      MT5_HOST: 0.0.0.0
      VNC_PASSWORD: CHANGE_ME_STRONG_PASSWORD
    ports:
      - "6081:6081"
      - "127.0.0.1:18812:18812"
    restart: unless-stopped
```

Start server:

```bash
docker compose up -d
docker ps --filter name=mt5linux
docker logs --tail 100 mt5linux
```

## 4) One-time MT5 UI login (important)

Open in browser:

`http://<VPS_PUBLIC_IP>:6081`

Then:

- Enter VNC password.
- Wait for MT5 window.
- Login to your broker account in MT5.
- Save credentials if you want auto reconnect.

Note: In many cases, `mt5.initialize()` returns `False` until this first UI/login step is completed at least once.

## 5) Python client setup (venv) and mt5linux install

Do not install in system Python directly (PEP 668 "externally managed environment").

```bash
cd ~
mkdir -p bot-client
cd bot-client
sudo apt install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mt5linux
```

## 6) Connection test

```bash
cd ~/bot-client
source .venv/bin/activate
python - <<'PY'
from mt5linux import MetaTrader5

mt5 = MetaTrader5(host="127.0.0.1", port=18812)
ok = mt5.initialize(timeout=120000)
print("initialize:", ok)
print("last_error:", mt5.last_error())
print("terminal_info:", mt5.terminal_info())
mt5.shutdown()
PY
```

Expected: `initialize: True` and `last_error: (1, 'Success')`.

## 7) How to call from your bot code

```python
from mt5linux import MetaTrader5

mt5 = MetaTrader5(host="127.0.0.1", port=18812)

if not mt5.initialize(timeout=120000):
    raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

info = mt5.terminal_info()
print(info)

mt5.shutdown()
```

If bot is on another machine, use:

- `host="<VPS_PUBLIC_IP>"`, `port=18812`
- Open firewall for `18812` only to bot IP(s)

## 8) Daily operations

Start:

```bash
cd ~/mt5linux
docker compose up -d
```

Stop:

```bash
cd ~/mt5linux
docker compose down
```

Logs:

```bash
docker logs -f mt5linux
```

## 9) Quick troubleshooting

- `ImportError ... from mt5linux (unknown location)`:
  - You are likely in a folder named `mt5linux`; run from another folder (example `~/bot-client`).
- `error: externally-managed-environment`:
  - Use virtualenv (`python3 -m venv .venv`), then install with that venv's `pip`.
- `initialize: False`:
  - Open noVNC (`6081`) and complete MT5 first-run/login.
  - Check `mt5.last_error()` and `docker logs mt5linux`.

## 10) Run full project stack (MT5 + API + bot)

This repository now includes a root `docker-compose.yml` that runs:

- `mt5linux` (`lprett/mt5linux:mt5-installed`)
- `api` (`chart_server.py` via `uvicorn`)
- `bot` (`bot.py`)

Notes:

- `bot_config.json` is set to `mt5.host = "mt5linux"` for internal Compose networking.
- Port `18812` is internal-only (not published to host).
- noVNC is exposed on `6081`.

Start:

```bash
cd /path/to/trader-bot
docker compose up --build -d
docker compose ps
```

First-time login (required):

1. Open `http://localhost:6081`.
2. Enter password: `password`.
3. Complete MT5 broker login in the UI and save credentials.

Validate services:

```bash
# API logs
docker compose logs -f api

# Bot logs
docker compose logs -f bot
```

Then open:

- `http://localhost:8000/`
- `http://localhost:8000/api/candles?limit=100`

Stop:

```bash
docker compose down
```
