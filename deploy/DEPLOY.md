# Deploying PHVE behind Cloudflare Tunnel

This stack runs the Streamlit demonstrator (`simulations/app_demo.py`)
behind a Cloudflare Tunnel, so the origin host needs **no inbound port**
opened. SSL termination is handled by Cloudflare's edge.

```
 user --(HTTPS)--> Cloudflare edge --(tunnel, outbound)--> cloudflared --(HTTP)--> streamlit:8501
```

## Prerequisites

- Docker Engine + Docker Compose v2 on the host.
- A Cloudflare account that owns the zone (e.g.\ `names.legal`).

## One-time Cloudflare setup

1. Open <https://one.dash.cloudflare.com> > **Zero Trust** >
   **Networks** > **Tunnels** > **Create a tunnel**.
2. Connector type: **Cloudflared**. Tunnel name: `phve`.
3. On the **Install and run a connector** screen, copy the long
   `--token` value (a base64 blob starting with `ey...`). Save it: this
   is `CF_TUNNEL_TOKEN`.
4. Click **Next** and add a **Public hostname**:
   - Subdomain: `phve`
   - Domain: `names.legal`
   - Type: `HTTP`
   - URL: `streamlit:8501`
5. Save.

The tunnel will appear as **Inactive** until the connector starts on
the host.

## Deploy on the host

```bash
git clone https://github.com/Altius-Academy-SNC/PHVE.git /opt/phve
cd /opt/phve/deploy
cp .env.example .env
# edit .env, paste the token after CF_TUNNEL_TOKEN=
docker compose --env-file .env up -d --build
docker compose ps
```

Within ~30 seconds, the tunnel turns **Healthy** in the Cloudflare
dashboard and `https://phve.names.legal` resolves the Streamlit app.

## Update

```bash
cd /opt/phve
git pull
cd deploy
docker compose --env-file .env up -d --build
```

## Logs / debug

```bash
docker compose logs -f streamlit       # the app
docker compose logs -f cloudflared     # the tunnel connector
```

## Tear down

```bash
docker compose --env-file .env down
```

The named volume `phve_nilearn_cache` (MNI152 templates, ~50 MB) is
preserved across restarts. Add `-v` to `down` to drop it.
