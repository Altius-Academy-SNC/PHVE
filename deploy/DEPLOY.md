# Deploying PHVE on a host with shared nginx + certbot

This stack assumes the host already runs:

- Nginx as a reverse proxy on the host (not in a container), aggregating
  virtual hosts in `/etc/nginx/sites-available/default`.
- Certbot with Let's Encrypt for TLS certificates on every served domain.
- Docker Engine for application containers, each bound to a private port
  on `127.0.0.1`.

The Streamlit demonstrator is added the same way as any other app on
the host:

```
 user --(HTTPS)--> nginx (host) --(HTTP)--> streamlit (docker, 127.0.0.1:8501)
```

## 1. Clone the repo on the server

```bash
cd /opt
git clone https://github.com/Altius-Academy-SNC/PHVE.git phve
```

## 2. Build and start the Streamlit container

```bash
cd /opt/phve
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml ps
```

This binds Streamlit to `127.0.0.1:8501` only; the container is *not*
reachable from the public internet.

## 3. Add the nginx server block

Append the contents of `deploy/nginx-phve.conf` (a single port-80 server
block that proxies to `127.0.0.1:8501`) to
`/etc/nginx/sites-available/default`, then reload:

```bash
cat /opt/phve/deploy/nginx-phve.conf >> /etc/nginx/sites-available/default
nginx -t && systemctl reload nginx
```

## 4. Issue the TLS certificate

```bash
certbot --nginx -d phve.names.legal
```

Certbot performs the HTTP-01 challenge through the freshly added port-80
block, then injects `listen 443 ssl` + `ssl_certificate` paths into the
config and reloads nginx. Choose `2: Redirect` when prompted to
auto-redirect HTTP -> HTTPS.

## 5. Sanity checks

```bash
curl -I https://phve.names.legal
docker compose -f deploy/docker-compose.yml logs -f streamlit
```

## Update later

```bash
cd /opt/phve
git pull
docker compose -f deploy/docker-compose.yml up -d --build
```

No nginx changes are needed for code updates.

## Notes

- Streamlit live updates use WebSockets; the `Upgrade`/`Connection` and
  `proxy_read_timeout 86400` directives in `nginx-phve.conf` are
  required.
- The named volume `phve_nilearn_cache` (the MNI152 templates, ~50 MB)
  persists across restarts so the brain mask is not redownloaded.
- If `phve.names.legal` is proxied through Cloudflare (orange cloud),
  HTTP-01 still works because Cloudflare passes
  `/.well-known/acme-challenge/` through. If issuance fails, set the
  record to **DNS only** (gray cloud) for the duration of `certbot
  --nginx ...`, then flip back to **Proxied**.
