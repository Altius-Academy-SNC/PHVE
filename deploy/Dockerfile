# Streamlit demonstrator for PHVE.
#
# Build context is the repository root. Run:
#   docker compose -f deploy/docker-compose.yml up -d --build
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NILEARN_DATA=/data/nilearn_data

# System libraries needed by scipy / scikit-image / nilearn / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libxext6 \
        libsm6 \
        libxrender1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY simulations/ /app/simulations/

# Pre-create the nilearn cache dir so the first request doesn't race on it.
RUN mkdir -p /data/nilearn_data

# Build the brain surface once, here, instead of on the first request.
# Measured in this container, the first render cost 3.58 s, of which 2.20 s
# was importing nilearn (1.57 s) and scipy.ndimage (0.63 s) -- both needed
# only to turn the MNI152 template into a triangulated surface that never
# changes. Shipping the result (0.81 MiB) keeps those imports off the
# runtime path of the two pages the demonstrator serves.
RUN python /app/simulations/prewarm.py

WORKDIR /app/simulations

EXPOSE 8501

# --server.address 0.0.0.0 so the container is reachable on its docker network.
# --server.headless true disables the auto-open browser nag.
# --browser.gatherUsageStats false avoids phoning home.
# --server.enableXsrfProtection true is the default; leave it on.
CMD ["streamlit", "run", "app_demo.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
