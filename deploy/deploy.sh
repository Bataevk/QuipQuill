#!/usr/bin/env bash
# Roll the stack to a cloud image: ./deploy.sh ghcr.io/bataevk/quipquill-cloud:<sha>
# Without an argument it re-applies the image already pinned in .env. Rollback = an older tag.
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -ge 1 ]]; then
  if grep -q '^CLOUD_IMAGE=' .env; then
    sed -i "s|^CLOUD_IMAGE=.*|CLOUD_IMAGE=$1|" .env
  else
    echo "CLOUD_IMAGE=$1" >> .env
  fi
fi

docker compose pull --quiet
docker compose up -d --wait --remove-orphans
# Old releases are dropped (the disk is 10 GB); a rollback pulls its tag from GHCR again.
docker image prune -af >/dev/null
docker compose ps --format 'table {{.Service}}\t{{.Image}}\t{{.Status}}'
