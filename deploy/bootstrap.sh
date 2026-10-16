#!/usr/bin/env bash
# One-time VPS setup for the QuipQuill cloud (Ubuntu 24.04+). Run as root; safe to re-run.
#
#   scp deploy/bootstrap.sh root@HOST: && ssh root@HOST 'bash bootstrap.sh "<deploy public key>"'
#
# Installs Docker from Ubuntu's own packages, adds 2 GB swap (the 1 GB plan is tight),
# opens only 22/80/443, creates the `deploy` user that CI logs in as, and turns off
# SSH password login. Secrets (/opt/quipquill/.env, cloud.env) are written separately.
set -euo pipefail

DEPLOY_KEY="${1:?usage: bootstrap.sh \"<ssh public key for the deploy user>\"}"
APP_DIR=/opt/quipquill

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get upgrade -yq
apt-get install -yq docker.io docker-compose-v2 docker-buildx ufw unattended-upgrades

# Swap: Postgres + RabbitMQ + API + Caddy fit in 1 GB only with headroom on disk.
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
sysctl -q -w vm.swappiness=10
echo 'vm.swappiness=10' > /etc/sysctl.d/90-quipquill.conf

# Container logs are rotated, otherwise they slowly fill the 10 GB disk.
cat > /etc/docker/daemon.json <<'JSON'
{
  "log-driver": "json-file",
  "log-opts": { "max-size": "10m", "max-file": "3" }
}
JSON
systemctl enable docker
systemctl restart docker

# Firewall. Only Caddy publishes ports, so Docker bypassing ufw does not expose anything else.
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 443/udp
ufw --force enable

# The user CI deploys as. Membership in `docker` is root-equivalent: the key lives only in
# the GitHub secret DEPLOY_SSH_KEY.
id deploy >/dev/null 2>&1 || useradd --create-home --shell /bin/bash deploy
usermod -aG docker deploy
install -d -m 700 -o deploy -g deploy /home/deploy/.ssh
echo "$DEPLOY_KEY" > /home/deploy/.ssh/authorized_keys
chown deploy:deploy /home/deploy/.ssh/authorized_keys
chmod 600 /home/deploy/.ssh/authorized_keys
install -d -m 750 -o deploy -g deploy "$APP_DIR"

# SSH: keys only. cloud-init's drop-in turns passwords on, so ours must sort before it.
cat > /etc/ssh/sshd_config.d/01-quipquill.conf <<'CONF'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin prohibit-password
CONF
sshd -t
systemctl reload ssh

echo "bootstrap done: $(docker --version), $(docker compose version)"
