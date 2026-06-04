#!/usr/bin/env bash
# Helper to create the LXC container on the Proxmox host. Run on the HOST as root.
# Adjust CTID / storage / template to your environment before running.
set -euo pipefail

CTID="${CTID:-210}"
HOSTNAME="${HOSTNAME:-voice-ai}"
STORAGE="${STORAGE:-local-zfs}"
TEMPLATE="${TEMPLATE:-local:vztmpl/debian-12-standard_12.7-1_amd64.tar.zst}"
DISK_GB="${DISK_GB:-40}"
MEMORY_MB="${MEMORY_MB:-3072}"
CORES="${CORES:-2}"
BRIDGE="${BRIDGE:-vmbr0}"

echo ">> Creating LXC $CTID ($HOSTNAME)"
pct create "$CTID" "$TEMPLATE" \
    --hostname "$HOSTNAME" \
    --cores "$CORES" \
    --memory "$MEMORY_MB" \
    --rootfs "$STORAGE:$DISK_GB" \
    --net0 "name=eth0,bridge=$BRIDGE,ip=dhcp" \
    --features nesting=1 \
    --unprivileged 1 \
    --onboot 1

CONF="/etc/pve/lxc/$CTID.conf"
echo ">> Appending GPU passthrough config to $CONF"
cat "$(dirname "${BASH_SOURCE[0]}")/lxc-gpu.conf.snippet" >>"$CONF"

echo ">> Start with: pct start $CTID"
echo ">> Then follow nvidia-host-setup.md step 3 inside the container."
