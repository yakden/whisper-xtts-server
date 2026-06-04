# GPU on the Proxmox host + LXC passthrough (NVIDIA Tesla T4)

This deploys the driver on the Proxmox **host** and shares the GPU into an
**LXC container** (lighter than a passthrough VM; no dedicated RAM/disk for a guest OS).

## 1. Host: install the NVIDIA driver

```bash
# Disable the open-source nouveau driver
echo -e "blacklist nouveau\noptions nouveau modeset=0" >/etc/modprobe.d/blacklist-nouveau.conf
update-initramfs -u

# Install build prerequisites + headers for the running kernel
apt-get update
apt-get install -y pve-headers-$(uname -r) build-essential

# Install the driver (use NVIDIA's .run installer or the CUDA repo).
# Record the EXACT version — the container must match it.
#   bash NVIDIA-Linux-x86_64-<version>.run --no-questions --ui=none
modprobe nvidia nvidia_uvm
nvidia-smi          # should list the Tesla T4

# Create device nodes at boot and keep the GPU initialized
nvidia-persistenced || true
```

If `nvidia-smi` only works after a reboot (DKMS/initramfs), schedule it — the
host also runs production VMs.

## 2. Host: pass the GPU into the LXC

Find the device major/minor numbers:

```bash
ls -l /dev/nvidia*        # note the major numbers (usually 195 for nvidia*, 511/510 for uvm)
```

Append `deploy/proxmox/lxc-gpu.conf.snippet` to `/etc/pve/lxc/<CTID>.conf`
(adjust minor numbers if they differ), then restart the container:

```bash
pct stop <CTID> && pct start <CTID>
```

## 3. Container: install matching userspace driver

Inside the container install the **same driver version** with `--no-kernel-module`:

```bash
bash NVIDIA-Linux-x86_64-<version>.run --no-kernel-module --no-questions --ui=none
nvidia-smi            # should now list the Tesla T4 from inside the container
```

## 4. Deploy the app

Native (recommended on Proxmox LXC):

```bash
git clone https://github.com/yakden/whisper-xtts-server /opt/src
cd /opt/src && cp .env.example .env
bash scripts/download-models.sh
bash scripts/install-native.sh
```

Or Docker (requires nvidia-container-toolkit + LXC nesting): `docker compose up -d --build`.
