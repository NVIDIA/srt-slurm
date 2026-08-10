#!/usr/bin/env bash
# Install the NVIDIA Nsight Systems CLI in the ephemeral worker container.

set -euo pipefail

readonly nvidia_devtools_keyring="/usr/share/keyrings/nvidia-devtools-keyring.gpg"

export DEBIAN_FRONTEND=noninteractive

. /etc/os-release
if [[ "${ID}" != "ubuntu" ]]; then
  echo "ERROR: Nsight Systems setup expects Ubuntu, found ${ID}." >&2
  exit 1
fi

ubuntu_version="${VERSION_ID//./}"
architecture="$(dpkg --print-architecture)"

case "${architecture}" in
  amd64)
    nsys_target="target-linux-x64"
    ;;
  arm64)
    nsys_target="target-linux-sbsa-armv8"
    ;;
  *)
    echo "ERROR: Unsupported architecture for Nsight Systems: ${architecture}." >&2
    exit 1
    ;;
esac

apt-get update -o Acquire::Retries=3
apt-get install -y --no-install-recommends ca-certificates gnupg wget
wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
  | gpg --dearmor --yes -o "${nvidia_devtools_keyring}"
printf 'deb [signed-by=%s] https://developer.download.nvidia.com/devtools/repos/ubuntu%s/%s/ /\n' \
  "${nvidia_devtools_keyring}" "${ubuntu_version}" "${architecture}" \
  > /etc/apt/sources.list.d/nvidia-devtools.list
apt-get update -o Acquire::Retries=3
apt-get install -y --no-install-recommends nsight-systems-cli

# Always select the newest binary installed by the repository's rolling
# nsight-systems-cli package, even if the image registered an older version.
nsys_bin="$({
  find /opt/nvidia/nsight-systems-cli -type f \
    -path "*/${nsys_target}/nsys" -print 2>/dev/null || true
} | sort -V | tail -n 1)"

if [[ -z "${nsys_bin}" ]]; then
  echo "ERROR: Nsight Systems installed, but no ${nsys_target} executable was found." >&2
  exit 1
fi

ln -sfn "${nsys_bin}" /usr/local/bin/nsys
hash -r

echo "Using $(command -v nsys)"
nsys --version
