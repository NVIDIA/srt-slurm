#!/usr/bin/env bash
# Refuse to publish a release whose binary set is incomplete.
#
#   verify_release_assets.sh <dist-dir> <tool>...
#
# For every tool that has any asset in <dist-dir>, require one binary and its
# .sha256 for each of x86_64 and aarch64, and verify the checksums. A tool with
# no assets at all is reported and allowed: the first release has nothing to
# carry forward, and a binary added later is absent from older releases.
#
# Exit 1 on any partial set: a .sha256 without its binary, a binary without its
# .sha256, a missing architecture, or a checksum mismatch. v2.7.0 shipped the
# x86_64 tachometer-scraper .sha256 without the binary because a reset
# connection during the carry-forward download was only warned about, and
# every later release copied that gap forward until someone noticed.
set -euo pipefail

usage="usage: verify_release_assets.sh <dist-dir> <tool>..."
dist="${1:?$usage}"
shift
if [ $# -eq 0 ]; then
  echo "$usage" >&2
  exit 2
fi
if [ ! -d "$dist" ]; then
  echo "::error::$dist is not a directory"
  exit 1
fi

status=0
for tool in "$@"; do
  shopt -s nullglob
  present=("$dist/$tool"-*)
  shopt -u nullglob
  if [ ${#present[@]} -eq 0 ]; then
    echo "::warning::No $tool assets in $dist; this release omits $tool"
    continue
  fi
  for arch in x86_64 aarch64; do
    shopt -s nullglob
    sums=("$dist/$tool-$arch-"*.sha256)
    binaries=()
    for f in "$dist/$tool-$arch-"*; do
      case "$f" in *.sha256) ;; *) binaries+=("$f") ;; esac
    done
    shopt -u nullglob
    if [ ${#sums[@]} -ne 1 ] || [ ${#binaries[@]} -ne 1 ]; then
      echo "::error::$tool $arch: expected one binary and one .sha256 in $dist," \
        "found ${#binaries[@]} binary and ${#sums[@]} .sha256"
      status=1
      continue
    fi
    sum_file=$(basename "${sums[0]}")
    binary=$(basename "${binaries[0]}")
    if [ "$binary.sha256" != "$sum_file" ]; then
      echo "::error::$tool $arch: $sum_file does not describe $binary"
      status=1
      continue
    fi
    if ! (cd "$dist" && sha256sum --check --strict --status "$sum_file"); then
      echo "::error::$tool $arch: checksum mismatch for $binary"
      status=1
      continue
    fi
    echo "$tool $arch: $binary ok"
  done
done
exit $status
