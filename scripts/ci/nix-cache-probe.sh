#!/usr/bin/env bash
# Prove the daemon-side S3 substituter works (root's credentials, the
# nix.conf line the installer wrote) before anything depends on it: add a
# throwaway path, copy it to the store, delete it locally, and realise it
# back — the daemon must fetch it from S3. If any leg fails or hangs, the
# substituter is removed from the daemon's configuration and the daemon
# restarted, so a broken cache can cost this probe's timeout and nothing
# else; NIX_S3_STORE is emptied so the push step is skipped as well.
#
# Needs NIX_S3_STORE and AWS_* from .github/actions/s3-cache. Runs from the
# repository checkout, right after the Nix installer step.
set -uo pipefail

: "${NIX_S3_STORE:?NIX_S3_STORE is not set — no S3 cache credentials}"
timeout_secs=${NIX_CACHE_PROBE_TIMEOUT:-120}

# macOS runners have no coreutils `timeout`, and pulling one through nix
# would go through the very substituter under test.
with_timeout() {
  local secs=$1 pid watchdog rc; shift
  "$@" & pid=$!
  ( sleep "$secs"; kill "$pid" 2>/dev/null ) & watchdog=$!
  wait "$pid"; rc=$?
  kill "$watchdog" 2>/dev/null; wait "$watchdog" 2>/dev/null
  return "$rc"
}

disable_substituter() {
  echo "::warning::$1 — removing the S3 substituter from the Nix daemon configuration"
  for conf in /etc/nix/nix.conf /etc/nix/nix.custom.conf; do
    [ -e "$conf" ] && sudo sed -i.bak '/^extra-substituters = s3:/d' "$conf"
  done
  if [ "$(uname)" = Darwin ]; then
    sudo launchctl kickstart -k system/systems.determinate.nix-daemon 2>/dev/null \
      || sudo launchctl kickstart -k system/org.nixos.nix-daemon
  else
    sudo systemctl restart nix-daemon.service
  fi
  echo "NIX_S3_STORE=" >> "${GITHUB_ENV:-/dev/null}"
  exit 0
}

probe_dir=$(mktemp -d)
echo "reussir nix-cache probe ${GITHUB_RUN_ID:-local} ${RANDOM}${RANDOM}" \
  > "${probe_dir}/reussir-nix-cache-probe"
probe=$(nix-store --add "${probe_dir}/reussir-nix-cache-probe")

with_timeout "$timeout_secs" nix copy --to "$NIX_S3_STORE" "$probe" \
  || disable_substituter "nix copy to the S3 store failed or exceeded ${timeout_secs}s"
nix store delete "$probe"
with_timeout "$timeout_secs" nix-store --realise "$probe" > /dev/null \
  || disable_substituter "the daemon could not substitute from the S3 store within ${timeout_secs}s"

echo "S3 substituter round-trip through the daemon: OK"
