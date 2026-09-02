#!/usr/bin/env bash
# Push a flake dev shell's input closure to the S3 Nix binary cache — only
# the store paths cache.nixos.org does not already serve (the fenix Rust
# toolchains, the LLVM/MLIR symlink join, the shell itself), so the bucket
# stays small and a run with nothing new costs a few HEAD requests.
#
# Usage: scripts/ci/nix-cache-push.sh <devShell name>   (e.g. default)
# Needs NIX_S3_STORE and AWS_* from .github/actions/s3-cache, and the shell
# already materialized (`nix develop … --command true`) so nothing builds here.
#
# Before pushing, a throwaway path is round-tripped through the store and
# realised back through the daemon: that is the only way to prove the
# daemon-side substituter (root's credentials, nix.conf line) works, which
# the client-side `nix copy` cannot tell us.
set -euo pipefail

shell=${1:?usage: $0 <devShell name>}
: "${NIX_S3_STORE:?NIX_S3_STORE is not set — no S3 cache credentials}"

system=$(nix eval --impure --raw --expr builtins.currentSystem)

probe_dir=$(mktemp -d)
echo "reussir nix-cache probe ${system} ${GITHUB_RUN_ID:-local} ${RANDOM}${RANDOM}" \
  > "${probe_dir}/reussir-nix-cache-probe"
probe=$(nix-store --add "${probe_dir}/reussir-nix-cache-probe")
nix copy --to "${NIX_S3_STORE}" "${probe}"
nix store delete "${probe}"
if nix-store --realise "${probe}" > /dev/null; then
  echo "S3 substituter round-trip through the daemon: OK"
else
  echo "::warning::the Nix daemon could not substitute from ${NIX_S3_STORE%%\?*}; pushes will not be used"
fi

# mkShell's inputDerivation is a trivial derivation whose inputs are the
# shell's inputs; its closure is exactly what `nix develop` needs.
input=$(nix build --no-link --print-out-paths ".#devShells.${system}.${shell}.inputDerivation")
work=$(mktemp -d)
nix path-info -r "${input}" > "${work}/closure"

# Keep only paths whose narinfo cache.nixos.org does not have. A failed
# HEAD (network hiccup) counts as missing — an extra upload, never a gap.
# shellcheck disable=SC2016  # $0/$1 are the sh -c arguments, not expanded here
while read -r path; do
  hash=${path#/nix/store/}
  printf '%s %s\n' "${hash%%-*}" "${path}"
done < "${work}/closure" \
  | xargs -P 16 -n 2 sh -c \
      'curl -sfI -o /dev/null --retry 2 "https://cache.nixos.org/$0.narinfo" || echo "$1"' \
  | sort > "${work}/missing"

echo "closure: $(wc -l < "${work}/closure") paths, not on cache.nixos.org: $(wc -l < "${work}/missing")"
if [ -s "${work}/missing" ]; then
  # --no-recursive: the references of these paths that live upstream must
  # not be re-uploaded; a binary cache need not hold a path's full closure.
  xargs nix copy --no-recursive --to "${NIX_S3_STORE}" < "${work}/missing"
fi
