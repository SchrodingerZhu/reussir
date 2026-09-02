#!/usr/bin/env bash
# Push a flake dev shell's input closure to the S3 Nix binary cache.
#
# Usage: scripts/ci/nix-cache-push.sh <devShell name>   (e.g. default)
# Needs NIX_S3_STORE and AWS_* from .github/actions/s3-cache, and the shell
# already materialized (`nix develop … --command true`) so nothing builds here.
#
# The whole closure goes up, cache.nixos.org paths included: a binary cache
# refuses a path whose references it does not hold ("cannot add … because
# the reference … is not valid"), so the store cannot be limited to the
# paths upstream lacks. Only the first push per toolchain pays for it —
# `nix copy` skips what the store already has — and the substituter's
# priority (below cache.nixos.org) keeps upstream paths coming from the
# CDN; this store effectively serves just the fenix Rust toolchains, the
# LLVM/MLIR join, and the shells themselves.
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
echo "closure: $(nix path-info -r "${input}" | wc -l) paths"
nix copy --to "${NIX_S3_STORE}" "${input}"
