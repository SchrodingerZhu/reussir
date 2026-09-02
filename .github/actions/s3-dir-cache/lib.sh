# Shared by the restore and save actions in this directory. Sourced, not run.

S3_DIR_CACHE_REMOTE="s3:sccache/dir-cache"

# rclone talks to the bucket through an environment-defined remote so no
# config file with credentials ever touches the disk.
export RCLONE_CONFIG_S3_TYPE=s3
export RCLONE_CONFIG_S3_PROVIDER=Other
export RCLONE_CONFIG_S3_ENDPOINT=https://usc1.contabostorage.com
export RCLONE_CONFIG_S3_REGION=default
export RCLONE_CONFIG_S3_FORCE_PATH_STYLE=true
export RCLONE_CONFIG_S3_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
export RCLONE_CONFIG_S3_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"

s3_dir_cache_available() {
  if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
    echo "no S3 credentials in the environment; dir-cache disabled"
    return 1
  fi
}

# rclone and zstd from the flake's pinned nixpkgs: identical on every runner
# (macOS ships neither zstd nor GNU tar-compatible flags reliably).
s3_dir_cache_tool() {
  nix shell --inputs-from . nixpkgs#rclone nixpkgs#zstd --command "$@"
}

# ~ and relative paths become absolute; prints the result.
s3_dir_cache_abspath() {
  local p=$1
  # shellcheck disable=SC2088  # a literal leading tilde is what we match on
  case "$p" in
    "~"|"~/"*) p="$HOME${p#"~"}" ;;
    /*) ;;
    "") return 1 ;;
    *) p="$PWD/$p" ;;
  esac
  printf '%s\n' "$p"
}

# Prints the key to restore: the exact key if its snapshot exists, else the
# newest snapshot under any restore-keys prefix, else nothing.
s3_dir_cache_lookup() {
  local key=$1 restore_keys=$2 prefix hit
  if s3_dir_cache_tool rclone lsf --files-only --include "$key.tar.zst" \
        "$S3_DIR_CACHE_REMOTE" 2>/dev/null | grep -qx "$key.tar.zst"; then
    printf '%s\n' "$key"
    return 0
  fi
  while IFS= read -r prefix; do
    [ -n "$prefix" ] || continue
    hit=$(s3_dir_cache_tool rclone lsf --files-only --format tp \
            --include "${prefix}*.tar.zst" "$S3_DIR_CACHE_REMOTE" 2>/dev/null \
          | sort -r | head -1 | cut -d';' -f2-)
    if [ -n "$hit" ]; then
      printf '%s\n' "${hit%.tar.zst}"
      return 0
    fi
  done <<< "$restore_keys"
  return 0
}
