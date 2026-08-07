#!/usr/bin/env python3

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def build_and_run(crate: Path, target: Path, value: int) -> str:
    env = os.environ.copy()
    env.update(
        {
            "CARGO_INCREMENTAL": "0",
            "CARGO_TARGET_DIR": str(target),
            "REUSSIR_CACHE_PROBE_VALUE": str(value),
            "RUSTC_WRAPPER": env.get("REUSSIR_SCCACHE", "sccache"),
        }
    )
    subprocess.run(
        ["cargo", "build", "--locked", "--quiet"],
        cwd=crate,
        env=env,
        check=True,
    )
    suffix = ".exe" if os.name == "nt" else ""
    executable = target / "debug" / f"native-archive-cache-probe{suffix}"
    return subprocess.run(
        [executable], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    crate = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="reussir-native-cache-") as tmp:
        target = Path(tmp) / "target"
        first = build_and_run(crate, target, 1)
        second = build_and_run(crate, target, 2)

    if (first, second) != ("1", "2"):
        print(
            "native archive cache invalidation failed: "
            f"expected ('1', '2'), got ({first!r}, {second!r})",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
