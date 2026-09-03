#!/usr/bin/env python3
"""Print the faulting thread of a macOS .ips crash report (temporary CI aid)."""
import json
import sys

raw = open(sys.argv[1]).read()
header, body = raw.split("\n", 1)
report = json.loads(body)
print("exception:", report.get("exception"))
print("termination:", report.get("termination"))
images = report.get("usedImages", [])
threads = report.get("threads", [])
fault = report.get("faultingThread", 0)
thread = threads[fault] if fault < len(threads) else threads[0]
for i, frame in enumerate(thread.get("frames", [])[:80]):
    idx = frame.get("imageIndex", 0)
    image = images[idx].get("name", "?") if idx < len(images) else "?"
    sym = frame.get("symbol", "?")
    off = frame.get("symbolLocation", frame.get("imageOffset", "?"))
    print(f"  #{i:2d} {image:30s} {sym} + {off}")
