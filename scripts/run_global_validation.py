#!/usr/bin/env python3
"""Reproduce the exact-Gaussian endpoint and mesh-dependence checks."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"build"/"matplotlib"))
sys.path.insert(0, str(ROOT/"src"))
from euler_path.global_validation import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"results"/"global_validation")
    args = parser.parse_args()
    result = generate(args.output)
    print(json.dumps({"output": str(args.output), "validation": result["validation"],
                      "constants": result["constants"], "runtime_seconds": result["runtime_seconds"]}, indent=2))

