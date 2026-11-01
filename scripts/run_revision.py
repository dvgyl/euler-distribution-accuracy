#!/usr/bin/env python3
"""Generate the nonradial construction, certificates, and cost experiment."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"build"/"matplotlib"))
sys.path.insert(0,str(ROOT/"src"))
from euler_path.revision_experiments import generate

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"results"/"revision")
    args=parser.parse_args()
    summary=generate(args.output)
    print(json.dumps({"output":str(args.output),
                      "correction":summary["correction_discovery"]["global_minimizer"],
                      "continuous_path_max_difference":summary["continuous_path_max_difference"]},indent=2))
