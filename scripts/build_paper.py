#!/usr/bin/env python3
"""Compile the complete manuscript with Tectonic."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
SOURCE_DATE_EPOCH = "1793318400"


def main():
    BUILD.mkdir(exist_ok=True)
    compiler = os.environ.get("TECTONIC", "tectonic")
    with tempfile.TemporaryDirectory(prefix="compile-", dir=BUILD) as temporary:
        stage = Path(temporary)
        shutil.copytree(ROOT / "paper", stage / "paper")
        for source in (ROOT / "results").rglob("*"):
            if source.is_file() and source.suffix in {".pdf", ".tex"}:
                target = stage / "results" / source.relative_to(ROOT / "results")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        output = stage / "output"
        output.mkdir()
        result = subprocess.run(
            [compiler, "--untrusted", "--keep-logs", "--outdir", str(output), "main.tex"],
            cwd=stage / "paper", env=dict(os.environ, SOURCE_DATE_EPOCH=SOURCE_DATE_EPOCH),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (BUILD / "compiler_output.txt").write_text(result.stdout.replace(str(stage), "."))
        if (output / "main.log").exists():
            shutil.copyfile(output / "main.log", BUILD / "manuscript.log")
        if result.returncode:
            raise SystemExit("The compiler failed. Read build/compiler_output.txt.")
        shutil.copyfile(output / "main.pdf", ROOT / "manuscript.pdf")
    print("Built manuscript.pdf")


if __name__ == "__main__":
    main()
