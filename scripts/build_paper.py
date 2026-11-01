#!/usr/bin/env python3
"""Build IEEE main, complete proof supplement, and the combined manuscript."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
SOURCE_DATE_EPOCH = "1793491200"

def main():
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as error:
        raise SystemExit("Install the declared pypdf dependency to split the paper.") from error
    BUILD.mkdir(exist_ok=True)
    compiler = os.environ.get("LATEXMK", "latexmk")
    if not shutil.which(compiler):
        raise SystemExit("latexmk and pdfLaTeX are required (standard TeX Live).")
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
            [compiler, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
             "-file-line-error", f"-outdir={output}", "main.tex"],
            cwd=stage / "paper", env=dict(os.environ, SOURCE_DATE_EPOCH=SOURCE_DATE_EPOCH),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (BUILD / "compiler_output.txt").write_text(result.stdout.replace(str(stage), "."))
        if (output / "main.log").exists():
            shutil.copyfile(output / "main.log", BUILD / "manuscript.log")
        if result.returncode:
            raise SystemExit("Compilation failed; see build/compiler_output.txt.")
        log = (output / "main.log").read_text(errors="replace")
        if "There were undefined references" in log or "multiply defined" in log:
            raise SystemExit("Unresolved or duplicate references; see build/manuscript.log.")
        aux = (output / "main.aux").read_text()
        match = re.search(r"\\newlabel\{sec:supplement-start\}\{\{[^}]*\}\{(\d+)\}", aux)
        if not match:
            raise SystemExit("Supplement boundary missing from LaTeX auxiliary file.")
        count = int(match.group(1)) - 1
        reader = PdfReader(output / "main.pdf")
        if not 0 < count < len(reader.pages):
            raise SystemExit("Invalid supplement boundary.")
        shutil.copyfile(output / "main.pdf", ROOT / "manuscript.pdf")
        # The complete file above retains its annotations. Remove them only
        # from this in-memory reader before making standalone page subsets.
        for page in reader.pages:
            page.pop("/Annots", None)
        for name, start, stop, title in [
            ("ieee_tai_main.pdf", 0, count, "Main article"),
            ("ieee_tai_supplement.pdf", count, len(reader.pages), "Supplementary material")]:
            writer = PdfWriter()
            # References crossing the split stay as printed numbers. The
            # combined manuscript retains every live hyperlink; omit link
            # annotations here rather than leave dangling destinations.
            for page_index in range(start, stop):
                writer.add_page(reader.pages[page_index], excluded_keys=["/Annots"])
            writer.add_metadata({"/Title": title + ": When Euler Is Second-Order Accurate in Distribution",
                                 "/Author": "Dev D. Goyal"})
            with (ROOT / name).open("wb") as stream:
                writer.write(stream)
        (BUILD / "paper_pages.txt").write_text(
            f"main={count}\nsupplement={len(reader.pages)-count}\ncombined={len(reader.pages)}\n")
    print(f"Built manuscript.pdf and separate IEEE main ({count} pages) and supplement PDFs.")

if __name__ == "__main__":
    main()
