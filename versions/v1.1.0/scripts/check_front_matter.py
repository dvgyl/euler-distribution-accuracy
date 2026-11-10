"""Check the IEEE TAI limits for the title, running head, abstract, impact statement, and keywords."""
import re
import sys
from pathlib import Path

paper = Path(__file__).resolve().parents[1] / 'paper'
main = (paper / 'main.tex').read_text()
preamble = (paper / 'preamble.tex').read_text()


def words(text):
    text = re.sub(r'\$[^$]*\$', 'x', text)
    text = re.sub(r'\\[A-Za-z]+(\{\})?', 'x', text)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'/.,-]*", text))


def block(name):
    return re.search(r'\\begin\{%s\}(.*?)\\end\{%s\}' % (name, name), main, re.S).group(1)


title = re.search(r'\\newcommand\{\\ArticleTitle\}\{(.*?)\}', preamble).group(1)
head = re.search(r'\\markboth\{(.*?)\}', main).group(1)
keywords = [k.strip() for k in block('IEEEkeywords').replace('\n', ' ').rstrip('. ').split(',')]
checks = [
    ('title words', len(title.split()), 1, 15),
    ('running head characters', len(head), 1, 25),
    ('abstract words', words(block('abstract')), 1, 250),
    ('impact statement words', words(block('IEEEImpStatement')), 100, 150),
    ('keywords', len(keywords), 3, 6),
]
failed = False
for name, value, low, high in checks:
    ok = low <= value <= high
    failed |= not ok
    print(f"{'OK  ' if ok else 'FAIL'} {name}: {value} (limit {low} to {high})")
if re.search(r'\b(novel|new)\b', title, re.I):
    failed = True
    print('FAIL title contains "novel" or "new"')
sys.exit(1 if failed else 0)
