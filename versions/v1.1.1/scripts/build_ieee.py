"""Build article and supplement PDFs with resolved references."""
from pathlib import Path
import re
import subprocess
from pypdf import PdfReader, PdfWriter

root = Path(__file__).resolve().parents[1]
paper = root / 'paper'
build = root / 'build'
build.mkdir(exist_ok=True)

def expanded(path):
    text = path.read_text()
    for match in list(re.finditer(r'\\input\{([^}]+)\}', text)):
        child = (path.parent / match.group(1)).with_suffix('.tex')
        if child.exists():
            text = text.replace(match.group(0), expanded(child))
    return text

entries = {}
for entry in re.split(r'(?m)(?=^@)', (paper / 'references.bib').read_text()):
    match = re.match(r'@\w+\s*\{([^,]+),', entry)
    if match:
        entries[match.group(1)] = entry.strip() + '\n'
for name in ['main', 'supplement']:
    content = expanded(paper / (name + '.tex'))
    keys = set()
    for group in re.findall(r'\\cite\w*(?:\[[^\]]*\]){0,2}\{([^}]+)\}', content):
        keys.update(key.strip() for key in group.split(','))
    missing = keys - entries.keys()
    if missing:
        raise SystemExit(f'Bibliography entries missing for {name}: {sorted(missing)}')
    (paper / (name + '-references.bib')).write_text('\n'.join(entries[k] for k in sorted(keys)))

for iteration in range(3):
    for name in ['main', 'supplement']:
        result = subprocess.run(['tectonic', '-X', 'compile', name + '.tex', '--outdir', str(build),
                                 '--keep-logs', '--keep-intermediates'], cwd=paper, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (build / f'{name}-pass{iteration+1}.txt').write_text(result.stdout)
        if result.returncode:
            raise SystemExit(f'{name} compilation failed. See build/{name}-pass{iteration+1}.txt.')
        lines = (build / (name + '.aux')).read_text().splitlines()
        labels = [line for line in lines if line.startswith('\\newlabel{')]
        (build / (name + '-labels.aux')).write_text('\n'.join(labels) + '\n')

for name in ['main', 'supplement']:
    log = (build / (name + '.log')).read_text()
    if any(message in log for message in ['undefined references', 'multiply defined', 'Citation `', 'Reference `']):
        raise SystemExit(f'{name} has unresolved references. See build/{name}.log.')
    reader = PdfReader(build / (name + '.pdf'))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page, excluded_keys=['/Annots'])
    writer.add_metadata({'/Title': ('Supplementary Material: ' if name == 'supplement' else '') +
                         'A Two-Grid Characterization of Euler Sampling Accuracy in Flow Matching', '/Author': 'Dev D. Goyal'})
    with (root / f'ieee_tai_{name}.pdf').open('wb') as stream:
        writer.write(stream)
    print(f'{name}: {len(reader.pages)} pages.')
