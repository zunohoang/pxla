"""Package deliverables without virtualenv, caches, QA images or private photos."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Include reproducible UCI baseline, trained model and editable documents."""
    selected = []
    for name in ['README.md', 'START_DEMO.bat', 'app.py', 'run_experiments.py',
                 'requirements.txt', 'requirements-lock.txt', 'requirements-docs.txt',
                 'pytest.ini', '.gitignore']:
        selected.append(ROOT / name)
    for folder in ['src', 'scripts', 'tests', 'models', 'results', 'data', 'docs']:
        for path in (ROOT / folder).rglob('*'):
            if not path.is_file() or any(p in {'__pycache__', 'qa', 'tmp', 'field', 'annotations', 'masks'}
                                         for p in path.relative_to(ROOT).parts):
                continue
            selected.append(path)
    out = ROOT / 'LeafLab_BTL.zip'
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(selected)):
            archive.write(path, 'LeafLab/' + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError('Corrupt archive entry: ' + bad)
    print(f'{out}: {len(selected)} files; {out.stat().st_size/1024**2:.1f} MiB')


if __name__ == '__main__':
    main()
