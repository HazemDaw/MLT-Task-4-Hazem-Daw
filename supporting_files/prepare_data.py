"""Download and safely extract the original course split; audit every image."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from PIL import Image

FILE_ID = '1lPLlLKhRXvtCG4SNXYflIiy09DMqqPkN'
ARCHIVE_SHA256 = '540915915226479357a1e738bd920c3e19e5a3d8fc56c0622dffa5d682e3631c'


def prepare(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / 'Cats_and_Dogs'
    marker = root / 'extraction.json'
    if marker.exists():
        metadata = json.loads(marker.read_text())
        images = [p for p in destination.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        if len(images) == metadata['images']:
            return destination
        raise ValueError('Extracted dataset is incomplete. Restore the original archive before continuing.')
    archive = root / 'Cats_and_Dogs.zip'
    if not archive.exists():
        import gdown
        gdown.download(id=FILE_ID, output=str(archive), quiet=False)
    if not zipfile.is_zipfile(archive):
        raise ValueError('Download is not a complete ZIP archive.')
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_digest != ARCHIVE_SHA256:
        raise ValueError('Course archive differs from the recorded experiment; check its source/version.')
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            parts = Path(member.filename).parts
            if 'Cats_and_Dogs' not in parts or '__MACOSX' in parts:
                continue
            parts = parts[parts.index('Cats_and_Dogs') + 1:]
            if not parts or member.is_dir() or parts[-1].startswith('.'):
                continue
            target = destination.joinpath(*parts).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError('Unsafe ZIP path')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != member.file_size:
                with zf.open(member) as source, target.open('wb') as out:
                    shutil.copyfileobj(source, out)
    marker.write_text(json.dumps(dict(images=len(list(destination.rglob('*.jpg'))),
                                      archive_sha256=archive_digest), indent=2))
    return destination


def audit(data_root, out_dir):
    data_root, out_dir = Path(data_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in data_root.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})

    def inspect(path):
        parts = path.relative_to(data_root).parts
        with Image.open(path) as im:
            rgb = im.convert('RGB')
            rgb.load()
            width, height = rgb.size
            pixels = hashlib.sha256(f'{width}x{height}'.encode() + rgb.tobytes()).hexdigest()
        return dict(path=path.relative_to(data_root).as_posix(), split=parts[0],
                    label=parts[1], width=width, height=height,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), pixel_sha256=pixels)

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(inspect, paths))
    with (out_dir / 'data_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts, hashes = {}, {}
    for row in rows:
        key = f"{row['split']}/{row['label']}"
        counts[key] = counts.get(key, 0) + 1
        hashes.setdefault(row['pixel_sha256'], []).append(row['path'])
    duplicates = [v for v in hashes.values() if len(v) > 1]
    cross = [v for v in duplicates if len({p.split('/')[0] for p in v}) > 1]
    summary = dict(total=len(rows), counts=counts, unreadable_images=0,
                   duplicate_groups=duplicates, cross_split_duplicates=cross,
                   source=f'https://drive.google.com/file/d/{FILE_ID}/view')
    (out_dir / 'data_audit.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data')
    parser.add_argument('--results', default='results')
    args = parser.parse_args()
    audit(prepare(args.root), args.results)
