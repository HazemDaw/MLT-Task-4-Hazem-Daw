"""Download the published model files and verify their SHA-256 digests."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent


def download():
    manifest = json.loads((ROOT / 'results/checkpoint_manifest.json').read_text())
    destination = ROOT / 'checkpoints'
    destination.mkdir(exist_ok=True)
    for name, entry in manifest.items():
        target = destination / name
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != entry['sha256']:
            temporary = target.with_suffix('.download')
            urllib.request.urlretrieve(entry['url'], temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError(f'Checksum mismatch: {name}')
            temporary.replace(target)
        print(f'Verified {name}')


if __name__ == '__main__':
    download()
