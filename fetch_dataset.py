"""Fetch official CDD-11 archives, verify SHA-256, and map official split paths."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import zipfile
import requests

REPO = 'https://huggingface.co/datasets/gy65896/CDD-11'
API = 'https://huggingface.co/api/datasets/gy65896/CDD-11/tree/main'


def download(url, destination, size, sha256):
    partial = destination.with_suffix('.zip.partial')
    if not destination.exists():
        for attempt in range(6):
            offset = partial.stat().st_size if partial.exists() else 0
            try:
                with requests.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                                  stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:
                        offset = 0
                    mode = 'ab' if offset else 'wb'
                    completed, last = offset, offset
                    with partial.open(mode) as output:
                        for chunk in response.iter_content(8 * 1024 * 1024):
                            output.write(chunk)
                            completed += len(chunk)
                            if completed - last >= 128 * 1024 * 1024:
                                print(f'Downloaded {completed / 1e9:.2f}/{size / 1e9:.2f} GB', flush=True)
                                last = completed
                if partial.stat().st_size != size:
                    raise RuntimeError(f'Incomplete download: {partial.stat().st_size}/{size}')
                partial.rename(destination)
                break
            except (requests.RequestException, RuntimeError) as error:
                print(f'Download attempt {attempt + 1} failed: {error}', flush=True)
                if attempt == 5:
                    raise
                time.sleep(2)
    digest = hashlib.file_digest(destination.open('rb'), 'sha256').hexdigest()
    if destination.stat().st_size != size or digest != sha256:
        raise RuntimeError(f'Archive integrity check failed: {destination}')
    print(f'SHA-256 verified: {digest}', flush=True)


def extract_pairs(archive, root, split):
    lists = [root/'splits'/f'cdd11_{split}.txt'] if split == 'test' else [
        root/'splits/cdd11_train.txt', root/'splits/cdd11_val.txt']
    required = set()
    for path in lists:
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 3:
                required.update(parts[:2])
    with zipfile.ZipFile(archive) as z:
        index = {}
        for entry in z.infolist():
            if entry.is_dir():
                continue
            parts = PurePosixPath(entry.filename.replace('\\', '/')).parts
            if len(parts) >= 2:
                key = '/'.join(parts[-2:])
                if key in index:
                    raise RuntimeError(f'Ambiguous archive filename: {key}')
                index[key] = entry
        missing = [name for name in required if '/'.join(PurePosixPath(name).parts[-2:]) not in index]
        if missing:
            raise RuntimeError(f'{len(missing)} split paths not found in archive, examples: {missing[:5]}')
        unpacked = sum(index['/'.join(PurePosixPath(name).parts[-2:])].file_size for name in required)
        if shutil.disk_usage(root).free < unpacked + 1024**3:
            raise RuntimeError('Not enough free space to extract dataset')
        for i, name in enumerate(sorted(required), 1):
            target = (root/name).resolve()
            if not target.is_relative_to(root.resolve()):
                raise RuntimeError(f'Invalid split path: {name}')
            entry = index['/'.join(PurePosixPath(name).parts[-2:])]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                # Never overwrite an existing image silently.
                if target.stat().st_size != entry.file_size:
                    raise RuntimeError(f'Existing file size differs: {target}')
                continue
            with z.open(entry) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)
            if i % 200 == 0:
                print(f'Extracted {i}/{len(required)} referenced images', flush=True)
    return len(required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['test', 'train'], default='test')
    parser.add_argument('--root', type=Path, default=Path('data/cdd11'))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    metadata = requests.get(API, timeout=30)
    metadata.raise_for_status()
    entry = next(item for item in metadata.json() if item['path'] == f'{args.split}.zip')
    downloads = args.root.parent/'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads/entry['path']
    url = f'{REPO}/resolve/main/{entry["path"]}?download=true'
    manifest = {'repository': REPO, 'archive_url': url, 'metadata': entry,
                'split': args.split, 'status': 'download_pending'}
    report = args.root/f'{args.split}_download.json'
    report.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    download(url, archive, entry['size'], entry['lfs']['oid'])
    count = extract_pairs(archive, args.root, args.split)
    manifest.update(status='downloaded_verified_extracted', unique_referenced_images=count)
    report.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
