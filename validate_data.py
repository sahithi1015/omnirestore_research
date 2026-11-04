"""Validate all referenced CDD-11 image pairs without loading models."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/cdd11')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    args = parser.parse_args()
    root = Path(args.root)
    path = root/'splits'/f'cdd11_{args.split}.txt'
    if not path.is_file():
        parser.error(f'Missing split list: {path}')
    counts, invalid, missing, total = Counter(), [], [], 0
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3:
            invalid.append(number)
            continue
        total += 1
        counts[parts[2]] += 1
        for name in parts[:2]:
            if not (root/name).is_file():
                missing.append(name)
    print(json.dumps({'root': str(root), 'split': args.split, 'pairs': total,
                      'class_counts': dict(counts), 'invalid_lines': invalid,
                      'missing_path_references': len(missing), 'missing_examples': missing[:5]}, indent=2))
    raise SystemExit(1 if missing or invalid or total == 0 else 0)


if __name__ == '__main__':
    main()
