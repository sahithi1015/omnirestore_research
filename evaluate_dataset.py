"""Evaluate official paired CDD-11 data, saving per-image and per-class results."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from torchvision.utils import save_image
from infer import DEFAULT_EMBEDDER, MEAN, STD, load_models
from utils.dataset_loader import CDD11Dataset, collate_fn
from utils.metrics import peak_signal_noise_ratio, structural_similarity


def mean_report(values):
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    return {'finite_mean': float(array[finite].mean()) if finite.any() else None,
            'finite_count': int(finite.sum()), 'infinite_count': int(np.isinf(array).sum()),
            'nan_count': int(np.isnan(array).sum())}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/cdd11')
    parser.add_argument('--output', default='results_cdd11')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--max-per-class', type=int, help='Optional partial-data execution check')
    parser.add_argument('--conditioning', choices=['image', 'fused', 'both'], default='both')
    parser.add_argument('--shifted', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = CDD11Dataset(args.root, split='test', image_size=(args.size, args.size),
                        max_samples_per_class=args.max_per_class)
    if not len(data):
        raise RuntimeError('No test samples found')
    missing = [p for deg, clear, _ in data.samples for p in (deg, clear) if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} image references are missing; first: {missing[0]}')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                        num_workers=args.workers, collate_fn=collate_fn,
                                        persistent_workers=args.workers > 0,
                                        pin_memory=args.device == 'cuda')
    embedder, restorer, projection = load_models(DEFAULT_EMBEDDER, 'ckpts/best.ckpt', args.device, args.shifted)
    modes = ['image', 'fused'] if args.conditioning == 'both' else [args.conditioning]
    mean = torch.tensor(MEAN, device=args.device)[None, :, None, None]
    std = torch.tensor(STD, device=args.device)[None, :, None, None]
    rows, saved = [], set()
    start = time.perf_counter()
    for i, batch in enumerate(loader):
        inp, target = batch['input'].to(args.device), batch['target'].to(args.device)
        image_emb = embedder(inp)
        tensors = {'input': (inp * std + mean).clamp(0, 1)}
        for mode in modes:
            condition = image_emb
            if mode == 'fused':
                condition = projection(torch.cat([embedder.embed_text_for_style(batch['source_weather']), image_emb], -1))
            restored = restorer(inp, condition)[0]
            if not torch.isfinite(restored).all():
                raise RuntimeError(f'Nonfinite output at batch {i}')
            tensors[mode] = (restored * std + mean).clamp(0, 1)
        tensors['target'] = (target * std + mean).clamp(0, 1)
        arrays = {k: v.cpu().permute(0, 2, 3, 1).numpy() for k, v in tensors.items()}
        for j, weather in enumerate(batch['source_weather']):
            row = {'file': str(Path(batch['file'][j]).relative_to(Path(args.root).resolve())), 'weather': weather}
            for mode in ['input', *modes]:
                row[f'{mode}_psnr'] = peak_signal_noise_ratio(arrays['target'][j], arrays[mode][j])
                row[f'{mode}_ssim'] = structural_similarity(arrays['target'][j], arrays[mode][j])
            rows.append(row)
            if weather not in saved:
                save_image(torch.cat([tensors['input'][j], *[tensors[m][j] for m in modes], tensors['target'][j]], dim=2), output/f'{weather}_comparison.png')
                saved.add(weather)
        if (i + 1) % 10 == 0 or i + 1 == len(loader):
            print(f'Evaluated {len(rows)}/{len(data)} pairs in {time.perf_counter()-start:.1f}s', flush=True)
    metric_names = [key for key in rows[0] if key not in ('file', 'weather')]
    manifest_path = Path(args.root)/'test_download.json'
    source = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    report = {'dataset': 'gy65896/CDD-11 official test' if source and source.get('status') == 'downloaded_verified_extracted' else 'unverified paired data at supplied root',
              'download_manifest': source, 'pair_count': len(rows),
              'class_counts': dict(Counter(row['weather'] for row in rows)),
              'settings': vars(args), 'elapsed_seconds': time.perf_counter()-start,
              'metrics': {key: mean_report([row[key] for row in rows]) for key in metric_names},
              'degraded_only_metrics': {key: mean_report([row[key] for row in rows if row['weather'] != 'clear']) for key in metric_names},
              'per_class': {weather: {key: mean_report([row[key] for row in rows if row['weather']==weather])
                                     for key in metric_names} for weather in saved},
              'lpips': 'not measured: package/weights unavailable',
              'assumptions': ['ImageNet-normalized evaluator pipeline preserved',
                              'prototype labels follow source row order; learned semantics unverified',
                              'shift setting is explicit but historical setting unknown'],
              'paper_reproduction': False}
    with (output/'per_image.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'per_class'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
