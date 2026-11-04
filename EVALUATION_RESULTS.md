# CDD-11 evaluation results

Completed: 2026-11-04. Source: official gy65896/CDD-11 test.zip, SHA-256 verified. All 2,400 project test pairs passed validation and were evaluated on CUDA. No training was needed.

Configuration: 224x224, batch size 8, non-shifted windows, ImageNet-normalized released-evaluator pipeline, source prototype row names. No LPIPS measurement. Total data/inference/metrics time: 525.3 seconds. Historical shift settings remain unverified; this is a measured local run, not a claim of exact paper reproduction.

| Mode | PSNR (dB), all 2,400 | SSIM, all 2,400 | PSNR, 2,200 degraded | SSIM, 2,200 degraded |
| --- | ---: | ---: | ---: | ---: |
| Image only | 19.4439 | 0.814978 | 18.6281 | 0.800027 |
| Fused image + known weather label | 29.7476 | 0.922571 | 27.2855 | 0.915544 |

The all-case score includes 200 clean identity pairs. Do not compare input finite-PSNR means directly with all-case output means without accounting for those cases: clean inputs have infinite PSNR. The JSON separately records finite, infinite, and NaN counts.

Fused conditioning uses the known dataset weather labels. Its advantage here does not establish the same score for unknown-weather images using image-only conditioning.

Artifacts in `results_cdd11/`:

- `metrics.json`: full source/configuration, aggregate and per-weather measurements.
- `per_image.csv`: 2,400 per-image rows.
- `<weather>_comparison.png`: one example per condition, ordered input / image-only / fused / clean reference.

Re-evaluate:

```powershell
.\.venv\Scripts\python.exe evaluate_dataset.py --root data/cdd11 --output results_cdd11_rerun --conditioning both --batch-size 8 --workers 0
```

Official source: https://huggingface.co/datasets/gy65896/CDD-11 . Download/validation instructions are in `DATASET.md`.
