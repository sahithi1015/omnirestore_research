# Run OmniRestore ? working setup

Updated: 2026-11-04. Pretrained inference now runs locally on CPU and CUDA. The original procedure is preserved in `SETUP_PLAN_ORIGINAL.md`. This file replaces the earlier unexecuted plan with tested instructions.

## 1. Open the project

```powershell
Set-Location 'C:\Users\E028.11\Documents\OMNIRESTORE RESEARCH'
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The environment uses Python 3.12 and inherits the existing CUDA PyTorch packages through `--system-site-packages`. No package installation succeeded: this session prohibits outbound network sockets. The offline inference path requires only the installed torch, torchvision, NumPy, and Pillow packages. `requirements-inference.txt` records their versions; `requirements-lock.txt` snapshots the complete inherited environment.

## 2. Restore an image

```powershell
.\.venv\Scripts\python.exe infer.py --input 'path\to\your_image.png' --output 'results\restored.png' --device cuda
```

Use `--device cpu` if needed. Checkpoints default to the supplied paths. Output defaults to 224x224 RGB; `--height`, `--width`, and `--original-size` control sizing. Each output gets a `.png.json` configuration report. Timing includes processing/transfers and is not a warmed benchmark.

For a sample already available here:

```powershell
.\.venv\Scripts\python.exe infer.py --input data/cdd11/test/haze/00002.png --output results/restored.png --device cuda
```

For the training-style fusion path, specify a supported weather label:

```powershell
.\.venv\Scripts\python.exe infer.py --input data/cdd11/test/haze/00002.png --output results/restored_fused.png --conditioning fused --weather haze --device cuda
```

These sample commands now use a real downloaded CDD-11 haze image. `results_cdd11/haze_comparison.png` shows input, image-only output, fused output, and clean reference from left to right. The older figure-crop execution fixture remains separately under `results_setup`.

Default shift is off, matching legacy script defaults. Add `--shifted` to test shifted windows. Both execute, but the supplied checkpoint does not identify its training shift setting. Weather prototype names use the source row order; their learned semantics have not been independently validated. ImageNet preprocessing is preserved from the supplied evaluator.

## 3. Verify data before evaluation

```powershell
.\.venv\Scripts\python.exe validate_data.py --root data/cdd11 --split test
```

This returns a nonzero exit status when pairs are missing. The official CDD-11 test archive has now been downloaded, SHA-256 verified, and extracted. All 2,400 test pairs pass path validation. Training images have not been downloaded because pretrained evaluation does not require them. See `DATASET.md` for the official source and acquisition command. Preserve the official lists. The fixture under `results_setup/fixture` is deliberately separate.

## 4. Run paired evaluation

```powershell
$embedderPath = 'logs/embedder_resnet18_bs64_optadamw_lr1.00e-05_cw1.0_conw0.1_temp0.07_20260101-190539.pt'
.\.venv\Scripts\python.exe test_with_text.py --embedder-model-path $embedderPath --best-ckpt ckpts/best.ckpt --cdd-root results_setup/fixture --output results_setup/evaluate_again --bs 1 --num-works 0 --save-n 1
```

Use `--cdd-root data/cdd11` for the downloaded official test set. For saved per-image/per-class reports from both modes, run `.\.venv\Scripts\python.exe evaluate_dataset.py --root data/cdd11 --output results_cdd11 --conditioning both --batch-size 8`. Substitute `test_only_images.py` for image-only evaluation. Effective batch size and worker count now match the arguments, and test classes are not downsampled. The alternate `main_test_full.py` also runs with `--restore-ckpt` instead of `--best-ckpt`.

PSNR and default uniform-window SSIM are implemented locally. LPIPS is reported as NaN with a warning when its package is unavailable; no LPIPS measurement is implied. Installing `lpips` and obtaining its pretrained weights is still required to measure it.

## 5. Run regression checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Eight checks cover archive path mapping, missing/conflicting dataset files, plus strict executed-weight loading, shifted/odd-size forward passes, parity between the two non-shifted implementations, deterministic optimizer resume, metric reference values, shared label mapping, worker settings, and complete uneven-class test coverage.

## 6. Optional training

A one-epoch training/validation run followed by a checkpoint resume has already succeeded on the tiny figure fixture. This verifies execution only. The fixture deliberately overlaps training and validation and must not be used to assess generalization.

The command used for the first epoch was:

```powershell
.\.venv\Scripts\python.exe main_train.py --embedder-model-path $embedderPath --save-model-path results_setup/train_ckpts --cdd-train-root results_setup/fixture --output results_setup/train --epoch 1 --bs 1 --num-works 0 --image_size 32 32 --loss_weight 1 0 0 0
```

Resume from that checkpoint by adding `--restore-model-path results_setup/train_ckpts/best.ckpt`, setting `--epoch 2`, and choosing a separate save/output directory.

For actual training, first supply real, scene-separated training and validation data. The checked offline configuration uses SmoothL1 only. Nonzero VGG contrastive/perceptual weights need downloadable pretrained VGG weights. MS-SSIM aggregation and pixel conversion were repaired; untrained random feature targets were disabled. These training corrections change the original objective and do not constitute paper reproduction.

Training a new embedder still requires `pykan`, `open_clip_torch`, pretrained CLIP/ResNet assets, and actual data. Its default grid is 19,200 runs: explicitly restrict every hyperparameter list before launching. Pretrained image restoration does not require new embedder training.

## Original setup-step outcomes

| Original step | Outcome |
| --- | --- |
| 1: Environment | Completed using inherited installed GPU packages |
| 2: Dependencies | Offline inference works; package installation blocked by network restrictions |
| 3: Model imports | Repaired; both shift modes execute, historical setting unknown |
| 4: Checkpoint I/O | Implemented; executed tensors load strictly, resume tested |
| 5: Tensor/startup checks | Passed |
| 6: Single-image inference | Implemented and executed on CPU/CUDA |
| 7: Dataset | Official CDD-11 test set downloaded, hash verified, all 2,400 pairs validated; training archive not needed for pretrained evaluation |
| 8: Evaluation | Paths, labels, metrics, batch/workers, sample coverage repaired; learned label semantics remain unverified |
| 9: Evaluation runs | Both fixture paths passed; complete 2,400-pair official test evaluation passed; results saved under results_cdd11 |
| 10: Training | Offline one-epoch and resume checks passed; full research training not performed |
| 11: Records | Status, action log, dependency snapshots, and per-image metadata saved |

## Official dataset sources

See [DATASET.md](DATASET.md). The official release is https://huggingface.co/datasets/gy65896/CDD-11 . No user-provided dataset path is now needed for pretrained test evaluation.

## Completed real-data evaluation

Both conditioning modes completed on all 2,400 official pairs. Fused: 29.7476 dB PSNR / 0.922571 SSIM; image-only: 19.4439 / 0.814978. These means include 200 clean identity pairs. See `EVALUATION_RESULTS.md` for degraded-only scores and exact settings. No retraining was necessary.

## Upload images through Gradio

Run `.\.venv\Scripts\python.exe gradio_app.py` and open http://127.0.0.1:7860 . Choose image-only or image-plus-weather, upload a photo, restore it, and download the PNG. Optional clean-reference upload enables per-image PSNR/SSIM. The benchmark tab compares the saved complete execution with the original README. Full details: `README_COMPARISON.md`.
