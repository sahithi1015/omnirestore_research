# OmniRestore action log

Started: 2026-11-04

This file separates completed actions from planned work. Planned work has not been executed. Append an entry after each future meaningful change or validation, including the command, result, and any remaining blocker. This is a manual project log; no automatic background logger has been installed.

## Completed actions

| ID | Date | Action | Result |
| --- | --- | --- | --- |
| A01 | 2026-11-04 | Located and inventoried `OMNIRESTORE RESEARCH`; checked repository instructions and working tree | Identified 14 Python source files, README, configuration, weights, assets, and split lists; no project AGENTS.md found |
| A02 | 2026-11-04 | Reviewed all Python source files and traced imports, loaders, conditioning, losses, training, and evaluation | Identified missing modules, interface mismatches, label-order differences, and correctness concerns |
| A03 | 2026-11-04 | Parsed all Python sources with Python AST | All 14 files passed syntax checks |
| A04 | 2026-11-04 | Checked package availability and imported PyTorch/torchvision | Recorded Python 3.12.10, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, available CUDA, and missing packages |
| A05 | 2026-11-04 | Inspected checkpoint ZIP metadata and loaded checkpoints with restricted loading; allowlisted necessary NumPy types for restoration metadata | Confirmed readable model tensors, epoch 45, fusion projection, identity bottleneck state, and stored metrics |
| A06 | 2026-11-04 | Checked every split-list pair against `data/cdd11` | Zero referenced image pairs present; split counts recorded in status.md |
| A07 | 2026-11-04 | Ran `python test_only_images.py --help` and `python main_train.py --help` | Both failed on missing local modules; no successful inference or training run |
| A08 | 2026-11-04 | Rechecked repository before documentation | Existing untracked `model/__pycache__/` and `utils/__pycache__/` observed after Python startup checks; no model source edits |
| A09 | 2026-11-04 | Created `status.md`, `log.md`, and `RUN_MODEL.md` | Saved assessment, action history, planned repairs, and step-by-step execution procedure |
| A10 | 2026-11-04 | Read back all three documents; checked nonempty contents, balanced code fences, and Git status | Documentation checks passed; Git shows the three new Markdown files and previously observed Python caches; no tracked source changes |

## Setup checklist (partial and blocked items remain unchecked)

- [x] P01: Create an isolated environment and record its Python, PyTorch, torchvision, and CUDA details.
- [ ] P02: Install missing dependencies; preserve the working GPU stack where possible and freeze verified versions after validation.
- [ ] P03: Identify the architecture and shift settings intended for the supplied restoration checkpoint; repair `model.wadt_net` imports.
- [x] P04: Implement the missing checkpoint functions with strict state validation, CPU loading, and explicit resume behavior.
- [x] P05: Verify checkpoint loading and run a synthetic shape/finite-output check.
- [x] P06: Add a single-image inference entry point; document resize, normalization, conditioning, device, and output behavior.
- [x] P07: Restore a real image, inspect output, and record the exact command and output path.
- [x] P08: Obtain paired CDD-11 data and verify split paths and pair counts.
- [ ] P09: Repair evaluation roots, split selection, embedder JSON handling, label mapping, output naming, coverage, and timing if latency is reported.
- [x] P10: Evaluate a small paired sample, then the intended complete test split; record metrics and sample counts.
- [ ] P11: If training is needed, repair unsupported loader arguments, label semantics, normalization, losses, preprocessing, and checkpoint resume behavior.
- [x] P12: If training is needed, run one short training/validation cycle and resume from its checkpoint before a full experiment.
- [x] P13: Save verified dependency versions and commands; update status.md with actual results and remaining limitations.

## Entry template

```text
ID:
Date/time:
Action:
Files changed:
Command/check:
Result:
Remaining blocker or next step:
```

## Setup execution ? 2026-11-04

- A11: Created `.venv` with `--system-site-packages` to reuse the installed CUDA PyTorch stack. This environment depends on those base packages; it is not a standalone deployment environment.
- A12: Attempted pip installation of missing dependencies. It failed because network sockets are forbidden in this session (WinError 10013); no package installation succeeded.
- A13: Added canonical WADNet import and native PyTorch reshape operations. Default is explicitly non-shifted; historical shift setting remains unknown.
- A14: Added restricted checkpoint I/O, strict executed-state loading, and offline ResNet/prototype inference. Eleven unused KAN adapter tensors are excluded explicitly; the saved prototype rows are preserved.
- A15: Added `infer.py`, optional LPIPS handling, native PSNR/SSIM metrics, and execution metadata JSON files. Repaired loader labels, workers, batch size, full test coverage, evaluator paths, and optional embedder JSON handling.
- A16: Corrected pixel-space training metrics/loss normalization and MS-SSIM aggregation; disabled random detached feature targets. Updated auto-generated split logic to group scenes. Existing checkpoint weights and official split lists are preserved.
- A17: All five checked entry points now accept `--help`. Fixed Windows cp1252 console errors encountered during execution.
- A18: CUDA image-only inference succeeded on `assets/teaser.png`. CUDA image-only and fused inference succeeded on a photo crop from `assets/qualitative.png`; output PNGs and configuration JSONs are in `results_setup/`.
- A19: Fused paired evaluation on the figure crop completed: input PSNR 11.8271 / SSIM 0.4314; output PSNR 18.6913 / SSIM 0.5061. LPIPS is unavailable (NaN). These compressed figure crops are execution fixtures, not CDD-11 benchmark measurements.
- A20: Started a one-epoch, single-fixture training/validation check with SmoothL1 only (`--loss_weight 1 0 0 0`) at 32x32. Outcome and resume check to follow.

- A21: One-epoch restoration training and validation completed on the fixture with SmoothL1 only, saving `results_setup/train_ckpts/best.ckpt`. Resuming to epoch 2 completed and saved `results_setup/resumed_ckpts/best.ckpt`. These runs test execution and optimizer state, not quality/generalization.
- A22: Five regression checks passed: strict missing-weight rejection; non-shifted parity and odd-size shifted output; exact optimizer resume; analytic metric values; shared labels and uneven-class full coverage with zero workers.
- A23: CPU inference at 32x32 and shifted CUDA inference at 224x224 succeeded. Image-only paired evaluation and embedder extraction stage 1 succeeded on the fixture. Classification accuracy on its single crop was zero; prototype semantics remain unverified.
- A24: Alternate `main_test_full.py` successfully evaluated the fixture after path/split and strict-load fixes. A later consistency edit aligns its SSIM window with the other paired evaluators and weights the overall mean by sample count.
- A25: Saved dependency snapshots and rewrote `RUN_MODEL.md` around verified commands. Preserved the prior plan as `SETUP_PLAN_ORIGINAL.md`; prepended current status to `status.md`. Added `validate_data.py` to report missing image pairs before evaluation.
- A26: Inspected `results_setup/comparison.png` (input/fused output/figure reference). Output visibly reduces haze; figure crop results are not a benchmark or full-resolution validation.

### Remaining external inputs

Original dataset paths and images have been requested. No original CDD-11 images were downloaded, no full dataset run was launched, and no new embedder was trained. Network restrictions prevent installation of LPIPS/CLIP/KAN packages here. Full research-loss training, learned class semantics, and historical shift settings remain unverified.

- A27: Final syntax parsing passed for all 23 Python source files. Git whitespace check passed; supplied checkpoint binaries have no Git changes. Saved SHA-256 hashes and validation scope in `results_setup/verification.json`.
- A28: Final five-test regression run passed after SSIM consistency and output-naming fixes. The alternate evaluator ran again on the fixture with the aligned seven-pixel SSIM window.
- A29: `validate_data.py --root data/cdd11 --split test` deliberately returned exit status 1: 2,400 listed pairs, 4,800 missing path references, no malformed lines. Full evaluation cannot proceed until original images are supplied.
- A30: Updated completed checklist items and current execution documentation. `metric_result.csv` is the training validation record; Excel output falls back to CSV because no Excel engine is installed. No external messages, deployment, or full training job was started.
- A31: Removed incidental formatting-only changes from the preprocessing script and empty model package initializer; retained the functional setup changes.

## Official dataset acquisition ? 2026-11-04

- A32: Searched primary sources and confirmed CDD-11 is the dataset directly supported by this checkout. Official model repository: https://github.com/Judith989/omnirestore ; dataset authors: https://github.com/gy65896/OneRestore ; official dataset: https://huggingface.co/datasets/gy65896/CDD-11 . Other datasets mentioned in the paper are optional additional benchmarks.
- A33: Confirmed official archive sizes from the Hugging Face API: test.zip 3,770,337,157 bytes; train.zip 22,403,961,860 bytes. Approximately 667 GB free storage was available. Selected the test split first because pretrained evaluation needs no training archive.
- A34: Default sandbox networking failed with WinError 10013. An explicitly scoped elevated metadata request succeeded under the updated approval policy. Started the authorized official test archive download using fetch_dataset.py, with resumable streaming, SHA-256 verification, safe path mapping, and no silent file overwrites. No training archive download has been started.

- A35: Added evaluate_dataset.py for actual paired test evaluation, both conditioning paths, per-image CSV, per-weather metrics/comparisons, explicit finite/infinite PSNR counts, and a degraded-only aggregate. Checked it against the existing fixture and confirmed metric parity with the repaired legacy evaluator.
- A36: Three archive setup tests passed: nested-layout mapping, rejecting missing source files before extraction, and preserving conflicting existing files. Added dataset/download exclusions to .gitignore. The test archive download remains in progress; no full dataset metrics have been measured yet.

- A37: All eight combined dataset/runtime regression tests passed. Saved DATASET.md with official source links, archive sizes/hash, exact acquisition/evaluation commands, and dataset-role explanation. The archive is still downloading; results are not yet reported as real-data measurements.

- A38: Official test.zip download completed. SHA-256 matched publisher metadata: 421567fceda89659a6d047fc78d82f77c68703203336d9c795213ba7096646dc. Extracted 2,400 unique referenced images into data/cdd11/test; provenance saved to data/cdd11/test_download.json.
- A39: validate_data.py --root data/cdd11 --split test passed: 2,400 pairs, 200 per condition, zero missing references, zero invalid split lines. Started complete CUDA evaluation with both conditioning modes at 224x224, batch size 8, output results_cdd11.

- A40: The first full evaluation reached at least 320 real pairs, but CPU SSIM was a bottleneck. Stopped only the identified evaluation process (verified command line) and restarted after switching uniform-window SSIM to summed-area calculations. Five random reference comparisons agreed within 4e-16; all eight regression tests passed again. Model inputs/weights and metric definitions were unchanged. Added optional parallel data loading to the evaluator; final settings are recorded in metrics.json.

- A41: Complete official-data evaluation finished: 2,400/2,400 pairs, 12 conditions x 200 each, CUDA, 224x224, batch size 8, non-shifted, both conditioning modes, 525.3 seconds. Actual data-loader worker count was zero; optional worker support was added while the run was already active.
- A42: All-case output metrics: image-only PSNR 19.443850 / SSIM 0.814978; fused PSNR 29.747568 / SSIM 0.922571. Degraded-only (2,200 cases): image-only 18.628066 / 0.800027; fused 27.285528 / 0.915544. The all-case score includes 200 clean identity pairs. LPIPS was not measured; no paper-reproduction claim.
- A43: Checked saved results: exactly 2,400 unique CSV rows, 12 class counts of 200, 12 comparison images, zero NaN metric entries. Preserved provenance in metrics.json. Updated status.md, RUN_MODEL.md, DATASET.md and added EVALUATION_RESULTS.md. No original training archive download or real retraining was necessary for this successful pretrained test evaluation.
- A44: Verified the optional Windows four-worker data-loader path on 12 real images (one per condition); both modes completed successfully. This separate small check did not replace or alter the full 2,400-pair report.

- A45: User redirected the interrupted training request to test-set execution. No training was started in that turn. Confirmed no active training/evaluation/download jobs. Starting a fresh complete 2,400-pair pretrained evaluation in both conditioning modes; output directory results_test_execution_20261104_142057.

- A46: Fresh test execution completed successfully: results_test_execution_20261104_142057, all 2,400 official pairs, CUDA, 224x224, batch size 8, four data workers, both modes, 310.85 seconds. Verified 2,400 unique CSV rows, 12 classes x 200, 12 comparison images, no NaN metrics. Scores exactly matched the previous complete run: image-only 19.443850 dB / 0.814978 SSIM; fused known-weather 29.747568 / 0.922571 (including 200 clean cases). Original checkpoints and prior reports were preserved.

## Image-upload UI and owner comparison ? 2026-11-04

- A47: Installed Gradio 6.31.0 and dependencies in the project virtual environment; retained the installed torch/torchvision stack. Added requirements-ui.txt and refreshed the environment snapshot. Installed Playwright for validation using the existing local Chrome browser; no browser binary download was needed.
- A48: Added gradio_app.py with image upload, image-only/fused modes, weather selection, before/after slider, downloadable PNG, optional original-dimension output, real dataset examples, optional clean-reference quality measurements, and a separate benchmark tab. The server binds locally to 127.0.0.1:7860 with public sharing disabled. Executed weights are cached, and requests are serialized for GPU use.
- A49: Added benchmark_comparison.py and generated README_COMPARISON.md from the original README values (29.86 PSNR, 0.9244 SSIM) and the latest completed test report. Fused gap is -0.1124 dB / -0.001829 SSIM; image-only gap is -10.4161 / -0.109422. Included settings, clean-case inclusion, degraded-only scores, and limits on claiming exact reproduction. Original benchmark figures were preserved; linked the new docs from README.md and RUN_MODEL.md.
- A50: API checks passed for actual fused upload with clean reference, original-size image-only output, PNG retrieval, and clear missing-image validation. The UI does not invent PSNR/SSIM when no clean reference is supplied. Upload scores are explicitly identified as 8-bit PNG comparisons at 224x224.
- A51: All 12 regression tests passed, including four new checks for dynamically sourced benchmark values/differences, clean-reference validation before model loading, no invented quality scores, and identical-reference metrics.
- A52: Browser checks passed for actual upload/restoration, real example loading with clean reference, PNG download, benchmark figures, and no JavaScript errors. Corrected narrow-screen overflow and stacked mobile columns. Screenshots and browser validation JSON are in results_setup/gradio_*. The live local server remains available at http://127.0.0.1:7860 .
- A53: Final responsive check passed on the running app after adjusting for Gradio CSS scoping: mobile input/result columns stack, both tabs avoid page overflow, and the service returns HTTP 200. Verified the comparison still reads owner values 29.86 / 0.9244. The server remains running at http://127.0.0.1:7860 .

## GitHub publication preparation ? 2026-11-04

- A54: User authorized pushing the complete codebase to https://github.com/sahithi1015/omnirestore.git, especially all Markdown files. Confirmed origin already targets that repository and the remote main branch matches local HEAD before new work.
- A55: Remote metadata reports the connected app has no push permission. A scoped elevated Git dry-run reached GitHub but returned HTTP 403: permission denied to PSaathvika. Used only a per-command safe.directory exception for sandbox ownership; no global trust change, force push, or credential disclosure.
- A56: Removed the blanket results ignore rule so all created reports, CSVs, screenshots, example images, restored outputs, and smoke-training artifacts can be included. Added PUBLISHING.md explaining the publication inventory, external dataset/runtime exclusions, and credential correction. Preparing a file manifest and completeness audit for every Markdown file.

- A57: Completeness audit passed for 158 publication files, including every one of 10 Markdown documents. All generated results are included; no indexed file exceeds 100 MiB. A token/private-key pattern scan found no credential-like content. Preparing PUBLISH_MANIFEST.json with per-file sizes/hashes and a local commit. Push remains blocked by GitHub account authorization.
