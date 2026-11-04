# OmniRestore: current results versus owner-reported results

This comparison uses the figures reported by the project owner in the original [README.md](README.md), and our latest completed execution report [results_test_execution_20261104_142057/metrics.json](results_test_execution_20261104_142057/metrics.json). Higher PSNR and SSIM are better. These are restoration-quality metrics, not classification accuracy percentages.

## Overall comparison

| Result | PSNR (dB) | SSIM | PSNR difference | SSIM difference |
| --- | ---: | ---: | ---: | ---: |
| Owner-reported OmniRestore (original README) | 29.8600 | 0.924400 | Reference | Reference |
| Current execution: image + known weather | 29.7476 | 0.922571 | -0.1124 dB | -0.001829 |
| Current execution: image only | 19.4439 | 0.814978 | -10.4161 dB | -0.109422 |

Difference = current execution minus the owner-reported value. Fused mode is **0.1124 dB lower in PSNR** and **0.001829 lower in SSIM** than the original README. Image-only inference uses a different conditioning path and should be interpreted separately.

## What was executed

- Official CDD-11 test archive, SHA-256 verified against publisher metadata.
- 2,400 test pairs: 200 per condition, comprising 11 degradation conditions and one clean identity condition.
- Saved pretrained checkpoints; no real-data training was performed.
- CUDA, 224x224 input resizing, batch size 8, non-shifted windows, ImageNet-normalized released-evaluator pipeline.
- Fused mode uses the known weather labels provided by the dataset. Image-only mode receives no weather label.
- Pixel-space PSNR and uniform-window SSIM; LPIPS was not measured.

The overall table includes 200 clean identity cases. Their input PSNR is infinite, so raw-input finite PSNR averages are not directly comparable with all-case output means.

## Degraded images only

The following scores exclude clean identity cases (2,200 degraded pairs). The original README does not supply a separate degraded-only reference; no reference gap is assigned here.

| Current mode | PSNR (dB) | SSIM |
| --- | ---: | ---: |
| Image + known weather | 27.2855 | 0.915544 |
| Image only | 18.6281 | 0.800027 |

## Limits of this comparison

The owner's README does not fully specify the evaluation protocol behind its table. Historical shift settings and learned prototype semantics remain unverified. Resizing, SSIM settings, clean-case inclusion, and conditioning can affect scores. This is an explicit numerical comparison of the reported and measured results, not a verified reproduction of the owner's experiment. We cannot assign the remaining gap to one cause from the available evidence.

## Use the image-upload frontend

From the project folder:

```powershell
.\.venv\Scripts\python.exe gradio_app.py
```

Open **http://127.0.0.1:7860**. Upload a photo and click **Restore image**. Use **Image only** when the weather is unknown; select **Image + weather label** and the matching condition when it is known. The app displays a before/after slider and a downloadable PNG. It uses the same pretrained inference path as the execution run, with 224x224 processing by default. An optional control resizes the saved output back to original dimensions; it cannot recover detail discarded during processing.

To measure PSNR/SSIM for your own image, upload the corresponding clean reference of the same scene and original dimensions. Without a clean reference, those scores are not available. The UI measures the exported 8-bit PNG against a resized reference; these per-upload scores can differ slightly from the full-test floating-point tensor metrics. The benchmark tab always shows the saved full-test results, not invented scores for a new upload.

Two dataset examples are included to try known degraded/clean pairs. Uploaded images are processed by the local server; a public Gradio share link is not enabled. Restored PNGs and run metadata are saved under `results_uploads/`.

Install the frontend dependencies in another environment with `python -m pip install -r requirements-ui.txt`, after installing the compatible inference dependencies. This workspace already has them installed.

## Reproduce or refresh

```powershell
.\.venv\Scripts\python.exe evaluate_dataset.py --root data/cdd11 --output results_cdd11_rerun --conditioning both --batch-size 8 --workers 4
.\.venv\Scripts\python.exe benchmark_comparison.py
```

The comparison generator reads `current_execution_run.txt` (currently `results_test_execution_20261104_142057`) to select the execution report. To compare a different completed run, set that file to its project-relative result directory before regenerating. The original owner README has been preserved apart from links to the local UI/comparison documentation.
