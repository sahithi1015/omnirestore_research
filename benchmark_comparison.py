"""Compare measured execution metrics with the original README's reported values."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_comparison(root=ROOT):
    readme = (root/'README.md').read_text(encoding='utf-8')
    line = next(line for line in readme.splitlines() if line.startswith('| **OmniRestore**'))
    cells = [cell.replace('**', '').strip() for cell in line.strip('|').split('|')]
    reference = {'psnr': float(cells[2]), 'ssim': float(cells[3])}
    pointer = root/'current_execution_run.txt'
    folder = pointer.read_text().strip() if pointer.exists() else 'results_cdd11'
    path = (root/folder/'metrics.json').resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Execution report must be inside the project')
    report = json.loads(path.read_text(encoding='utf-8'))
    return reference, report, path


def comparison_table(reference, report):
    lines = ['| Result | PSNR (dB) | SSIM | PSNR difference | SSIM difference |',
             '| --- | ---: | ---: | ---: | ---: |',
             f'| Owner-reported OmniRestore (original README) | {reference["psnr"]:.4f} | {reference["ssim"]:.6f} | Reference | Reference |']
    for mode, label in [('fused', 'Current execution: image + known weather'),
                        ('image', 'Current execution: image only')]:
        psnr = report['metrics'][f'{mode}_psnr']['finite_mean']
        ssim = report['metrics'][f'{mode}_ssim']['finite_mean']
        lines.append(f'| {label} | {psnr:.4f} | {ssim:.6f} | {psnr-reference["psnr"]:+.4f} dB | {ssim-reference["ssim"]:+.6f} |')
    return '\n'.join(lines)


def write_readme(root=ROOT):
    reference, report, path = load_comparison(root)
    degraded = report['degraded_only_metrics']
    content = f'''# OmniRestore: current results versus owner-reported results

This comparison uses the figures reported by the project owner in the original [README.md](README.md), and our latest completed execution report [{path.relative_to(root).as_posix()}]({path.relative_to(root).as_posix()}). Higher PSNR and SSIM are better. These are restoration-quality metrics, not classification accuracy percentages.

## Overall comparison

{comparison_table(reference, report)}

Difference = current execution minus the owner-reported value. Fused mode is **{reference['psnr']-report['metrics']['fused_psnr']['finite_mean']:.4f} dB lower in PSNR** and **{reference['ssim']-report['metrics']['fused_ssim']['finite_mean']:.6f} lower in SSIM** than the original README. Image-only inference uses a different conditioning path and should be interpreted separately.

## What was executed

- Official CDD-11 test archive, SHA-256 verified against publisher metadata.
- {report['pair_count']:,} test pairs: 200 per condition, comprising 11 degradation conditions and one clean identity condition.
- Saved pretrained checkpoints; no real-data training was performed.
- CUDA, {report['settings']['size']}x{report['settings']['size']} input resizing, batch size {report['settings']['batch_size']}, non-shifted windows, ImageNet-normalized released-evaluator pipeline.
- Fused mode uses the known weather labels provided by the dataset. Image-only mode receives no weather label.
- Pixel-space PSNR and uniform-window SSIM; LPIPS was not measured.

The overall table includes 200 clean identity cases. Their input PSNR is infinite, so raw-input finite PSNR averages are not directly comparable with all-case output means.

## Degraded images only

The following scores exclude clean identity cases (2,200 degraded pairs). The original README does not supply a separate degraded-only reference; no reference gap is assigned here.

| Current mode | PSNR (dB) | SSIM |
| --- | ---: | ---: |
| Image + known weather | {degraded['fused_psnr']['finite_mean']:.4f} | {degraded['fused_ssim']['finite_mean']:.6f} |
| Image only | {degraded['image_psnr']['finite_mean']:.4f} | {degraded['image_ssim']['finite_mean']:.6f} |

## Limits of this comparison

The owner's README does not fully specify the evaluation protocol behind its table. Historical shift settings and learned prototype semantics remain unverified. Resizing, SSIM settings, clean-case inclusion, and conditioning can affect scores. This is an explicit numerical comparison of the reported and measured results, not a verified reproduction of the owner's experiment. We cannot assign the remaining gap to one cause from the available evidence.

## Use the image-upload frontend

From the project folder:

```powershell
.\\.venv\\Scripts\\python.exe gradio_app.py
```

Open **http://127.0.0.1:7860**. Upload a photo and click **Restore image**. Use **Image only** when the weather is unknown; select **Image + weather label** and the matching condition when it is known. The app displays a before/after slider and a downloadable PNG. It uses the same pretrained inference path as the execution run, with 224x224 processing by default. An optional control resizes the saved output back to original dimensions; it cannot recover detail discarded during processing.

To measure PSNR/SSIM for your own image, upload the corresponding clean reference of the same scene and original dimensions. Without a clean reference, those scores are not available. The UI measures the exported 8-bit PNG against a resized reference; these per-upload scores can differ slightly from the full-test floating-point tensor metrics. The benchmark tab always shows the saved full-test results, not invented scores for a new upload.

Two dataset examples are included to try known degraded/clean pairs. Uploaded images are processed by the local server; a public Gradio share link is not enabled. Restored PNGs and run metadata are saved under `results_uploads/`.

Install the frontend dependencies in another environment with `python -m pip install -r requirements-ui.txt`, after installing the compatible inference dependencies. This workspace already has them installed.

## Reproduce or refresh

```powershell
.\\.venv\\Scripts\\python.exe evaluate_dataset.py --root data/cdd11 --output results_cdd11_rerun --conditioning both --batch-size 8 --workers 4
.\\.venv\\Scripts\\python.exe benchmark_comparison.py
```

The comparison generator reads `current_execution_run.txt` (currently `{path.parent.name}`) to select the execution report. To compare a different completed run, set that file to its project-relative result directory before regenerating. The original owner README has been preserved apart from links to the local UI/comparison documentation.
'''
    (root/'README_COMPARISON.md').write_text(content, encoding='utf-8')


if __name__ == '__main__':
    write_readme()
