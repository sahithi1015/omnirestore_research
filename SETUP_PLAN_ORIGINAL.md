# Procedure to make OmniRestore functional

Current state: implemented architecture and supplied weights; execution is blocked. The steps below describe work still required. Commands that use repaired or new scripts only become usable after those repairs are completed. No code repair, package installation, data download, or model run was performed while writing this guide.

Start with pretrained inference. Retraining is optional and should follow a working inference and evaluation path.

## 1. Open the project and create an isolated environment

Use PowerShell:

```powershell
Set-Location 'C:\Users\E028.11\Documents\OMNIRESTORE RESEARCH'
git status --short
py -0p
```

The README targets Python 3.10. If it is installed:

```powershell
py -3.10 -m venv .venv
```

Otherwise, the detected Python 3.12 can be tried first:

```powershell
python -m venv .venv
```

Use `.\.venv\Scripts\python.exe` explicitly in the following commands, so PowerShell activation is unnecessary. A normal new virtual environment does not inherit the packages in the current working GPU environment.

## 2. Install dependencies and verify the device

Install PyTorch and torchvision as a compatible pair using the PyTorch installation selector at <https://pytorch.org/get-started/locally/> for Windows, pip, and the desired compute platform. The existing environment already has a working CUDA build; record its versions before replacing anything. Run the selector's pip command through `.\.venv\Scripts\python.exe -m pip` in the new environment. Do not assume the installed CUDA-enabled build will be reproduced by an unspecified default installation.

Then install the project packages from the README plus packages required by the inspected code:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install timm einops pykan open_clip_torch lpips pillow numpy matplotlib opencv-python scikit-image pandas scikit-learn tqdm fvcore openpyxl
.\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
.\.venv\Scripts\python.exe -c "import einops, kan, open_clip, lpips, skimage, fvcore, openpyxl; print('Imports OK')"
```

These are initial installation commands, not a tested version lock. KAN checkpoint compatibility must be checked against the installed `pykan` version. Record installation outcomes in `log.md`.

## 3. Repair the restoration model import

All restoration scripts expect `model/wadt_net.py`, but that file is missing.

Required change:

- Determine whether the supplied checkpoint was trained with shifted or non-shifted windows. Ask the model's training metadata or author if the repository cannot establish it; tensor shapes alone do not identify shift settings.
- Add `model/wadt_net.py` to export the intended `WADNet`, or update imports consistently.
- If supporting both variants, expose an explicit architecture/shift option and save that choice with results. Avoid silently choosing a variant and claiming checkpoint fidelity.

For example, once the shifted implementation is confirmed as the intended implementation, a wrapper can contain:

```python
from .wadt_net_shift import WADNet
```

That implementation defaults `use_shifted_wsa` to `False`. Pass the confirmed setting explicitly; changing the import alone does not enable shifted windows.

## 4. Restore checkpoint loading and saving

Implement `utils/ckpt_utils.py` with the interfaces already called by the scripts:

- `load_checkpoint_if_any`: load the embedder's raw state dictionary or supported wrapped state, strip a leading `module.` prefix, load strictly, and fail clearly when a required checkpoint is missing.
- `load_restore_ckpt_with_optim`: load or initialize restorer, projection, and bottleneck, build the optimizer without duplicate parameters, restore its state when resuming, and return the five values expected by `main_train.py`.
- `save_checkpoint`: save all three module states, optimizer state, epoch, and metrics using a format compatible with the loaders.

Load on CPU first and move the modules to the selected device. Prefer restricted tensor loading; the existing restoration file includes NumPy scalar metadata that requires a narrow allowlist or conversion to plain Python values. Preserve strict loading errors rather than silently accepting incompatible weights.

Confirm that the supplied checkpoint contains:

```text
model_states.restorer
model_states.projection  # Linear(1024, 512)
model_states.bottleneck  # Empty state for Identity
```

The embedder constructor currently initializes CLIP, KAN, and a pretrained ResNet before checkpoint loading. Initial execution may download external weights. A later inference-only constructor can skip unnecessary initialization downloads and use saved prototypes, provided it reproduces the same image encoder and state layout.

## 5. Check startup and model tensor flow

After steps 2–4:

```powershell
.\.venv\Scripts\python.exe test_only_images.py --help
.\.venv\Scripts\python.exe test_with_text.py --help
.\.venv\Scripts\python.exe main_train.py --help
```

Next, perform a short synthetic check using the intended model variant, strict checkpoint loading, an evaluation-mode embedder, an input of shape `(1, 3, 224, 224)`, and a condition vector of shape `(1, 512)`.

Acceptance: output has shape `(1, 3, 224, 224)`, contains finite values, and loading reports no missing or unexpected keys. A synthetic check validates execution and shapes, not restoration quality.

## 6. Add a real single-image inference command

The existing test scripts require clean targets and dataset structure. Add `infer.py` for ordinary image use, with an explicit interface such as:

```text
--input --output --embedder-model-path --best-ckpt --device --height --width
--conditioning image|fused
--weather <supported weather label, required for fused mode>
```

Required behavior:

1. Read RGB input and retain its original dimensions.
2. Apply the checkpoint-compatible resize and normalization. The existing evaluation pipeline uses ImageNet normalization for both embedder and restorer inputs; preserve that behavior initially and validate it before changing the input contract.
3. Strictly load the embedder and restorer. For fused mode, also load the projection and identity bottleneck.
4. In fused mode, concatenate text and image embeddings, then apply the saved projection. This follows the supplied restoration training path.
5. In image mode, document that the supplied evaluator uses the image embedding directly; assess quality separately from fused mode.
6. Run in evaluation/inference mode, convert output back to image space consistently with the selected pipeline, clamp for saving, and write an RGB output.
7. State whether the result is resized or restored to original dimensions. Resizing back does not recover details lost during resizing.

First run on one real image and inspect input/output side by side. Record command, conditioning mode, architecture, output path, runtime, and visible problems in `log.md`. Do not claim full functionality until this succeeds.

## 7. Prepare paired evaluation data

Obtain the CDD-11 images separately and arrange them as expected by the supplied loader:

```text
data/cdd11/
  train/
    clear/
    haze/
    rain/
    snow/
    ...other weather folders...
  test/
    clear/
    haze/
    rain/
    snow/
    ...other weather folders...
  splits/
    cdd11_train.txt
    cdd11_val.txt
    cdd11_test.txt
```

Each split entry contains a degraded path, clean path, and weather label. For example:

```text
test/haze/00002.png test/clear/00002.png haze
```

Both files must exist relative to `data/cdd11`. Verify every listed pair before launching evaluation. The supplied validation list references files under `train/`; a separate `val/` directory is not required for this list-based workflow.

Do not run `dataset_preprocess.py` blindly: it requires an ACDC path and creates a separate validation directory that the automatic loader split does not consume directly. Use the supplied lists with matching data, or adapt preprocessing and list generation together.

## 8. Repair evaluation correctness

Before reporting metrics:

- Make `main_test_full.py` use `args.cdd_root` and pass `split="test"`.
- Use strict checkpoint loading and the confirmed architecture/shift settings.
- Fix the missing JSON handling in `eval_embedder.py`: supply accurate metadata or allow explicit backbone and embedding dimension arguments without a sidecar file.
- Establish the checkpoint's prototype semantics. Use one class mapping for dataset labels and evaluation; do not reorder checkpoint prototypes without evidence. The provided training script used mismatched numeric IDs, so the saved checkpoint's label names cannot be assumed correct merely from current source order.
- Calculate PSNR/SSIM/LPIPS in the intended pixel range. Avoid changing checkpoint preprocessing and metric conversion independently.
- Verify that balancing does not remove samples from an intended full-test run; report actual sample and class counts.
- Include weather or relative paths in saved filenames to prevent overwrites.
- If measuring GPU latency, use warm-up and CUDA synchronization; distinguish per-image from per-batch timing.

## 9. Run a small evaluation, then the complete split

After the preceding repairs, use this PowerShell command for fused evaluation:

```powershell
$embedderPath = '.\logs\embedder_resnet18_bs64_optadamw_lr1.00e-05_cw1.0_conw0.1_temp0.07_20260101-190539.pt'
.\.venv\Scripts\python.exe test_with_text.py `
  --embedder-model-path $embedderPath `
  --best-ckpt .\ckpts\best.ckpt `
  --cdd-root .\data\cdd11 `
  --output .\results_fused `
  --bs 2 `
  --num-works 0 `
  --image-size-h 224 `
  --image-size-w 224 `
  --save-n 10
```

First point `--cdd-root` to a separate small dataset root with its own small split file and valid pairs. Keep the original lists intact. After this succeeds, use the full root above.

The current loader forces at least one evaluation worker even with `--num-works 0`, and halves the requested evaluation batch size. Repair worker handling if true single-process loading is needed; record the effective batch size.

For image-only comparison, run the same arguments with `test_only_images.py` and a separate output directory. The text script uses known weather class names from dataset folders; it is not a general free-text prompt interface.

Acceptance: finite outputs, inspectable saved images, valid per-image metrics, verified split coverage, and separately reported conditioning modes. Compare benchmark claims only when dataset, preprocessing, model settings, and metric definitions match.

## 10. Repair and validate training only if needed

Pretrained inference does not require this step. If retraining or fine-tuning is the goal:

1. Remove unsupported `acdc_root` arguments from both training scripts, or implement actual ACDC support deliberately.
2. Unify numeric class labels before starting embedder training. Decide how checkpoint label semantics affect reuse of its learned prototypes and fusion weights.
3. Define tensor ranges for the embedder, restorer, losses, and metrics. Eliminate unintended double normalization; use pixel-space metrics for validation.
4. Review the MS-SSIM aggregation and the detached, randomly initialized ground-truth feature extractor. Choose justified loss definitions and save the final configuration.
5. Fix checkpoint directory creation, strict resume handling, optimizer parameter groups, and resumed best-metric tracking.
6. Keep validation scenes separate from training scenes when multiple degradation variants share the same clean source image. The current pair-level splitting can place variants of one scene in both splits.
7. Run one short training/validation cycle and a resume check. Confirm finite loss, gradients on intended trainable modules, updated weights, saved checkpoints, and reproducible loading.
8. Start full training only after these checks pass. Narrow the embedder sweep explicitly: its defaults describe 19,200 configurations, not a single training run.

## 11. Save the working setup and update project status

After successful validation:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt
git status --short
```

Record the selected model variant, preprocessing, class mapping, conditioning mode, checkpoints, data source/split, package versions, device, commands, and measured results. Update `status.md` and complete the relevant `log.md` checklist items.

The project is functional for inference when a documented command restores a real image using compatible pretrained weights. Training readiness and reproduction of published results require their own completed checks.
