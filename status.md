# Current execution status ? 2026-11-04

**The pretrained model now runs on the downloaded official CDD-11 test set.** All 2,400 pairs validate, and complete image-only/fused CUDA evaluation succeeded. The 3.77 GB official archive matched its publisher SHA-256; extracted images are under `data/cdd11/test` and provenance is in `data/cdd11/test_download.json`.

| Mode | PSNR / SSIM, all 2,400 pairs | PSNR / SSIM, 2,200 degraded pairs |
| --- | --- | --- |
| Image only | 19.4439 / 0.814978 | 18.6281 / 0.800027 |
| Fused image + known weather label | 29.7476 / 0.922571 | 27.2855 / 0.915544 |

The all-case mean includes 200 clean identity pairs. Evaluation used 224x224 inputs, non-shifted windows, batch size 8, CUDA, and the preserved ImageNet-normalized evaluator pipeline. Fused results use the dataset's known weather labels. No LPIPS was measured, and exact paper reproduction is not claimed.

Artifacts: `results_cdd11/metrics.json`, `results_cdd11/per_image.csv`, and 12 weather comparison PNGs. The CSV has 2,400 unique rows, with 200 cases per condition. Eight regression tests pass. Existing checkpoint files were preserved. Earlier CPU/CUDA execution and short SmoothL1 training/resume checks also passed.

Completed setup repairs include checkpoint helpers, offline image/prototype inference, model imports/reshapes, shared labels, evaluation paths/coverage/workers, metric handling, and training normalization/loss fixes. See the action log for scope and validations.

Remaining optional work: downloading the 22.4 GB training archive and real retraining if requested; LPIPS packages/weights; CLIP/KAN packages/assets for new embedder training; validation of historical shift settings and learned prototype semantics. None is required to rerun the completed pretrained CDD-11 evaluation. The environment reuses the installed CUDA stack through `.venv --system-site-packages`.

Use `RUN_MODEL.md` for commands, `DATASET.md` for data provenance, `EVALUATION_RESULTS.md` for measured results, and `log.md` for actions.

**Latest execution:** A fresh full test run completed in 310.85 seconds with four data workers. Reports and 12 comparison images: `results_test_execution_20261104_142057`. The metrics exactly match the earlier full run. No training was started.

**Image-upload frontend:** Gradio is available at http://127.0.0.1:7860 while gradio_app.py is running. It supports upload, both conditioning modes, before/after viewing, PNG download, and optional reference-based PSNR/SSIM. Restart with `.\.venv\Scripts\python.exe gradio_app.py`. Owner-versus-current tables and gap explanations are in README_COMPARISON.md and the benchmark tab. Twelve tests pass; browser upload/download and responsive layout checks passed.

---

# Original assessment (historical; superseded by the status above)

# OmniRestore implementation status

Assessment date: 2026-11-04

## Verdict

The architecture is implemented and pretrained weights are included. This checkout is not ready for direct execution. It needs integration fixes, missing dependencies, input data, and successful end-to-end validation. Implementing the architecture from scratch is unnecessary; retraining is not a prerequisite for trying the supplied weights.

Scope: all 14 Python files, README, supplied configuration, dataset split lists, and checkpoint contents were reviewed. No model source code was changed during this assessment. No successful restoration run or paper-result reproduction has been demonstrated.

## What is implemented

| Component | Current state |
| --- | --- |
| Restoration backbone | Implemented in `model/wadt_net_shift.py` and `model/wadt_net_no_shift.py` |
| Weather embedder | ResNet image encoder, CLIP text prototypes, and KAN adapter implemented |
| Restoration training | Training, validation, losses, and checkpoint calls exist; integration is broken |
| Embedder training | Training and hyperparameter sweep exist; integration and label issues remain |
| Evaluation | Image-only, image-plus-weather-label, and embedder evaluation scripts exist |
| Pretrained weights | Restoration and embedder checkpoint files are present and readable |
| Simple single-image application | No dedicated arbitrary-image inference command supplied |

## Verified environment and artifacts

- Python: 3.12.10. README proposes Python 3.10; this difference alone is not a demonstrated failure.
- PyTorch: 2.11.0+cu128; torchvision: 0.26.0+cu128.
- CUDA is available. Detected GPU: NVIDIA RTX 2000 Ada Generation.
- All 14 Python files pass AST syntax parsing. Syntax success does not establish runtime correctness.
- `ckpts/best.ckpt`: 37,865,053 bytes; epoch 45; contains `model_states`, optimizer state, PSNR, and SSIM.
- Restoration state contains 2,609,038 tensor elements. The fusion projection contains 524,800; its weight shape is `(512, 1024)`. The bottleneck state is empty, consistent with an identity module. State tensor counts include any buffers and are not a measured deployment footprint.
- Embedder checkpoint: 65,845,563 bytes; text prototype shape `(12, 512)` and image final-layer weight shape `(512, 1024)`.
- Checkpoints were inspected using `weights_only=True`; the restoration checkpoint required allowlisting specific NumPy scalar/dtype objects.
- Restoration checkpoint records approximately PSNR 16.9899 and SSIM 0.7286. These are stored metadata, not reproduced measurements. They do not establish comparability with the README's benchmark results.

## Execution blockers

1. **Missing model import target.** `main_train.py`, `main_test_full.py`, `test_only_images.py`, and `test_with_text.py` import `model.wadt_net`, which does not exist. Choose and expose the correct supplied architecture. Shift behavior is not established by checkpoint tensor shapes alone.
2. **Missing checkpoint utilities.** `utils/ckpt_utils.py` is absent. Restoration training requires `load_restore_ckpt_with_optim` and `save_checkpoint`; the two documented inference scripts require `load_checkpoint_if_any`.
3. **Missing packages.** Environment discovery found no `timm`, `einops`, `kan`, `open_clip`, `lpips`, `cv2`, `skimage`, or `fvcore`. Neither `openpyxl` nor `xlsxwriter` is installed for pandas Excel output. Some packages are only needed for particular scripts. No pinned dependency manifest is supplied.
4. **No dataset images.** Only split lists exist under `data/cdd11`. There are 11,609 training, 1,289 validation, and 2,400 test entries; zero referenced image pairs are present there.
5. **Unsupported training loader argument.** Both training scripts pass `acdc_root`, but `load_combined_dataset` accepts CDD-11 arguments only. They will fail even after import repairs.
6. **Missing embedder evaluation configuration.** Stage 1 of `eval_embedder.py` unconditionally reads a JSON file beside its checkpoint. That JSON is absent.
7. **Broken alternate evaluator path.** `main_test_full.py` ignores `--cdd-root`, uses `/glade/derecho/scratch/njoku/CVPR/data/cdd11`, and does not request `split="test"`. It also accepts missing/unexpected checkpoint keys through `strict=False` without reporting them.

## Correctness issues requiring validation

- **Label ordering differs:** dataset `wid=1` means `haze`, whereas embedder index 1 means `snow`. Numeric IDs are used directly in embedder training and evaluation. The intended label semantics of the supplied checkpoint must be established before changing prototype rows or trusting class metrics.
- **Normalization and metrics:** restoration training computes PSNR/SSIM with `data_range=1` on ImageNet-normalized tensors. The perceptual loss applies ImageNet normalization again. The backbone also applies its own affine transform; preserve existing checkpoint behavior until a consistent input contract is validated.
- **Conditioning differs:** restoration training fuses text and image embeddings through a learned projection. Image-only evaluation passes the image embedding directly. Matching dimensions does not establish matching quality.
- **Model initialization downloads:** building the embedder loads CLIP and ImageNet ResNet weights before loading the supplied state. It is not presently a self-contained offline inference path.
- **Saved visualizations can overwrite:** different weather folders can share basenames; saving only the basename can replace earlier outputs.
- **Test-set coverage:** the loader can downsample classes to the smallest class size. A claim of full-set evaluation needs explicit sample-count verification.
- **Training loss review:** the ground-truth feature extractor produces detached targets from randomly initialized convolutions that are not trained by the restoration optimizer. The MS-SSIM aggregation also needs review before claiming faithful retraining.
- **Preprocessing mismatch:** `dataset_preprocess.py` creates a separate `val/` directory, while automatic loader splitting reads pairs under `train/` and writes lists. Select one consistent split workflow before training.

## Executed startup checks

```text
python test_only_images.py --help
ModuleNotFoundError: No module named 'model.wadt_net'

python main_train.py --help
ModuleNotFoundError: No module named 'utils.ckpt_utils'
```

## Completion criteria

- Required entry points start and checkpoints load strictly with the intended architecture.
- A real input image produces a finite, viewable restored output.
- Paired evaluation runs with consistent normalization, verified labels, and explicit sample counts.
- Environment versions and exact commands are recorded.
- If retraining is required, a short training-and-resume run succeeds before a full run.

See `RUN_MODEL.md` for the ordered procedure and `log.md` for completed actions and planned work.
