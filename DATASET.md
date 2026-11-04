# Dataset used for OmniRestore

The dataset directly supported by this checkout is **CDD-11**, released by the OneRestore authors. It contains paired clean images and 11 individual/composite degradations: haze, rain, snow, low light, low+haze, low+rain, low+snow, haze+rain, haze+snow, low+haze+rain, and low+haze+snow. The project test list additionally evaluates clean images as an identity condition.

Primary sources:

- Model code and dataset instructions: https://github.com/Judith989/omnirestore
- Dataset authors: https://github.com/gy65896/OneRestore
- Official downloadable dataset: https://huggingface.co/datasets/gy65896/CDD-11

## What is being obtained

The official `test.zip` archive is 3,770,337,157 bytes. Expected SHA-256 from the publisher's LFS metadata:

```text
421567fceda89659a6d047fc78d82f77c68703203336d9c795213ba7096646dc
```

It is sufficient to evaluate the pretrained checkpoint. The training archive is 22,403,961,860 bytes and is needed only for retraining/fine-tuning. Snow100K, LOL, and WeatherBench are additional benchmarks mentioned in the model README; they are not required to run the supplied CDD-11 workflow.

The downloader preserves the existing split files, maps archive entries into `data/cdd11/test/<weather>/<filename>`, verifies the archive hash, and writes a provenance manifest to `data/cdd11/test_download.json`. Dataset files and archive bytes are excluded from Git.

## Reproduce acquisition and evaluation

```powershell
.\.venv\Scripts\python.exe fetch_dataset.py --split test
.\.venv\Scripts\python.exe validate_data.py --root data/cdd11 --split test
.\.venv\Scripts\python.exe evaluate_dataset.py --root data/cdd11 --output results_cdd11 --conditioning both --batch-size 8
```

The download command requires internet access. This session required a scoped sandbox escalation for downloads; ordinary local PowerShell has its own network permissions.

Expected evaluation coverage: 2,400 pairs, 200 per condition, comprising 11 degraded conditions plus clean identity. Reports contain all-case and degraded-only aggregates, per-image CSVs, and input/restoration/reference images. Input PSNR is infinite for identical clean pairs; reports explicitly count and exclude infinite values from the finite mean rather than silently treating them as finite scores.

Current acquisition and evaluation outcomes are recorded in `status.md` and `log.md`. The supplied checkpoint's historical shift setting and learned prototype semantics remain unverified; test execution does not by itself reproduce the paper.

## Completed acquisition

The official test archive is now downloaded and checksum verified. All 2,400 test pairs validate and have been evaluated in both modes. See `EVALUATION_RESULTS.md`. Training images have not been downloaded because the pretrained model does not require them.
