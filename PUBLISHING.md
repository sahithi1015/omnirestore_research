# Publishing inventory and GitHub push

Target repository: https://github.com/sahithi1015/omnirestore.git

The publication includes every project source file, Markdown document (including the fixture SOURCE.md), dependency file, checkpoint already present in the project, regression test, generated report, CSV, screenshot, example image, smoke-training checkpoint, and restored output currently in the workspace.

Excluded runtime/external assets:

- `.git/`: local repository metadata, maintained by Git.
- `.venv/`, `__pycache__/`, and `*.pyc`: installed dependencies and interpreter caches, recreated from requirements.
- `data/downloads/` and `data/cdd11/test/`: the official 3.77 GB archive and third-party extracted dataset. Their source, SHA-256, download script, split files, and provenance manifest are included; run `fetch_dataset.py --split test` to restore them. The archive exceeds GitHub's normal 100 MiB individual-file limit. No original data has been deleted.
- `data/cdd11/train/`: third-party training data, if downloaded later.

`PUBLISH_MANIFEST.json` records the publication file set, sizes, and SHA-256 hashes. It excludes itself to avoid a recursive hash. A scan verifies every Markdown file outside the runtime environment is in the Git index.

## Authentication

The configured GitHub credential currently belongs to PSaathvika, and GitHub returned HTTP 403 for pushing to sahithi1015/omnirestore. The connected GitHub app also reports no push permission for that repository.

Authenticate as the repository owner (or an account granted write access) on this computer, then push the prepared commit:

```powershell
git credential-manager github login --username sahithi1015 --device --force
git -c credential.username=sahithi1015 push origin main
```

Do not paste passwords or access tokens into chat. If Git reports a workspace ownership warning, use a per-command trust exception for this project path rather than changing broad global ownership settings.
