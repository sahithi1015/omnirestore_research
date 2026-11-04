"""Strict checkpoint I/O without unrestricted pickle execution."""
from pathlib import Path
import numpy as np
import torch


def safe_load(path, map_location='cpu'):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    # Existing checkpoint metrics use NumPy scalar objects.
    with torch.serialization.safe_globals([
        np._core.multiarray.scalar, np.dtype,
        np.dtypes.Float64DType, np.dtypes.Float32DType,
    ]):
        return torch.load(path, map_location=map_location, weights_only=True)


def strip_module(state):
    return {k.removeprefix('module.'): v for k, v in state.items()}


def load_checkpoint_if_any(model, ckpt_path, map_location='cpu'):
    ckpt = safe_load(ckpt_path)
    state = strip_module(ckpt.get('state_dict', ckpt))
    if hasattr(model, 'load_pretrained_state'):
        model.load_pretrained_state(state)
    else:
        model.load_state_dict(state, strict=True)
    return model.to(map_location)


def load_restore_ckpt_with_optim(device, models_to_load, local_rank=None,
                                 freeze_model=False, ckpt_name=None, lr=1e-4):
    models = {k: m.to(device) for k, m in models_to_load.items()}
    for model in models.values():
        for p in model.parameters():
            p.requires_grad_(not freeze_model)
    groups = [{'params': [p for p in m.parameters() if p.requires_grad], 'label': k}
              for k, m in models.items() if any(p.requires_grad for p in m.parameters())]
    optimizer = torch.optim.AdamW(groups, lr=lr)
    epoch = 0
    if ckpt_name:
        ckpt = safe_load(ckpt_name)
        for name, model in models.items():
            model.load_state_dict(strip_module(ckpt['model_states'][name]), strict=True)
        if ckpt.get('optimizer_state'):
            optimizer.load_state_dict(ckpt['optimizer_state'])
            # Restore semantic labels for legacy optimizer groups when unambiguous.
            if len(optimizer.param_groups) == len(groups):
                for old, new in zip(optimizer.param_groups, groups):
                    old['label'] = new['label']
        epoch = int(ckpt.get('epoch', 0))
    return models['restorer'], models['projection'], models['bottleneck'], optimizer, epoch


def save_checkpoint(models, optimizer, epoch, path, metrics=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {'epoch': int(epoch),
               'model_states': {k: m.state_dict() for k, m in models.items()},
               'optimizer_state': optimizer.state_dict()}
    payload.update({k: float(v) for k, v in (metrics or {}).items()})
    torch.save(payload, path)
