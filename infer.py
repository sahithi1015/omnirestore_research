"""Offline pretrained image restoration. Run --help for options."""
import argparse
import json
from pathlib import Path
import time
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from model.inference_embedder import load_inference_embedder
from model.labels import LABELS
from model.wadt_net import WADNet
from utils.ckpt_utils import safe_load, strip_module

DEFAULT_EMBEDDER = 'logs/embedder_resnet18_bs64_optadamw_lr1.00e-05_cw1.0_conw0.1_temp0.07_20260101-190539.pt'
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_models(embedder_path, restorer_path, device, shifted=False):
    embedder = load_inference_embedder(embedder_path).to(device).eval()
    states = safe_load(restorer_path)['model_states']
    restorer = WADNet(use_shifted_wsa=shifted).to(device).eval()
    restorer.load_state_dict(strip_module(states['restorer']), strict=True)
    projection = torch.nn.Linear(1024, 512).to(device).eval()
    projection.load_state_dict(strip_module(states['projection']), strict=True)
    torch.nn.Identity().load_state_dict(states['bottleneck'], strict=True)
    return embedder, restorer, projection


@torch.inference_mode()
def restore(image, models, device, size=(224, 224), conditioning='image', weather=None):
    embedder, restorer, projection = models
    resized = image.convert('RGB').resize((size[1], size[0]), Image.Resampling.BICUBIC)
    x = TF.normalize(TF.to_tensor(resized), MEAN, STD).unsqueeze(0).to(device)
    condition = embedder(x)
    if conditioning == 'fused':
        condition = projection(torch.cat([embedder.embed_text_for_style(weather), condition], -1))
    output = restorer(x, condition)[0]
    if not torch.isfinite(output).all():
        raise RuntimeError('Restoration produced nonfinite values')
    mean = output.new_tensor(MEAN).view(1, 3, 1, 1)
    std = output.new_tensor(STD).view(1, 3, 1, 1)
    return TF.to_pil_image((output * std + mean).clamp(0, 1)[0].cpu())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--embedder-model-path', default=DEFAULT_EMBEDDER)
    parser.add_argument('--best-ckpt', default='ckpts/best.ckpt')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--conditioning', choices=['image', 'fused'], default='image')
    parser.add_argument('--weather', choices=LABELS)
    parser.add_argument('--shifted', action='store_true', help='Enable shifts; original training setting is unknown')
    parser.add_argument('--original-size', action='store_true', help='Resize saved result back to original dimensions')
    args = parser.parse_args()
    if min(args.height, args.width) < 8:
        parser.error('height and width must be at least 8')
    if args.conditioning == 'fused' and args.weather is None:
        parser.error('fused conditioning requires --weather')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    with Image.open(args.input) as im:
        image = im.convert('RGB')
    models = load_models(args.embedder_model_path, args.best_ckpt, device, args.shifted)
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    output = restore(image, models, device, (args.height, args.width), args.conditioning, args.weather)
    elapsed = time.perf_counter() - start
    if args.original_size:
        output = output.resize(image.size, Image.Resampling.BICUBIC)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path)
    report = {**vars(args), 'device': device, 'input_size': image.size,
              'output_size': output.size, 'elapsed_seconds_including_transfer': elapsed,
              'checkpoint_load': 'strict for all executed tensors',
              'unused_kan_tensors': len(models[0].unused_adapter_keys),
              'assumptions': ['shift setting unverified', 'prototype names use released source row order',
                              'ImageNet preprocessing preserved from released evaluator'],
              'benchmark_validation': False}
    path.with_suffix(path.suffix + '.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
