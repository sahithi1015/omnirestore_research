"""Inference-only embedder: saved prototypes and image stream, no CLIP/KAN startup.

KAN adapter tensors are excluded explicitly and counted in the run report.
They are not executed: released inference reads the saved text_embeddings directly.
"""
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from .labels import LABELS
from utils.ckpt_utils import safe_load, strip_module


class InferenceEmbedder(nn.Module):
    def __init__(self, state, backbone='resnet18'):
        super().__init__()
        self.labels = list(LABELS)
        self.label2idx = {lab: i for i, lab in enumerate(self.labels)}
        # No pretrained downloads; all used tensors must be supplied by checkpoint.
        net = getattr(torchvision.models, backbone)(weights=None)
        width = net.fc.in_features
        mid, out = state['final.weight'].shape[1], state['final.weight'].shape[0]
        self.backbone = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                      net.layer1, net.layer2, net.layer3, net.layer4)
        self.proj = nn.Sequential(nn.Conv2d(width, mid, 1), nn.ReLU(inplace=True))
        self.final = nn.Linear(mid, out)
        self.text_embeddings = nn.Parameter(torch.empty(len(LABELS), out))
        self.load_pretrained_state(state)

    def load_pretrained_state(self, state):
        # The offline KAN adapter is not part of the executed inference graph.
        # Reject every unexpected key outside that explicit adapter namespace.
        used = {k: v for k, v in state.items() if not k.startswith('text_adapter.')}
        self.load_state_dict(used, strict=True)
        self.unused_adapter_keys = [k for k in state if k.startswith('text_adapter.')]

    def forward(self, x):
        return F.normalize(self.final(self.proj(self.backbone(x)).mean((-1, -2))), dim=-1)

    embed_for_style_transfer = forward

    def embed_text_for_style(self, labels):
        labels = labels if isinstance(labels, (list, tuple)) else [labels]
        return self.text_embeddings[[self.label2idx[label] for label in labels]]

    def classify(self, x):
        return self(x) @ F.normalize(self.text_embeddings, dim=-1).T / 0.07


def load_inference_embedder(path, backbone='resnet18'):
    ckpt = safe_load(path)
    return InferenceEmbedder(strip_module(ckpt.get('state_dict', ckpt)), backbone)
