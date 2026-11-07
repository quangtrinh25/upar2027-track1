"""Offline generic backbones for the prospective competition recipe."""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

import torch
from torch import nn

from upar.candidate2.storage import publish, read_json
from upar.io import sha256_file

MODEL_NAMES = {'A': 'convnextv2_base.fcmae_ft_in22k_in1k_384',
               'B': 'vit_base_patch14_reg4_dinov2.lvd142m', 'C': 'torchvision_convnext_tiny'}
FEATURE_DIMS = {'A': 1024, 'B': 1536, 'C': 768}
IMAGE_SIZES = {'A': (384, 192), 'B': (392, 196), 'C': (384, 192)}
MEAN = (.485, .456, .406)
STD = (.229, .224, .225)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import convnext_tiny
        base = convnext_tiny(weights=None)
        self.features, self.avgpool, self.norm = base.features, base.avgpool, base.classifier[0]
        self.grad_checkpointing = False

    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = bool(enable)

    def forward_features(self, x):
        if self.training and self.grad_checkpointing:
            from torch.utils.checkpoint import checkpoint
            for block in self.features:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            x = self.features(x)
        return self.norm(self.avgpool(x)).flatten(1)


class UpgradeModel(nn.Module):
    def __init__(self, backbone, *, arm, image_size, feature_dim, feature_mode, dropout):
        super().__init__()
        self.backbone, self.arm = backbone, arm
        self.image_size, self.feature_dim = tuple(image_size), int(feature_dim)
        self.feature_mode = feature_mode
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feature_dim, 40)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(bool(mode))
        return self

    def forward_features(self, x):
        # Inference is explicitly FP32; training remains eligible for caller AMP.
        if not self.training:
            with torch.autocast(device_type=x.device.type,enabled=False):
                return self._forward_features(x.float())
        return self._forward_features(x)

    def _forward_features(self, x):
        features = self.backbone.forward_features(x)
        if self.feature_mode == 'cls_patch_mean':
            prefix = int(self.backbone.num_prefix_tokens)
            if features.ndim != 3 or features.shape[1] <= prefix or prefix < 1:
                raise ValueError('DINOv2 representation requires CLS and patch tokens')
            features = torch.cat((features[:, 0], features[:, prefix:].mean(dim=1)), dim=1)
        elif features.ndim > 2:
            features = self.backbone.forward_head(features, pre_logits=True)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError('Unexpected backbone feature dimension')
        return features

    def forward(self, x):
        if not self.training:
            with torch.autocast(device_type=x.device.type,enabled=False):
                logits = self.head(self.forward_features(x))
        else:
            logits = self.head(self.dropout(self.forward_features(x)))
        return {'logits': logits, 'probabilities': logits.float().sigmoid()}


def _load_local(path):
    path = Path(path)
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        return load_file(str(path), device='cpu')
    return torch.load(path, map_location='cpu', weights_only=True)


def _provenance(path, arm):
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + '.json')
    if not sidecar.is_file():
        raise ValueError('Generic pretrained checkpoint provenance sidecar missing')
    record = read_json(sidecar)
    if record.get('arm') != arm or record.get('sha256') != sha256_file(path):
        raise ValueError('Generic pretrained checkpoint provenance changed')
    if record.get('scope') != 'generic_pretraining_no_external_PAR':
        raise ValueError('External PAR initialization forbidden')
    if arm in ('A', 'B'):
        if (record.get('repo_id') != 'timm/' + MODEL_NAMES[arm]
            or not re.fullmatch('[0-9a-f]{40}',str(record.get('revision','')))):
            raise ValueError('Generic pretrained repository/revision mismatch')
    return record


def build_model(arm, cfg, initialize=True):
    if arm not in MODEL_NAMES:
        raise ValueError('Unknown upgrade arm')
    spec = cfg['models'][arm]
    if spec['timm_name'] != MODEL_NAMES[arm] or tuple(spec['image_size']) != IMAGE_SIZES[arm]:
        raise ValueError('Backbone or image shape differs from the registered recipe')
    mode = 'cls_patch_mean' if arm == 'B' else 'global'
    if spec['feature_mode'] != mode or cfg['training']['dropout'] != .2:
        raise ValueError('Feature representation/dropout differs from registered recipe')
    if arm == 'C':
        backbone = TinyBackbone()
    else:
        import timm
        if timm.__version__ != '1.0.22':
            raise RuntimeError('The upgrade requires the pinned timm 1.0.22')
        # Retain the original DINOv2 learned square positional grid. timm
        # resample_abs_pos_embed handles its rectangular grid at forward time.
        kwargs = {'dynamic_img_size': True} if arm == 'B' else {}
        backbone = timm.create_model(spec['timm_name'], pretrained=False, num_classes=0, **kwargs)
    model = UpgradeModel(backbone, arm=arm, image_size=IMAGE_SIZES[arm],
                         feature_dim=FEATURE_DIMS[arm], feature_mode=mode, dropout=.2)
    if initialize:
        record = _provenance(spec['checkpoint'], arm)
        state = _load_local(spec['checkpoint'])
        if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError('Generic checkpoint must be a tensor state dictionary')
        if arm == 'C':
            if state.get('classifier.2.weight', torch.empty(0)).shape != (1000, 768):
                raise ValueError('Tiny initialization is not generic ImageNet1k')
            mapped = {k: v for k, v in state.items() if k.startswith('features.')}
            mapped.update({k.replace('classifier.0.', 'norm.', 1): v for k,v in state.items()
                           if k.startswith('classifier.0.')})
            backbone.load_state_dict(mapped, strict=True)
        else:
            excluded = ('head.weight', 'head.bias') if arm == 'B' else ('head.fc.weight', 'head.fc.bias')
            mapped = {k:v for k,v in state.items() if k not in excluded}
            backbone.load_state_dict(mapped, strict=True)
        model.pretrained_provenance = record
    return model


def download_weights(config_path):
    """Explicit setup only: resolve generic repositories once, download by SHA.

    Never called by build_model, training, preflight, inference or export.
    """
    from .protocol import configuration
    from huggingface_hub import HfApi, hf_hub_download
    cfg = configuration(config_path)
    result = {}
    for arm in ('A','B','C'):
        path = Path(cfg['models'][arm]['checkpoint'])
        sidecar = path.with_suffix(path.suffix + '.json')
        if sidecar.exists():
            result[arm] = _provenance(path, arm)
            continue
        if arm == 'C':
            if not path.is_file():
                raise FileNotFoundError('Existing generic torchvision Tiny checkpoint required')
            record = {'arm':arm, 'scope':'generic_pretraining_no_external_PAR',
                      'source':'torchvision ConvNeXt_Tiny_Weights.IMAGENET1K_V1',
                      'filename':path.name, 'sha256':sha256_file(path)}
        else:
            if path.exists():
                raise ValueError('Untracked new pretrained artifact exists; refusing replacement')
            repo = 'timm/' + MODEL_NAMES[arm]
            revision = HfApi().model_info(repo, revision='main').sha
            source = Path(hf_hub_download(repo, 'model.safetensors', revision=revision,
                                          cache_dir=str(path.parent/'hub')))
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
            try:
                with os.fdopen(fd,'wb') as handle, source.open('rb') as incoming:
                    shutil.copyfileobj(incoming,handle)
                    handle.flush(); os.fsync(handle.fileno())
                os.link(temporary,path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            record = {'arm':arm,'scope':'generic_pretraining_no_external_PAR',
                      'repo_id':repo,'revision':revision,'filename':'model.safetensors',
                      'sha256':sha256_file(path)}
        publish(sidecar,record)
        result[arm] = record
    return result
