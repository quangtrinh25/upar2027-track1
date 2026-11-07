"""dg_v1 backbones. Tiny keeps PARModel keys; base models use local timm weights."""
from __future__ import annotations

import torch
from torch import nn

from upar.model import PARModel

SPECS = {
    'tiny_bce': {'backbone': 'tiny', 'loss': 'bce', 'image_size': (256, 128), 'backbone_lr': 1e-4, 'drop_path': .2},
    'tiny_weighted': {'backbone': 'tiny', 'loss': 'weighted_bce', 'image_size': (256, 128), 'backbone_lr': 1e-4, 'drop_path': .2},
    'dinov2_b': {'backbone': 'B', 'loss': 'bce', 'image_size': (252, 126), 'backbone_lr': 1e-5, 'drop_path': .1,
                 'timm_name': 'vit_base_patch14_reg4_dinov2.lvd142m', 'checkpoint': 'data/pretrained/competition_upgrade/B.safetensors',
                 'feature_dim': 1536, 'feature_mode': 'cls_patch_mean'},
    'convnextv2_b': {'backbone': 'A', 'loss': 'bce', 'image_size': (256, 128), 'backbone_lr': 2e-5, 'drop_path': .2,
                     'timm_name': 'convnextv2_base.fcmae_ft_in22k_in1k_384', 'checkpoint': 'data/pretrained/competition_upgrade/A.safetensors',
                     'feature_dim': 1024, 'feature_mode': 'global'},
    'dinov2_l': {'backbone': 'L', 'loss': 'bce', 'image_size': (252, 126), 'backbone_lr': 1e-5, 'drop_path': .2,
                 'timm_name': 'vit_large_patch14_reg4_dinov2.lvd142m', 'checkpoint': '/home/milis/LinuxData/upar2027/pretrained/L.safetensors',
                 'feature_dim': 2048, 'feature_mode': 'cls_patch_mean'},
    'clip_l': {'backbone': 'CL', 'loss': 'bce', 'image_size': (252, 126), 'backbone_lr': 1e-5, 'drop_path': .1,
               'timm_name': 'vit_large_patch14_clip_336.openai', 'checkpoint': '/home/milis/LinuxData/upar2027/pretrained/CL.safetensors',
               'feature_dim': 2048, 'feature_mode': 'cls_patch_mean', 'input_norm': 'clip'},
    'siglip_l': {'backbone': 'SL', 'loss': 'bce', 'image_size': (256, 128), 'backbone_lr': 1e-5, 'drop_path': .1,
                 'timm_name': 'vit_large_patch16_siglip_256.webli', 'checkpoint': '/home/milis/LinuxData/upar2027/pretrained/SL.safetensors',
                 'feature_dim': 1024, 'feature_mode': 'global', 'input_norm': 'siglip'},
}
# Same model and recipe, different seed (head init, drop path, data order, augmentation draws).
SPECS['clip_l_s2'] = {**SPECS['clip_l'], 'seed': 43}
# dg_v4 layer-wise LR decay: the last block learns at 3x the flat dg_v1 rate, decaying toward the embeddings
# (bottom ~6-7% of the top), so early pretrained features drift less.
SPECS['dinov2_b_llrd'] = {**SPECS['dinov2_b'], 'backbone_lr': 3e-5, 'llrd': .8}
SPECS['dinov2_l_llrd'] = {**SPECS['dinov2_l'], 'backbone_lr': 3e-5, 'llrd': .9}
SPECS['clip_l_llrd'] = {**SPECS['clip_l'], 'backbone_lr': 3e-5, 'llrd': .9}
# dg_v5 = dg_v4 + random erasing (p .5, 2-15% area) + within-group softmax CE (weight .5); see v5.py.
for _arm in ('dinov2_b', 'dinov2_l', 'clip_l'):
    SPECS[f'{_arm}_v5'] = {**SPECS[f'{_arm}_llrd'], 'recipe': 'v5', 'erase': .5, 'group_ce': .5}
HEAD_LR = 1e-3
IMAGENET_MEAN, IMAGENET_STD = (.485, .456, .406), (.229, .224, .225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
INPUT_NORMS = {'clip': (CLIP_MEAN, CLIP_STD), 'siglip': ((.5, .5, .5), (.5, .5, .5))}
WEIGHT_DECAY = .05


class TimmPAR(nn.Module):
    """timm backbone plus a 40-logit linear head; no gradient checkpointing."""

    def __init__(self, backbone, *, feature_dim, feature_mode, input_norm=None):
        super().__init__()
        self.backbone, self.feature_mode = backbone, feature_mode
        self.head = nn.Linear(feature_dim, 40)
        # Inputs are always ImageNet-normalized; CLIP re-normalizes inside the model so every pipeline stays unchanged.
        scale, shift = torch.ones(3), torch.zeros(3)
        if input_norm is not None:
            mean, std = (torch.tensor(v) for v in INPUT_NORMS[input_norm])
            scale = torch.tensor(IMAGENET_STD) / std
            shift = (torch.tensor(IMAGENET_MEAN) - mean) / std
        self.register_buffer('input_scale', scale.view(1, 3, 1, 1), persistent=input_norm is not None)
        self.register_buffer('input_shift', shift.view(1, 3, 1, 1), persistent=input_norm is not None)

    def features(self, x):
        tokens = self.backbone.forward_features(x * self.input_scale + self.input_shift)
        if self.feature_mode == 'cls_patch_mean':
            prefix = int(self.backbone.num_prefix_tokens)
            return torch.cat((tokens[:, 0], tokens[:, prefix:].mean(dim=1)), dim=1)
        return self.backbone.forward_head(tokens, pre_logits=True)

    def forward(self, x):
        logits = self.head(self.features(x))
        return {'logits': logits, 'probabilities': logits.float().sigmoid()}


def scale_tiny_drop_path(model, maximum):
    """torchvision assigns block i probability .1*i/(n-1); rescale linearly to `maximum`."""
    from torchvision.ops import StochasticDepth
    layers = [m for m in model.modules() if isinstance(m, StochasticDepth)]
    top = max(m.p for m in layers)
    for m in layers: m.p = m.p * maximum / top
    return model


def build(arm, *, initialize=True):
    spec = SPECS[arm]
    if spec['backbone'] == 'tiny':
        if initialize: torch.hub.set_dir('data/pretrained')
        return scale_tiny_drop_path(PARModel('deterministic', pretrained=initialize), spec['drop_path'])
    import timm
    from upar.competition_upgrade.models import _load_local, _provenance
    vit = spec['backbone'] in ('B', 'L', 'CL', 'SL')
    kwargs = {'dynamic_img_size': True} if vit else {}
    backbone = timm.create_model(spec['timm_name'], pretrained=False, num_classes=0,
                                 drop_path_rate=spec['drop_path'], **kwargs)
    if initialize:
        _provenance(spec['checkpoint'], spec['backbone'])
        state = _load_local(spec['checkpoint'])
        excluded = ('head.weight', 'head.bias') if vit else ('head.fc.weight', 'head.fc.bias')
        backbone.load_state_dict({k: v for k, v in state.items() if k not in excluded}, strict=True)
    return TimmPAR(backbone, feature_dim=spec['feature_dim'], feature_mode=spec['feature_mode'], input_norm=spec.get('input_norm'))


def is_head(name):
    return name.startswith('head.')


def no_decay(name, parameter):
    return parameter.ndim < 2 or name.endswith('.bias') or any(k in name for k in ('pos_embed', 'cls_token', 'reg_token', 'gamma', 'layer_scale'))


def layer_id(name, depth):
    """LLRD depth: embeddings 0, ViT block i -> i + 1, everything after the blocks -> depth + 1."""
    rest = name.removeprefix('backbone.')
    if rest.startswith('blocks.'): return int(rest.split('.')[1]) + 1
    if rest.startswith(('patch_embed', 'cls_token', 'reg_token', 'pos_embed', 'norm_pre')): return 0
    return depth + 1


def param_groups(model, arm):
    spec, groups = SPECS[arm], {}
    rate = spec.get('llrd')
    depth = len(model.backbone.blocks) if rate else 0
    for name, parameter in model.named_parameters():
        scale = rate ** (depth + 1 - layer_id(name, depth)) if rate else 1.
        lr = HEAD_LR if is_head(name) else spec['backbone_lr'] * scale
        decay = 0. if no_decay(name, parameter) else WEIGHT_DECAY
        groups.setdefault((lr, decay), []).append(parameter)
    return [{'params': params, 'lr': lr, 'base_lr': lr, 'weight_decay': decay} for (lr, decay), params in groups.items()]
