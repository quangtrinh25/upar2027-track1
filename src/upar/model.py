"""ConvNeXt PAR and independent Beta evidence heads (not full UAPAR)."""
import torch
from torch import nn
from torch.nn import functional as F


def evidence_outputs(raw):
    evidence=F.softplus(raw)
    ep,en=evidence[...,0],evidence[...,1]
    strength=ep+en+2
    return {'probabilities':(ep+1)/strength,'uncertainty':2/strength,'evidence_positive':ep,'evidence_negative':en}


def evidential_loss(output,targets,*,epoch,awr_weight=1e-3):
    # Probability BCE is forbidden inside CUDA autocast even for float32 inputs.
    # Keep the Beta loss in float32 while the caller's backbone still uses AMP.
    with torch.autocast(device_type=output['probabilities'].device.type,enabled=False):
        targets=targets.float()
        p=output['probabilities'].float().clamp(1e-6,1-1e-6)
        ep,en=output['evidence_positive'].float(),output['evidence_negative'].float()
        bce=F.binary_cross_entropy(p,targets)
        mse=((targets-p)**2+p*(1-p)/(ep+en+3)).mean()
        incorrect=(targets*en+(1-targets)*ep).mean()
        return bce+min(max(epoch/5,0),1)*mse+awr_weight*incorrect


class PARModel(nn.Module):
    def __init__(self,kind='deterministic',pretrained=False):
        super().__init__()
        from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights
        if kind not in {'deterministic','evidential'}: raise ValueError('Unknown model kind')
        self.kind=kind
        base=convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features=base.features
        self.avgpool=base.avgpool
        self.norm=base.classifier[0]
        self.head=nn.Linear(768,80 if kind=='evidential' else 40)

    def forward(self,x):
        features=self.norm(self.avgpool(self.features(x))).flatten(1)
        raw=self.head(features)
        if self.kind=='evidential': return evidence_outputs(raw.reshape(-1,40,2))
        return {'logits':raw,'probabilities':raw.sigmoid()}
