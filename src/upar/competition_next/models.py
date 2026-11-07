"""Offline DINOv3 ConvNeXt-T feature architecture and strict generic releases.

Architecture follows Meta's dinov3/models/convnext.py (DINOv3 License
Agreement) and HF's modeling_dinov3_convnext.py / converter (Apache-2.0).
This implementation does not grant access to, or accept terms for, weights.
"""
from __future__ import annotations
import json
import importlib.util
import re
from pathlib import Path
import torch
from torch import nn
from upar.io import sha256_file, hash_object
from .artifacts import write_once_json

MODEL_ID = 'facebook/dinov3-convnext-tiny-pretrain-lvd1689m'
REVISION = '10d30274b4d445111e2d5bf75ac93bbd94db274b'
DINO_RAW_SHA = 'bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2'
DINO_RAW_SIZE = 111299216
IMAGENET_SHA = '983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__(); self.weight=nn.Parameter(torch.ones(channels)); self.bias=nn.Parameter(torch.zeros(channels))
    def forward(self,x):
        mean=x.mean(1,keepdim=True); variance=(x-mean).square().mean(1,keepdim=True)
        return (x-mean)/torch.sqrt(variance+1e-6)*self.weight[:,None,None]+self.bias[:,None,None]


class DinoBlock(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dwconv=nn.Conv2d(dim,dim,7,padding=3,groups=dim)
        self.norm=nn.LayerNorm(dim,eps=1e-6)
        self.pwconv1=nn.Linear(dim,4*dim); self.act=nn.GELU(); self.pwconv2=nn.Linear(4*dim,dim)
        self.gamma=nn.Parameter(torch.full((dim,),1e-6))
    def forward(self,x):
        value=self.dwconv(x).permute(0,2,3,1)
        value=self.pwconv2(self.act(self.pwconv1(self.norm(value))))*self.gamma
        return x+value.permute(0,3,1,2)


class DinoConvNeXtTiny(nn.Module):
    """Exact GAP + final LN representation, no stochastic depth or downloads."""
    def __init__(self):
        super().__init__()
        dims=(96,192,384,768); depths=(3,3,9,3)
        self.downsample_layers=nn.ModuleList([nn.Sequential(nn.Conv2d(3,96,4,stride=4),ChannelLayerNorm(96))])
        for left,right in zip(dims,dims[1:]):
            self.downsample_layers.append(nn.Sequential(ChannelLayerNorm(left),nn.Conv2d(left,right,2,stride=2)))
        self.stages=nn.ModuleList([nn.Sequential(*(DinoBlock(dim) for _ in range(depth))) for dim,depth in zip(dims,depths)])
        self.norm=nn.LayerNorm(768,eps=1e-6)
    def forward_features(self,x):
        for down,stage in zip(self.downsample_layers,self.stages): x=stage(down(x))
        return self.norm(x.mean((-2,-1)))
    def forward(self,x): return self.forward_features(x)


class ImageNetTiny(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import convnext_tiny
        base=convnext_tiny(weights=None)
        self.features=base.features; self.avgpool=base.avgpool; self.norm=base.classifier[0]
    def forward_features(self,x): return self.norm(self.avgpool(self.features(x))).flatten(1)


class NextModel(nn.Module):
    def __init__(self,backbone,dropout=0.):
        super().__init__(); self.backbone=backbone; self.dropout=nn.Dropout(dropout); self.head=nn.Linear(768,40)
    def forward_features(self,x): return self.backbone.forward_features(x)
    def forward(self,x):
        features=self.forward_features(x); logits=self.head(self.dropout(features))
        return {'features':features,'logits':logits,'probabilities':logits.float().sigmoid()}


def convert_dino_state(state, model, *, format='native'):
    """Invert only the documented HF converter; strict keys/shapes, no fallback."""
    if not isinstance(state,dict) or format not in ('native','huggingface'):
        raise ValueError('Unknown DINO checkpoint format')
    mapped={}
    for key,value in state.items():
        if format=='huggingface':
            if key.startswith('model.'): key=key[6:]
            key=re.sub(r'^stages\.(\d+)\.downsample_layers\.(\d+)\.',r'downsample_layers.\1.\2.',key)
            key=re.sub(r'^stages\.(\d+)\.layers\.(\d+)\.',r'stages.\1.\2.',key)
            key=key.replace('depthwise_conv','dwconv').replace('pointwise_conv','pwconv').replace('layer_norm','norm')
        if format=='native' and key in ('norms.3.weight','norms.3.bias'):
            canonical=key.replace('norms.3.','norm.')
            if canonical not in state or not torch.equal(value,state[canonical]):
                raise ValueError('Duplicated native final normalization disagrees')
            continue
        if key in mapped: raise ValueError('Duplicate converted keys')
        mapped[key]=value
    expected=model.state_dict()
    if set(mapped)!=set(expected): raise ValueError('DINO state keys differ from exact Tiny architecture')
    if any(not isinstance(mapped[k],torch.Tensor) or mapped[k].shape!=v.shape or not torch.isfinite(mapped[k]).all() for k,v in expected.items()):
        raise ValueError('DINO state shape/finite validation failed')
    return mapped


def prove_dino_origin(protocol,weights_path,format,upstream_weights_path,*,model=None):
    from .dino_origin import authenticated_state,state_digest
    if upstream_weights_path is None:
        raise RuntimeError('Authenticated upstream DINO raw weights are required')
    root=Path(protocol['artifact_root']).resolve()
    raw=Path(upstream_weights_path); runtime=Path(weights_path)
    if any(not path.resolve().is_relative_to(root) for path in (raw,runtime)):
        raise ValueError('DINO upstream/runtime weights must remain in new artifact root')
    original=authenticated_state(raw,expected_sha256=DINO_RAW_SHA,expected_size=DINO_RAW_SIZE)
    backbone=DinoConvNeXtTiny() if model is None else model
    expected=convert_dino_state(original,backbone,format='huggingface')
    actual=convert_dino_state(torch.load(runtime,map_location='cpu',weights_only=True),backbone,format=format)
    if any(actual[k].dtype!=expected[k].dtype or not torch.equal(actual[k],expected[k]) for k in expected):
        raise ValueError('DINO runtime conversion differs from authenticated upstream tensors')
    return {'upstream_weights_path':str(raw),'upstream_sha256':DINO_RAW_SHA,'upstream_size':DINO_RAW_SIZE,
            'model_id':MODEL_ID,'revision':REVISION,'weights_sha256':sha256_file(runtime),
            'runtime_format':format,'converted_state_sha256':state_digest(expected),
            'proof':'direct_tensor_equality','tensor_count':len(expected)}


def require_dino_release(protocol):
    path=Path(protocol['artifact_root'])/'reports/dino_model_release.json'
    if not path.is_file(): raise RuntimeError('DINO model release missing: official access, weights and CPU parity remain unmeasured')
    record=json.loads(path.read_text())
    required={'status':'completed','protocol_sha256':protocol['protocol_sha256'],'model_id':MODEL_ID,'revision':REVISION,
              'scope':'generic_pretraining_no_external_PAR','access_granted':True,'license_acknowledged_by_user':True}
    if any(record.get(k)!=v for k,v in required.items()): raise ValueError('DINO release identity/access/protocol mismatch')
    weights=Path(record['weights_path'])
    if not weights.resolve().is_relative_to(Path(protocol['artifact_root']).resolve()) or sha256_file(weights)!=record.get('weights_sha256'):
        raise ValueError('DINO release weights path/hash mismatch')
    origin=record.get('origin',{})
    proof=prove_dino_origin(protocol,weights,record['format'],origin.get('upstream_weights_path'))
    if proof!=origin: raise ValueError('DINO release upstream conversion proof differs')
    if record.get('license')!='DINOv3 License Agreement': raise ValueError('DINO release license missing')
    for name in ('reference_parity','portable_compatibility'):
        evidence=record.get(name,{})
        if evidence.get('status')!='passed' or evidence.get('weights_sha256')!=record['weights_sha256']:
            raise RuntimeError('DINO release CPU reference/portable compatibility not measured')
        report=Path(evidence.get('report_path',''))
        if not report.resolve().is_relative_to(Path(protocol['artifact_root']).resolve()) or not report.is_file() or sha256_file(report)!=evidence.get('report_sha256'):
            raise ValueError('DINO release CPU evidence hash mismatch')
        detail=json.loads(report.read_text())
        if detail.get('status')!='passed' or detail.get('protocol_sha256')!=protocol['protocol_sha256'] or detail.get('weights_sha256')!=record['weights_sha256']:
            raise ValueError('DINO CPU evidence identity mismatch')
        if detail.get('origin')!=proof or detail.get('actual_pretrained_weights') is not True:
            raise ValueError('DINO CPU evidence upstream conversion proof differs')
        if name=='portable_compatibility' and not str(detail.get('torch_version','')).startswith('2.4.1'):
            raise RuntimeError('DINO organizer Torch2.4.1 compatibility missing')
    return record


def build_model(protocol,arm,*,initialize=True):
    if arm not in ('dino_a','dino_b','ema_aug'): raise ValueError('Unknown exact registered arm')
    if arm.startswith('dino'):
        backbone=DinoConvNeXtTiny(); release=require_dino_release(protocol) if initialize else None
        if release:
            state=torch.load(release['weights_path'],map_location='cpu',weights_only=True)
            backbone.load_state_dict(convert_dino_state(state,backbone,format=release['format']),strict=True)
    else:
        backbone=ImageNetTiny()
        if initialize:
            path=Path(protocol['repo_root'])/'data/pretrained/checkpoints/convnext_tiny-983f1562.pth'
            if sha256_file(path)!=IMAGENET_SHA: raise ValueError('Generic ImageNet Tiny weights hash changed')
            sidecar=json.loads(path.with_suffix(path.suffix+'.json').read_text())
            if sidecar.get('sha256')!=IMAGENET_SHA or sidecar.get('scope')!='generic_pretraining_no_external_PAR':
                raise ValueError('Generic ImageNet Tiny provenance missing')
            state=torch.load(path,map_location='cpu',weights_only=True)
            mapped={k:v for k,v in state.items() if k.startswith('features.')}
            mapped.update({k.replace('classifier.0.','norm.',1):v for k,v in state.items() if k.startswith('classifier.0.')})
            backbone.load_state_dict(mapped,strict=True)
    return NextModel(backbone,dropout=.1 if arm=='ema_aug' else 0.)


def run_d1(protocol):
    """Report current CPU preparation/access status; never manufacture parity."""
    from .protocol import verify_protocol
    verify_protocol(Path(protocol['artifact_root'])/'reports/protocol.json')
    try:
        release=require_dino_release(protocol)
    except RuntimeError as error:
        return {'status':'blocked','protocol_sha256':protocol['protocol_sha256'],'reason':str(error),
                'weights_available':False,'gpu_job_started':False,'reference_parity':'unmeasured'}
    return {'status':'completed','protocol_sha256':protocol['protocol_sha256'],
            'release_sha256':sha256_file(Path(protocol['artifact_root'])/'reports/dino_model_release.json'),
            'weights_sha256':release['weights_sha256'],'gpu_job_started':False}


def measure_dino_reference(protocol,weights_path,format,reference_source_path,expected_reference_sha256,*,upstream_weights_path=None):
    """Measure real weights against an independently retained official module.

    Caller pins the downloaded source hash; this never downloads code or
    weights. The official source is executed only after hash verification.
    A random-weight architecture check is not a pretrained compatibility claim.
    """
    source=Path(reference_source_path)
    if sha256_file(source)!=expected_reference_sha256:
        raise ValueError('Independent official reference source hash changed')
    path=Path(weights_path)
    if not path.resolve().is_relative_to(Path(protocol['artifact_root']).resolve()):
        raise ValueError('New DINO weights must remain in the new artifact root')
    proof=prove_dino_origin(protocol,path,format,upstream_weights_path)
    spec=importlib.util.spec_from_file_location('_upar_dino_official_reference',source)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    state=torch.load(path,map_location='cpu',weights_only=True)
    portable=DinoConvNeXtTiny().eval(); native=convert_dino_state(state,portable,format=format)
    portable.load_state_dict(native,strict=True)
    reference=module.ConvNeXt(depths=[3,3,9,3],dims=[96,192,384,768],drop_path_rate=0.).eval()
    original=dict(native)
    original['norms.3.weight']=original['norm.weight']; original['norms.3.bias']=original['norm.bias']
    reference.load_state_dict(original,strict=True)
    generator=torch.Generator().manual_seed(42); differences=[]
    with torch.inference_mode():
        for batch in (1,2,3):
            x=torch.randn(batch,3,32,64,generator=generator)
            actual=portable(x); expected=reference(x)
            if actual.shape!=(batch,768) or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
                raise ValueError('Actual DINO reference output is not finite/aligned')
            differences.append(float((actual-expected).abs().max()))
    if max(differences)>1e-5: raise RuntimeError('Actual DINO independent CPU reference parity failed')
    return {'status':'passed','protocol_sha256':protocol['protocol_sha256'],'model_id':MODEL_ID,'revision':REVISION,
            'weights_sha256':sha256_file(path),'torch_version':torch.__version__,
            'reference_source_sha256':expected_reference_sha256,
            'reference_source_url':'https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/convnext.py',
            'portable_code_sha256':sha256_file(__file__),'batch_sizes':[1,2,3],'image_size':[32,64],
            'feature_shape':[768],'maximum_absolute_difference':max(differences),'tolerance':1e-5,
            'offline':True,'actual_pretrained_weights':True,'origin':proof,
            'source_revision_limitation':'Independent retained source hash is pinned; gated weight revision does not imply source revision identity.'}


def publish_dino_release(protocol,*,weights_path,format,reference_report_path,portable_report_path,
                         access_granted,license_acknowledged_by_user,upstream_weights_path=None):
    """Exclusive supplemental model release after real access and both CPU checks."""
    if access_granted is not True or license_acknowledged_by_user is not True:
        raise RuntimeError('Explicit user access and license acknowledgement are required')
    if format not in ('native','huggingface'): raise ValueError('Unknown DINO release checkpoint format')
    root=Path(protocol['artifact_root']); weights=Path(weights_path); digest=sha256_file(weights)
    proof=prove_dino_origin(protocol,weights,format,upstream_weights_path)
    record={'origin':proof,'status':'completed','protocol_sha256':protocol['protocol_sha256'],'model_id':MODEL_ID,'revision':REVISION,
            'weights_path':str(weights),'weights_sha256':digest,'format':format,
            'scope':'generic_pretraining_no_external_PAR','access_granted':True,'license_acknowledged_by_user':True,
            'license':'DINOv3 License Agreement','portable_code_sha256':sha256_file(__file__)}
    if not weights.resolve().is_relative_to(root.resolve()): raise ValueError('DINO release weights must remain in new artifact root')
    for name,path in [('reference_parity',Path(reference_report_path)),('portable_compatibility',Path(portable_report_path))]:
        if not path.resolve().is_relative_to(root.resolve()): raise ValueError('CPU reports must remain in new artifact root')
        detail=json.loads(path.read_text())
        if any(detail.get(k)!=v for k,v in {'status':'passed','protocol_sha256':protocol['protocol_sha256'],
            'model_id':MODEL_ID,'revision':REVISION,'weights_sha256':digest,'offline':True,
            'actual_pretrained_weights':True,'origin':proof,'portable_code_sha256':sha256_file(__file__)}.items()):
            raise RuntimeError('Actual hash-bound DINO CPU evidence prerequisites missing')
        if name=='portable_compatibility' and not str(detail.get('torch_version','')).startswith('2.4.1'):
            raise RuntimeError('Offline organizer Torch2.4.1 compatibility not measured')
        record[name]={'status':'passed','weights_sha256':digest,'report_path':str(path),'report_sha256':sha256_file(path)}
    write_once_json(root/'reports/dino_model_release.json',record)
    return require_dino_release(protocol)
