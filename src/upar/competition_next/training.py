"""Registered user-run 24-epoch arms; selection reads DEV exclusively."""
from __future__ import annotations
import copy
import io
import json
import math
import os
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader
from upar.io import sha256_file,hash_object
from upar.data import load_split,ATTRIBUTE_NAMES
from upar.jev import validate_response
from upar.runtime import preprocess
from .models import build_model
from .artifacts import write_once_json
from .resources import campaign_job
from .orchestration import jev_payload_sha256


class EMA:
    def __init__(self,model,decay=.999):
        if not 0<=decay<1: raise ValueError('Invalid EMA decay')
        self.model=copy.deepcopy(model).eval(); self.decay=decay; self.updates=0
        for parameter in self.model.parameters(): parameter.requires_grad_(False)
    @torch.no_grad()
    def update(self,model):
        source=model.state_dict()
        for name,value in self.model.state_dict().items():
            if value.is_floating_point(): value.mul_(self.decay).add_(source[name],alpha=1-self.decay)
            else: value.copy_(source[name])
        self.updates+=1


def fit_positive_weights(labels):
    y=np.asarray(labels)
    if y.ndim!=2 or y.shape[1]!=40 or not len(y) or np.any((y!=0)&(y!=1)):
        raise ValueError('Weighted BCE requires finite binary FIT labels')
    counts=y.sum(axis=0)
    return torch.from_numpy(np.clip((len(y)-counts)/np.maximum(counts,1),1,10).astype(np.float32))


def train_epoch(model,loader,optimizer,*,device,accumulation=2,ema=None,scaler=None,
                smoothing=0.,positive_weights=None,check=lambda:None,maximum_steps=None):
    """Sample-weighted accumulation including partial final groups."""
    if type(accumulation) is not int or accumulation<1 or not 0<=smoothing<1:
        raise ValueError('Invalid accumulation/smoothing')
    model.train(); pending=[]; total=0.; seen=0; steps=0
    def flush(group):
        nonlocal total,seen,steps
        check(); optimizer.zero_grad(set_to_none=True); count=sum(len(pair[0]) for pair in group)
        for images,targets in group:
            images,targets=images.to(device),targets.to(device).float()
            targets=targets*(1-smoothing)+.5*smoothing
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                output=model(images)
                loss=nn.functional.binary_cross_entropy_with_logits(output['logits'],targets,
                         pos_weight=positive_weights)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            weighted=loss*(len(images)/count)
            (scaler.scale(weighted) if scaler else weighted).backward()
            total+=float(loss.detach())*len(images); seen+=len(images)
        if scaler:
            scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
            updated=scaler.get_scale()>=scale
        else: optimizer.step(); updated=True
        if updated:
            if ema: ema.update(model)
            steps+=1
        check()
    for batch in loader:
        pending.append(batch[:2])
        if len(pending)==accumulation:
            flush(pending); pending=[]
            if maximum_steps is not None and steps>=maximum_steps: break
    if pending and (maximum_steps is None or steps<maximum_steps): flush(pending)
    if not seen: raise ValueError('Empty training loader')
    if maximum_steps is not None and steps!=maximum_steps: raise RuntimeError('Required optimizer steps were not completed')
    return {'loss':total/seen,'images':seen,'optimizer_steps':steps}


def selection_metric(role,track,epoch,metrics):
    if role!='dev' or track not in (1,2) or type(epoch) is not int or not 1<=epoch<=24:
        raise ValueError('Checkpoint selection requires frozen DEV epoch/track')
    if track==2 and epoch%4: raise ValueError('T2 selection only every four epochs')
    value=metrics['Challenge_Avg' if track==1 else 'mADM']
    if isinstance(value,bool) or not math.isfinite(float(value)): raise ValueError('Nonfinite selection metric')
    return float(value)


def _read_bound(protocol,name):
    path=Path(protocol['artifact_root'])/'reports'/name
    if not path.is_file(): raise RuntimeError(f'Missing measured prerequisite {name}')
    value=json.loads(path.read_text())
    if value.get('protocol_sha256')!=protocol['protocol_sha256']: raise ValueError('Prerequisite protocol mismatch')
    return value,path


def require_training_admission(protocol,arm):
    if arm not in ('dino_a','dino_b','ema_aug'): raise ValueError('Unknown training arm')
    from .runtime import require_measured_preflight
    preflight=require_measured_preflight(protocol,arm)
    path=Path(protocol['artifact_root'])/'reports'/(arm+'_preflight.json')
    review,_=_read_bound(protocol,arm+'_preflight_review.json')
    raw_path=Path(protocol['artifact_root'])/'reports'/(arm+'_preflight_jev.json')
    if not raw_path.is_file(): raise RuntimeError('Fresh raw Jev evidence missing')
    raw=json.loads(raw_path.read_text())
    if raw.get('status')!='completed': raise RuntimeError('Jev review pending/failed')
    validate_response(raw['request'],raw['response'])
    state=raw['request']['state']
    if not isinstance(state,dict) or any(state.get(k)!=v for k,v in {'protocol_sha256':protocol['protocol_sha256'],'arm':arm,'evidence_sha256':sha256_file(path)}.items()):
        raise ValueError('Jev request does not bind measured arm evidence')
    expected={'status':'completed','protocol_sha256':protocol['protocol_sha256'],'arm':arm,
              'evidence_sha256':sha256_file(path),'admitted':True,'model':raw['response']['model'],
              'request_sha256':jev_payload_sha256(raw['request']),'response_sha256':jev_payload_sha256(raw['response'])}
    if any(review.get(k)!=v for k,v in expected.items()) or raw.get('request_sha256')!=jev_payload_sha256(raw['request']):
        raise RuntimeError('Fresh hash-bound Jev training admission missing')
    answer=raw['response']['answers'].get('train_admission',{})
    if answer.get('type')!='choice' or answer.get('choice')!='train': raise RuntimeError('Jev advisory did not support this training handoff')
    if arm=='dino_b':
        from .runtime import require_dino_a_admission
        require_dino_a_admission(protocol)
    return preflight


class NextDataset(Dataset):
    def __init__(self,split,root,indices,*,arm,augment=False,image_size=(256,128)):
        self.split,self.root,self.indices=split,Path(root),list(indices)
        self.arm,self.augment,self.image_size=arm,augment,tuple(image_size)
    def __len__(self): return len(self.indices)
    def __getitem__(self,index):
        i=self.indices[index]
        path=(self.root/self.split.paths[i]).resolve()
        if not path.is_relative_to(self.root.resolve()): raise ValueError('Unsafe frozen image path')
        with Image.open(path) as source: image=source.convert('RGB')
        if self.augment:
            if random.random()<.5: image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image=ImageEnhance.Brightness(image).enhance(random.uniform(.9,1.1))
            image=ImageEnhance.Contrast(image).enhance(random.uniform(.9,1.1))
            if self.arm=='ema_aug':
                # One fixed bundled augmentation distribution, no sweep.
                if random.random()<.25:
                    factor=random.uniform(.6,1.); size=image.size
                    image=image.resize((max(1,int(size[0]*factor)),max(1,int(size[1]*factor))),Image.Resampling.BILINEAR).resize(size,Image.Resampling.BILINEAR)
                if random.random()<.2: image=image.filter(ImageFilter.GaussianBlur(random.uniform(.1,1.)))
                if random.random()<.2:
                    stream=io.BytesIO(); image.save(stream,format='JPEG',quality=random.randint(70,95)); stream.seek(0)
                    with Image.open(stream) as compressed: image=compressed.convert('RGB').copy()
                width,height=image.size; scale=random.uniform(.95,1.05)
                tx=random.uniform(-.03,.03)*width; ty=random.uniform(-.03,.03)*height
                image=image.transform(image.size,Image.Transform.AFFINE,(1/scale,0,tx,0,1/scale,ty),resample=Image.Resampling.BILINEAR)
        return preprocess(image,self.image_size),torch.tensor(self.split.labels[i],dtype=torch.float32)


def frozen_sources(protocol):
    from .protocol import partition_indices
    split=load_split(protocol['annotations'],'train')
    if split.hashes!=protocol['train_hashes'] or split.paths!=protocol['paths']: raise ValueError('Frozen annotations changed')
    for role in ('fit','dev'): partition_indices(protocol,role)
    return split,Path(protocol['repo_root'])/'data'


def _seed_worker(worker_id):
    seed=torch.initial_seed()%2**32; random.seed(seed); np.random.seed(seed)


def make_loader(dataset,*,train=False,workers=4):
    generator=torch.Generator().manual_seed(42)
    return DataLoader(dataset,batch_size=32,shuffle=train,num_workers=workers,pin_memory=True,
                      worker_init_fn=_seed_worker,generator=generator,drop_last=False)


def optimizer_for(model):
    return torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':1e-4,'base_lr':1e-4},
                             {'params':model.head.parameters(),'lr':1e-3,'base_lr':1e-3}],weight_decay=.05)


@torch.inference_mode()
def predict(model,loader,device,check=lambda:None):
    model.eval(); parts=[]
    for batch in loader:
        check(); values=model(batch[0].to(device))['probabilities'].float().cpu().numpy()
        if not np.isfinite(values).all(): raise ValueError('Nonfinite inference')
        parts.append(values)
    if not parts: raise ValueError('No inference images')
    return np.concatenate(parts).astype(np.float32)


def atomic_torch(path,payload,*,immutable=True):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent); os.close(fd)
    try:
        torch.save(payload,name)
        if immutable: os.link(name,path)
        else: os.replace(name,path)
    finally: Path(name).unlink(missing_ok=True)


def train_arm(protocol,arm):
    from .protocol import verify_protocol,partition_indices
    from .runtime import l1_distances,require_gpu_preflight
    from .evaluation import evaluate_track1,evaluate_track2
    from .features import save_prediction_cache
    verify_protocol(Path(protocol['artifact_root'])/'reports/protocol.json')
    readiness=require_training_admission(protocol,arm); require_gpu_preflight(protocol,arm)
    root=Path(protocol['artifact_root']); device=torch.device('cuda')
    with campaign_job(protocol,arm+'_train','train_hours',projected_hours=readiness['projected_arm_hours'],maximum_hours=4.,gpu=True) as check:
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        split,image_root=frozen_sources(protocol); fit=partition_indices(protocol,'fit'); dev=partition_indices(protocol,'dev')
        model=build_model(protocol,arm).to(device); ema=EMA(model) if arm=='ema_aug' else None
        optimizer=optimizer_for(model); scaler=torch.amp.GradScaler('cuda')
        fit_loader=make_loader(NextDataset(split,image_root,fit,arm=arm,augment=True),train=True)
        dev_loader=make_loader(NextDataset(split,image_root,dev,arm=arm))
        positive=None
        if arm=='dino_b':
            positive=fit_positive_weights(split.labels[fit]).to(device)
        selected={}; history=[]; best={1:-math.inf,2:-math.inf}; folder=root/'experiments'/arm; folder.mkdir(parents=True,exist_ok=True)
        for epoch in range(1,25):
            check(); factor=epoch/2 if epoch<=2 else .5*(1+math.cos(math.pi*(epoch-2)/22))
            for group in optimizer.param_groups: group['lr']=group['base_lr']*factor
            train=train_epoch(model,fit_loader,optimizer,device=device,scaler=scaler,ema=ema,
                              smoothing=.02 if ema else 0.,positive_weights=positive,check=check)
            chosen=ema.model if ema else model; p=predict(chosen,dev_loader,device,check)
            metrics={1:evaluate_track1(split.labels[dev],p,split.domains[dev])}
            clipped=np.clip(p.astype(np.float64),1e-6,1-1e-6)
            nll=float(-(split.labels[dev]*np.log(clipped)+(1-split.labels[dev])*np.log1p(-clipped)).mean())
            if epoch%4==0:
                metrics[2]=evaluate_track2(split.labels[dev],p,split.domains[dev],scorer=l1_distances,
                                           fit_counts=protocol['fit_query_counts'],checkpoint=check)
            raw_metrics=None
            if ema:
                raw_p=predict(model,dev_loader,device,check)
                raw_metrics={'track1':evaluate_track1(split.labels[dev],raw_p,split.domains[dev])}
                if epoch%4==0: raw_metrics['track2']=evaluate_track2(split.labels[dev],raw_p,split.domains[dev],scorer=l1_distances,checkpoint=check)
            for track,metric in metrics.items():
                score=selection_metric('dev',track,epoch,metric)
                if score>best[track]:
                    checkpoint=folder/f'best_track{track}_e{epoch}.pt'
                    payload={'model':chosen.state_dict(),'arm':arm,'protocol_sha256':protocol['protocol_sha256'],
                             'attribute_names':ATTRIBUTE_NAMES,'image_size':[256,128],'epoch':epoch,'track':track,'selection_split':'dev'}
                    atomic_torch(checkpoint,payload); digest=sha256_file(checkpoint)
                    cache=root/'caches'/f'{arm}_track{track}_e{epoch}_dev.npz'
                    save_prediction_cache(cache,[split.paths[i] for i in dev],p,{'protocol_sha256':protocol['protocol_sha256'],
                        'checkpoint_sha256':digest,'arm':arm,'split':'DEV','selection_split':'DEV','epoch':epoch,'track':track})
                    previous=selected.get('best_track'+str(track))
                    selected['best_track'+str(track)]={'path':str(checkpoint.relative_to(root)),'sha256':digest,'epoch':epoch,'score':score,
                              'dev_cache_path':str(cache.relative_to(root)),'dev_cache_sha256':sha256_file(cache)}
                    best[track]=score
                    # Only artifacts made by this running arm are pruned; historical artifacts remain read-only.
                    if previous:
                        (root/previous['path']).unlink(); (root/previous['dev_cache_path']).unlink()
            history.append({'epoch':epoch,'train':train,'selected_model':'EMA' if ema else 'raw','metrics':metrics,'raw_metrics':raw_metrics,
                            'auxiliary_mean_Bernoulli_NLL':nll,'track2_selection_scorer':'L1'})
            atomic_torch(folder/'resume.pt',{'model':model.state_dict(),'ema':ema.model.state_dict() if ema else None,
                        'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'history':history,
                        'protocol_sha256':protocol['protocol_sha256'],'arm':arm},immutable=False)
            check()
        result={'status':'completed','protocol_sha256':protocol['protocol_sha256'],'arm':arm,'epochs':24,
                'selected':selected,'history':history,'selection_split':'DEV','component_admission':'pending',
                'diversity_admission':'pending','training_seed':42,'selection_reuse':'DEV examined every epoch; historical TUNE exposure retained',
                'recipe':'one bundled EMA/augmentation recipe; no isolated EMA causal claim' if ema else arm}
        write_once_json(root/'reports'/(arm+'_train_result.json'),result); return result
