"""Standalone offline scripted-logit inference, copied into admitted ZIPs."""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image
import torch

COLORS=('Black','Blue','Brown','Green','Grey','Orange','Pink','Purple','Red','White','Yellow','Other')
ATTRIBUTE_NAMES=['Age-Young','Age-Adult','Age-Old','Gender-Female','Hair-Length-Short','Hair-Length-Long',
    'Hair-Length-Bald','UpperBody-Length-Short',*[f'UpperBody-Color-{c}' for c in COLORS],
    'LowerBody-Length-Short',*[f'LowerBody-Color-{c}' for c in COLORS],'LowerBody-Type-Trousers&Shorts',
    'LowerBody-Type-Skirt&Dress','Accessory-Backpack','Accessory-Bag','Accessory-Glasses-Normal',
    'Accessory-Glasses-Sun','Accessory-Hat']


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def config_digest(config):
    value={k:v for k,v in config.items() if k!='config_sha256'}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def validate_assets(root):
    root=Path(root).resolve(); c=json.loads((root/'assets/inference.json').read_text())
    if c.get('schema_version')!=1 or type(c.get('track')) is not int or c['track'] not in (1,2):
        raise ValueError('Unknown upgrade runtime schema/track')
    names=c.get('attribute_names',[])
    if names!=ATTRIBUTE_NAMES:
        raise ValueError('Invalid attribute vocabulary')
    if c.get('batch_size')!=8 or not isinstance(c.get('query_block_size'),int) or c['query_block_size']<1:
        raise ValueError('Invalid inference batch/query block')
    if len(c.get('image_size',[]))!=2 or any(type(n) is not int or n<1 for n in c['image_size']):
        raise ValueError('Invalid image shape')
    _calibration(c['track'],c['bias'],c['temperature'],c['new_weight'])
    if c.get('config_sha256')!=config_digest(c): raise ValueError('Configuration integrity failure')
    hashes=c.get('assets_sha256',{})
    if c.get('model') not in hashes: raise ValueError('Missing scripted model integrity hash')
    for name,digest in hashes.items():
        p=Path(name); path=root/p
        if (p.is_absolute() or '..' in p.parts or not path.resolve().is_relative_to(root)
                or not path.is_file() or file_sha256(path)!=digest):
            raise ValueError(f'Packaged asset integrity failure: {name}')
    if c['new_weight']<1:
        incumbent=c.get('incumbent')
        if not isinstance(incumbent,str) or incumbent not in hashes: raise ValueError('Missing incumbent configuration')
        old=json.loads((root/incumbent).read_text())
        for component in old['components']:
            name=str(Path(incumbent).parent/component['checkpoint'])
            if hashes.get(name)!=component['sha256']: raise ValueError('Incumbent asset integrity mismatch')
    elif c.get('incumbent') is not None: raise ValueError('New-only recipe must omit incumbent assets')
    return c


def _calibration(track,bias,temperature,new_weight):
    if (track not in (1,2) or isinstance(bias,bool) or bias not in (0.,-.2,.2,-.4,.4)
            or isinstance(temperature,bool) or not np.isfinite(temperature) or not .5<=temperature<=3
            or isinstance(new_weight,bool) or not np.isfinite(new_weight) or not 0<new_weight<=1):
        raise ValueError('Calibration outside registered recipe')
    if track==1 and (temperature!=1 or new_weight!=1): raise ValueError('Invalid Track1 calibration')
    if track==2 and bias!=0: raise ValueError('Track2 bias must be zero')


def combine_probabilities(logits,*,track,bias,temperature,new_weight,incumbent=None):
    _calibration(track,bias,temperature,new_weight)
    x=np.asarray(logits,dtype=np.float64)
    if x.ndim!=2 or not all(x.shape) or x.shape[1]!=40 or not np.isfinite(x).all():
        raise ValueError('Finite nonempty 40-logit matrix required')
    z=(x+bias)/temperature
    # Stable scipy.expit-equivalent computation, preserving float64 calibration.
    p=np.empty_like(z); positive=z>=0
    p[positive]=1/(1+np.exp(-z[positive]))
    ez=np.exp(z[~positive]); p[~positive]=ez/(1+ez)
    if new_weight<1:
        old=np.asarray(incumbent,dtype=np.float64)
        if old.shape!=p.shape or not np.isfinite(old).all() or np.any((old<0)|(old>1)):
            raise ValueError('Aligned finite incumbent effective probabilities required')
        p=new_weight*p+(1-new_weight)*old
    return p


def preprocess(image,image_size):
    height,width=image_size
    pixels=np.asarray(image.convert('RGB').resize((width,height),Image.Resampling.BILINEAR),dtype=np.float32)/255.
    pixels=(pixels-np.asarray([.485,.456,.406],np.float32))/np.asarray([.229,.224,.225],np.float32)
    return torch.from_numpy(pixels.transpose(2,0,1).copy())


class Predictor:
    def __init__(self,root,*,device=None):
        self.root=Path(root); self.config=validate_assets(self.root)
        self.names=self.config['attribute_names']; self.image_size=self.config['image_size']
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        self.model=torch.jit.load(str(self.root/self.config['model']),map_location=self.device).eval()
        self.incumbent=None
        if self.config['new_weight']<1:
            # Relative import resolves only the bundled unchanged legacy sources.
            from .candidate_runtime import CandidatePredictor
            self.incumbent=CandidatePredictor(self.root/self.config['incumbent'],device=str(self.device))
            if self.incumbent.names!=self.names: raise ValueError('Incumbent attribute order mismatch')

    @torch.inference_mode()
    def predict_samples(self,samples,*,batch_size=None,check=lambda:None):
        batch_size=self.config['batch_size'] if batch_size is None else batch_size
        if type(batch_size) is not int or batch_size<1 or not len(samples): raise ValueError('Nonempty samples and positive batch size required')
        parts=[]
        for start in range(0,len(samples),batch_size):
            check(); batch=samples[start:start+batch_size]; tensors=[]; orders=[]
            for sample in batch:
                names=sample['attribute_names']
                if len(names)!=40 or len(set(names))!=40 or set(names)!=set(self.names): raise ValueError('Invalid attribute vocabulary')
                orders.append([self.names.index(name) for name in names])
                with Image.open(sample['image_path']) as image: tensors.append(preprocess(image,self.image_size))
            torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
            with torch.autocast(device_type=self.device.type,enabled=False):
                logits=self.model(torch.stack(tensors).to(self.device).float()).float().cpu().numpy()
            logits=np.take_along_axis(logits,np.asarray(orders),axis=1)
            old=None if self.incumbent is None else self.incumbent.predict_samples(batch,batch_size=batch_size)['probabilities']
            parts.append(combine_probabilities(logits,track=self.config['track'],bias=self.config['bias'],
                temperature=self.config['temperature'],new_weight=self.config['new_weight'],incumbent=old))
        return {'probabilities':np.concatenate(parts)}


def blocked_nll(queries,probabilities,*,query_block=32,gallery_block=4096,check=lambda:None):
    q=np.asarray(queries); p=np.asarray(probabilities)
    if (q.ndim!=2 or p.ndim!=2 or not all(q.shape) or not all(p.shape) or q.shape[1]!=40 or p.shape[1]!=40
            or not np.issubdtype(q.dtype,np.number) and q.dtype!=np.bool_): raise ValueError('Expected 40-attribute query/gallery matrices')
    if not np.isfinite(q).all() or not np.isin(q,[0,1]).all() or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Binary queries and finite probabilities required')
    if any(type(n) is not int or n<1 for n in (query_block,gallery_block)): raise ValueError('Invalid block size')
    out=np.empty((len(q),len(p)),np.float32)
    for g0 in range(0,len(p),gallery_block):
        check(); safe=np.clip(p[g0:g0+gallery_block].astype(np.float64),1e-6,1-1e-6)
        c0=-np.log1p(-safe); slope=-np.log(safe)-c0
        for q0 in range(0,len(q),query_block):
            check(); qb=q[q0:q0+query_block].astype(np.float64)
            out[q0:q0+len(qb),g0:g0+len(safe)]=c0.sum(1)[None]+qb@slope.T
    if not np.isfinite(out).all(): raise ValueError('Nonfinite retrieval distances')
    return out
