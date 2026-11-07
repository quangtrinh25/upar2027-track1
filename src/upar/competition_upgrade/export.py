"""Admitted offline Torch 2.4.1 export and user-run new-package benchmark."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import warnings
from importlib.metadata import version

import numpy as np
import torch
from torch import nn
from PIL import Image

from upar.candidate2.storage import publish, read_json
from upar.io import hash_object, sha256_file
from .runtime import config_digest, preprocess

CPU_LOGIT_TOLERANCE=1e-5
PUBLIC_GPU_PROBABILITY_TOLERANCE=2e-5
OLD_FILES=('runtime.py','model.py','scoring.py','calibration.py','candidate_runtime.py')

ENTRYPOINT='''from pathlib import Path
from upar.upgrade_runtime import Predictor, blocked_nll
HERE=Path(__file__).resolve().parent
_MODEL=None
_CHECK=lambda: None
def load_model():
    global _MODEL
    _MODEL=Predictor(HERE)
def predict_batch(samples):
    if _MODEL is None: load_model()
    return _MODEL.predict_samples(samples,check=_CHECK)['probabilities'].tolist()
def predict_image(sample):
    return predict_batch([sample])[0]
def rank_gallery(sample):
    if _MODEL is None: load_model()
    if _MODEL.config['track']!=2: raise ValueError('This package is for Track1')
    gallery=[dict(g,attribute_names=sample['attribute_names']) for g in sample['gallery']]
    p=_MODEL.predict_samples(gallery,check=_CHECK)['probabilities']
    return {'distances':blocked_nll(sample['queries'],p,query_block=_MODEL.config['query_block_size'],check=_CHECK)}
'''


def require_export_torch(version=None):
    if str(torch.__version__ if version is None else version).split('+')[0]!='2.4.1':
        raise RuntimeError('Export requires exact Torch 2.4.1 CPU environment')


class Logits(nn.Module):
    def __init__(self,model):
        super().__init__(); self.model=model
    def forward(self,x):
        return self.model(x)['logits']


def synthetic_tensors(image_size,batches=3):
    height,width=image_size
    fixtures=[]
    for i in range(batches):
        grid=np.arange(height*width*3,dtype=np.uint32).reshape(height,width,3)
        pixels=((grid*17+i*71)%256).astype(np.uint8)
        fixtures.append(preprocess(Image.fromarray(pixels),image_size))
    return torch.stack(fixtures)


def trace_logits(model,path,image_size,*,check=lambda:None):
    require_export_torch(); model=model.cpu().float().eval(); x=synthetic_tensors(image_size)
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    started=time.monotonic(); check()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always',torch.jit.TracerWarning)
        with torch.inference_mode():
            traced=torch.jit.freeze(torch.jit.trace(model,x[:1],check_trace=False))
            torch.jit.save(traced,str(path))
    recorded_warnings=[{'category':category,'message':message} for category,message in
                       dict.fromkeys((warning.category.__name__,str(warning.message)) for warning in captured)]
    exported=time.monotonic(); check()
    scripted=torch.jit.load(str(path),map_location='cpu').eval(); loaded=time.monotonic()
    maximum=0.; batches=[]
    with torch.inference_mode():
        for n in (1,2,3):
            check(); eager=model(x[:n]).float().numpy(); actual=scripted(x[:n]).float().numpy()
            if eager.shape!=(n,40) or actual.shape!=(n,40) or not np.isfinite(eager).all() or not np.isfinite(actual).all():
                raise ValueError('Invalid eager/scripted export logits')
            delta=float(np.max(np.abs(actual-eager))); maximum=max(maximum,delta); batches.append(n)
            if delta>CPU_LOGIT_TOLERANCE: raise ValueError('Same-device eager/scripted CPU parity failed')
    ended=time.monotonic()
    return {'passed':True,'same_device':'cpu','batch_sizes':batches,'maximum_absolute_logit_difference':maximum,
            'absolute_tolerance':CPU_LOGIT_TOLERANCE,'export_seconds':exported-started,'load_seconds':loaded-exported,
            'parity_seconds':ended-loaded,'elapsed_seconds':ended-started,'scripted_bytes':path.stat().st_size,
            'scripted_sha256':sha256_file(path),'trace_warnings':recorded_warnings,
            'fixture':'deterministic RGB modular grid; ImageNet normalization'}


def cpu_export_smoke(config_path,output_path):
    """Preparation only: untrained PAR heads, local generic A/B, synthetic CPU."""
    from .protocol import configuration
    from .models import build_model
    require_export_torch(); cfg=configuration(config_path)
    output_path=Path(output_path)
    if output_path.exists(): raise FileExistsError(output_path)
    torch.set_num_threads(cfg['runtime']['cpu_threads'])
    arms={}; pretrained={}
    with tempfile.TemporaryDirectory(prefix='upgrade-export-smoke-',dir='/tmp') as scratch:
        for arm in ('A','B'):
            start=time.monotonic()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42); model=build_model(arm,cfg,initialize=True).cpu().eval()
            path=Path(scratch)/(arm+'.pt')
            evidence=trace_logits(Logits(model),path,cfg['models'][arm]['image_size'])
            io_start=time.monotonic()
            # Measure real local asset read/write traffic without retaining extra weights.
            copy=Path(scratch)/(arm+'.io'); shutil.copyfile(path,copy)
            io_hash=sha256_file(copy); io_end=time.monotonic()
            if io_hash!=evidence['scripted_sha256']: raise ValueError('Smoke asset I/O integrity failure')
            pretrained[arm]=sha256_file(cfg['models'][arm]['checkpoint'])
            arms[arm]={**evidence,'parity_passed':True,'tested_batches':evidence['batch_sizes'],
                       'asset_io_seconds':io_end-io_start,'asset_io_bytes':path.stat().st_size,
                       'elapsed_seconds':time.monotonic()-start,'quality_scope':'Untrained PAR head; no training or quality measurement'}
            del model; path.unlink(); copy.unlink()
    code={'src/upar/competition_upgrade/'+name:sha256_file(Path(__file__).with_name(name)) for name in ('models.py','runtime.py','export.py')}
    environment={name:version(name) for name in ('torch','torchvision','timm','numpy','Pillow')}
    environment['python']=sys.version.split()[0]
    result={'status':'completed','torch':str(torch.__version__),'device':'cpu','environment':environment,'config_sha256':hash_object(cfg),
            'code_sha256':code,'pretrained_sha256':pretrained,'arms':arms,
            'scope':'Measured generic-architecture synthetic CPU preparation; not final selected-checkpoint package parity'}
    publish(output_path,result)
    return result


def verify_admission(track,selected,public,internal,selection_sha):
    if type(track) is not int or track not in (1,2): raise ValueError('Track must be 1 or 2')
    key='track'+str(track); candidate=selected.get('selected',{}).get(key)
    valid=public.get('result',{}).get(key,{})
    if (not selected.get('frozen_before_public') or candidate is None or not valid.get('admitted')
            or candidate!=valid.get('candidate') or public.get('selection_sha256')!=selection_sha):
        raise ValueError('Exact frozen admitted public candidate required')
    arm=candidate.get('arm')
    screened=internal.get('candidates',{}).get(arm,{}).get(key,{})
    if (not screened.get('qualified') or screened.get('candidate')!=candidate
            or not selected.get('ood',{}).get(arm,{}).get(key,{}).get('eligible')):
        raise ValueError('Internal and held-source admission required')
    from .evaluation import public_admission,ood_admission,PRIMARY
    try:
        admission=public_admission(track,valid['metrics'],valid['reference'])
        if not admission['admitted'] or not np.isfinite(admission['primary_delta']):
            raise ValueError('Public numerical admission failed')
        for field in ('primary_delta','domain_deltas','minimum_gain'):
            if field in valid and valid[field]!=admission[field]: raise ValueError('Public admission receipt changed')
    except (KeyError,TypeError) as exc: raise ValueError('Public metric evidence missing') from exc
    try:
        ood=selected['ood'][arm][key]
        if ood!=ood_admission(track,ood['deltas']): raise ValueError('Held-source numerical admission changed')
        expected_recipe={name:candidate[name] for name in ('head','bias','temperature','new_weight')}
        if screened['full']['recipe']!=expected_recipe: raise ValueError('Internal full recipe differs from selected candidate')
        for measurement in (*screened['directions'].values(),screened['full']):
            delta=measurement['metrics'][PRIMARY[track]]-measurement['reference'][PRIMARY[track]]
            if not np.isfinite(delta) or delta<=0 or not np.isclose(delta,measurement['delta_vs_incumbent'],rtol=0,atol=1e-12):
                raise ValueError('Internal numerical incumbent admission failed')
        if set(screened['directions'])!={'a_to_b','b_to_a'} or any(
                not d['positive_new_weight'] or not 0<d['recipe']['new_weight']<=1 for d in screened['directions'].values()):
            raise ValueError('Both internal directions require positive new weight')
    except (KeyError,TypeError) as exc: raise ValueError('Internal or held-source numerical evidence missing') from exc
    from .runtime import _calibration
    _calibration(track,candidate['bias'],candidate['temperature'],candidate['new_weight'])
    if candidate['track']!=track or candidate['head'] not in ('original','balanced') or track==2 and candidate['head']!='original':
        raise ValueError('Invalid selected head/track')
    return candidate


def _context(config_path,track):
    from .protocol import load_manifest,roots,require_record
    from .features import load_arrays,checked_selection
    manifest=load_manifest(config_path); run,reports=roots(manifest)
    selection=require_record(manifest,'selection'); public=require_record(manifest,'public_validation')
    internal=require_record(manifest,'internal_selection'); selection_sha=sha256_file(reports/'selection.json')
    candidate=verify_admission(track,selection,public,internal,selection_sha)
    features=require_record(manifest,'features'); job=features['jobs']['train_'+candidate['arm']]
    training=require_record(manifest,'train_'+candidate['arm'])
    chosen=checked_selection(manifest,training,track,candidate['arm'],None)
    if (job['track'+str(track)]['selection']!=chosen or chosen['checkpoint']!=candidate['checkpoint']
            or chosen['sha256']!=candidate['checkpoint_sha256']): raise ValueError('Selected checkpoint changed')
    if sha256_file(candidate['checkpoint'])!=candidate['checkpoint_sha256']: raise ValueError('Checkpoint hash changed')
    if candidate['head']=='balanced':
        entry=job['track1']
        if candidate['head_path']!=entry['head_path'] or candidate['head_sha256']!=entry['head_sha256']:
            raise ValueError('Balanced selected head changed')
        if sha256_file(candidate['head_path'])!=candidate['head_sha256']: raise ValueError('Balanced head hash changed')
    validated=public['result']['track'+str(track)]; meta=validated['cache_metadata']
    expected={'protocol_sha256':manifest['protocol_sha256'],'selection_sha256':selection_sha,
              'candidate_sha256':hash_object(candidate),'attribute_names':manifest['attribute_names'],'dtype':'float64'}
    if any(meta.get(k)!=v for k,v in expected.items()): raise ValueError('Public cache candidate/protocol mismatch')
    cache=load_arrays(validated['cache'],meta)
    if (cache['probabilities'].dtype!=np.float64 or cache['probabilities'].shape!=(len(cache['paths']),40)
            or np.any((cache['probabilities']<0)|(cache['probabilities']>1))
            or meta.get('paths_sha256')!=hash_object(cache['paths'].tolist())
            or meta.get('domains_sha256')!=hash_object(cache['domains'].tolist())
            or meta.get('labels_sha256')!=hash_object(cache['labels'].tolist())): raise ValueError('Invalid effective public cache')
    artifacts={str(reports/(name+'.json')):sha256_file(reports/(name+'.json'))
               for name in ('selection','public_validation','internal_selection','features','train_'+candidate['arm'])}
    artifacts.update(public['artifacts']); artifacts.update(selection['artifacts'])
    return manifest,candidate,validated,cache,artifacts


def export(config_path,track):
    from .protocol import roots,require_record,stage_budget,check_campaign_disk
    from .training import load_checkpoint
    from .features import load_arrays,apply_head
    require_export_torch(); manifest,candidate,validated,cache,artifacts=_context(config_path,track)
    run,reports=roots(manifest); name='export_track'+str(track)
    if (reports/(name+'.json')).exists(): return require_record(manifest,name)
    torch.set_num_threads(manifest['config']['runtime']['cpu_threads'])
    folder=run/'export'/('track'+str(track)); folder.mkdir(parents=True,exist_ok=True)
    destination=folder/'model.pt'
    if destination.exists(): raise FileExistsError('Unrecorded immutable export already exists')
    with stage_budget(manifest,'export') as clock:
        check_campaign_disk(manifest,2*Path(candidate['checkpoint']).stat().st_size)
        model,payload=load_checkpoint(candidate['checkpoint'],manifest['config'],device='cpu',expected_protocol=manifest['protocol_sha256'])
        if payload['arm']!=candidate['arm'] or payload['heldout'] is not None: raise ValueError('Deployment checkpoint is not selected full-data arm')
        drift={'status':'not_applicable','scope':'Balanced NumPy-vs-Torch FP32 head; separate from export parity'}
        if candidate['head']=='balanced':
            receipt=read_json(candidate['head_path']+'.json'); head=load_arrays(candidate['head_path'],receipt['metadata'])
            if receipt['metadata']['checkpoint_sha256']!=candidate['checkpoint_sha256']: raise ValueError('Head checkpoint provenance changed')
            with torch.inference_mode(): features=model.forward_features(synthetic_tensors(model.image_size)).float().numpy()
            np_logits=apply_head(features,head,check=clock.check)
            with torch.no_grad():
                model.head.weight.copy_(torch.from_numpy(head['weight'])); model.head.bias.copy_(torch.from_numpy(head['bias']))
            with torch.inference_mode(): eager=model.head(torch.from_numpy(features)).numpy()
            difference=float(np.max(np.abs(np_logits-eager)))
            if difference>CPU_LOGIT_TOLERANCE: raise ValueError('Balanced NumPy/Torch head parity failed')
            drift={**drift,'status':'measured','maximum_absolute_logit_difference':difference,'absolute_tolerance':CPU_LOGIT_TOLERANCE}
        with tempfile.TemporaryDirectory(dir=folder) as scratch:
            staged=Path(scratch)/'model.pt'; parity=trace_logits(Logits(model),staged,model.image_size,check=clock.check)
            clock.check(); os.link(staged,destination)
        artifacts[str(destination)]=sha256_file(destination)
        result={'status':'completed','protocol_sha256':manifest['protocol_sha256'],'track':track,'candidate':candidate,
                'candidate_sha256':hash_object(candidate),'selection_sha256':sha256_file(reports/'selection.json'),
                'torch':str(torch.__version__),'device':'cpu','model':str(destination),'image_size':list(model.image_size),
                'parity':parity,'balanced_head_numpy_torch_drift':drift,
                'historical_public_cuda_cache_drift':{'status':'not_measured','scope':'Distinct device/cache question, never claimed as CPU export parity'},
                'artifacts':artifacts}
        clock.check(); publish(reports/(name+'.json'),result)
    return result


def write_runtime(root,*,track):
    """Copy runtime source, then seal all files and configuration hashes."""
    root=Path(root); (root/'upar').mkdir(exist_ok=True)
    (root/'upar/__init__.py').write_text('')
    shutil.copyfile(Path(__file__).with_name('runtime.py'),root/'upar/upgrade_runtime.py')
    (root/'run.py').write_text(ENTRYPOINT)
    config_path=root/'assets/inference.json'; config=read_json(config_path)
    if config['track']!=track: raise ValueError('Runtime track mismatch')
    config['assets_sha256']={str(p.relative_to(root)):sha256_file(p) for p in sorted(root.rglob('*'))
                             if p.is_file() and p!=config_path}
    config['config_sha256']=config_digest(config)
    config_path.write_text(json.dumps(config,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return config


def archive_bundle(root,destination):
    destination=Path(destination)
    if destination.exists(): raise FileExistsError(destination)
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(Path(root).rglob('*')):
            if not path.is_file(): continue
            info=zipfile.ZipInfo(str(path.relative_to(root)),date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o100644<<16
            with archive.open(info,'w',force_zip64=True) as target,path.open('rb') as source: shutil.copyfileobj(source,target)
    return destination


def package(config_path,track):
    from .protocol import roots,require_record,stage_budget,check_campaign_disk
    from upar.candidate_submission import expected_config
    manifest,candidate,validated,cache,artifacts=_context(config_path,track); run,reports=roots(manifest)
    name='package_track'+str(track)
    if (reports/(name+'.json')).exists(): return require_record(manifest,name)
    exported=require_record(manifest,'export_track'+str(track))
    if (exported['candidate']!=candidate or exported['torch'].split('+')[0]!='2.4.1' or exported['device']!='cpu'
            or not exported['parity']['passed'] or exported['selection_sha256']!=sha256_file(reports/'selection.json')):
        raise ValueError('Invalid admitted Torch2.4.1 export provenance')
    if exported['image_size']!=manifest['config']['models'][candidate['arm']]['image_size']: raise ValueError('Export image size changed')
    folder=run/'packages'; folder.mkdir(parents=True,exist_ok=True); destination=folder/('track'+str(track)+'.zip')
    if destination.exists(): raise FileExistsError('Unrecorded immutable package exists')
    with stage_budget(manifest,'package') as clock:
        with tempfile.TemporaryDirectory(dir=folder) as scratch:
            root=Path(scratch)/'bundle'; (root/'assets').mkdir(parents=True)
            shutil.copyfile(exported['model'],root/'assets/model.pt')
            config={'schema_version':1,'track':track,'attribute_names':manifest['attribute_names'],'image_size':exported['image_size'],
                    'batch_size':8,'query_block_size':manifest['config']['runtime']['query_block_size'],
                    'bias':candidate['bias'],'temperature':candidate['temperature'],'new_weight':candidate['new_weight'],
                    'model':'assets/model.pt','incumbent':None,'protocol_sha256':manifest['protocol_sha256'],
                    'candidate_sha256':hash_object(candidate),'selection_sha256':exported['selection_sha256']}
            if candidate['new_weight']<1:
                reference=manifest['config']['references']['ensemble_selection']
                if sha256_file(reference)!=manifest['inputs'][reference]: raise ValueError('Incumbent selection changed')
                old_candidate=read_json(reference)['result']['track2']['candidate']
                old=expected_config(2,old_candidate,manifest['reference_models'],{'frozen_selection_sha256':sha256_file(reference)})
                legacy=root/'assets/incumbent'; legacy.mkdir(); (root/'upar').mkdir()
                for side,component in zip(('left','right'),old['components']):
                    entry=manifest['reference_models'][old_candidate[side+'_model']]
                    if sha256_file(entry['checkpoint'])!=component['sha256']: raise ValueError('Incumbent checkpoint changed')
                    check_campaign_disk(manifest,2*Path(entry['checkpoint']).stat().st_size)
                    shutil.copyfile(entry['checkpoint'],legacy/component['checkpoint'])
                (legacy/'inference.json').write_text(json.dumps(old,sort_keys=True,allow_nan=False)+'\n')
                for name_old in OLD_FILES: shutil.copyfile(Path(__file__).parent.parent/name_old,root/'upar'/name_old)
                config['incumbent']='assets/incumbent/inference.json'
            metadata=Path('external/UPAR-Challenge-2027/examples')/('task'+str(track))/'sample_code_submission/metadata.yaml'
            shutil.copyfile(metadata,root/'metadata.yaml')
            (root/'assets/inference.json').write_text(json.dumps(config,sort_keys=True,allow_nan=False))
            config=write_runtime(root,track=track)
            from .runtime import validate_assets
            validate_assets(root); clock.check(); check_campaign_disk(manifest,2*sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
            staged=archive_bundle(root,Path(scratch)/'package.zip'); clock.check(); os.link(staged,destination)
        artifacts.update(exported['artifacts']); artifacts[str(reports/('export_track'+str(track)+'.json'))]=sha256_file(reports/('export_track'+str(track)+'.json'))
        artifacts[str(destination)]=sha256_file(destination)
        result={'status':'completed','protocol_sha256':manifest['protocol_sha256'],'track':track,'candidate':candidate,
                'archive':str(destination),'archive_sha256':sha256_file(destination),'inference_config':config,'artifacts':artifacts,
                'remote_status':'Not uploaded or remotely verified'}
        clock.check(); publish(reports/('package_track'+str(track)+'.json'),result)
    return result


def run_bounded(command,*,timeout,cwd=None):
    if not np.isfinite(timeout) or timeout<=0: raise TimeoutError('No remaining benchmark budget')
    environment=dict(os.environ); environment.pop('PYTHONPATH',None)
    try: result=subprocess.run(command,cwd=cwd,env=environment,capture_output=True,text=True,timeout=timeout)
    except subprocess.TimeoutExpired as exc: raise TimeoutError('New package benchmark hard timeout') from exc
    if result.returncode!=0: raise RuntimeError('New package benchmark failed: '+result.stderr[-4000:])
    return result


def verify_runtime_admission(track,metrics,reference):
    from .evaluation import public_admission
    admission=public_admission(track,metrics,reference)
    if not admission['admitted'] or not np.isfinite(admission['primary_delta']):
        raise ValueError('Exported package no longer meets unchanged public numerical admission')
    return admission


WORKER='''import time
started=time.monotonic()
import sys,json,resource,importlib.util
from pathlib import Path
import numpy as np
import torch
root=Path(sys.argv[1]).resolve(); sys.path.insert(0,str(root))
import run, upar
if not Path(upar.__file__).resolve().is_relative_to(root): raise RuntimeError('Host package leakage')
payload=json.loads(Path(sys.argv[2]).read_text()); torch.set_num_threads(payload['cpu_threads'])
spec=importlib.util.spec_from_file_location('benchmark_metrics',payload['metrics_source'])
metricmod=importlib.util.module_from_spec(spec); spec.loader.exec_module(metricmod)
if not torch.cuda.is_available(): raise RuntimeError('USER-RUN runtime benchmark requires CUDA')
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
torch.cuda.reset_peak_memory_stats(); run.load_model(); torch.cuda.synchronize(); loaded=time.monotonic()
deadline=started+payload['timeout']
def check():
    if time.monotonic()>=deadline: raise TimeoutError('Benchmark block budget exhausted')
run._CHECK=check
if payload['track']==2:
    observed={}; original_predict=run._MODEL.predict_samples
    def recording_predict(samples,**kwargs):
        output=original_predict(samples,**kwargs); torch.cuda.synchronize()
        observed['probabilities']=output['probabilities']; observed['predicted']=time.monotonic()
        return output
    run._MODEL.predict_samples=recording_predict
    official=run.rank_gallery({'attribute_names':payload['samples'][0]['attribute_names'],
                               'gallery':payload['samples'],'queries':payload['queries']})
    p=observed['probabilities']; predicted=observed['predicted']; distances=np.asarray(official['distances'])
    if set(official)!={'distances'} or distances.shape!=(len(payload['queries']),len(p)) or not np.isfinite(distances).all():
        raise ValueError('Invalid official rank_gallery distances contract')
else:
    p=np.asarray(run.predict_batch(payload['samples']),dtype=np.float64); torch.cuda.synchronize(); predicted=time.monotonic()
np.save(sys.argv[3],p,allow_pickle=False)
rank_seconds=0.; permutations=True; metric_parts=[]; labels=np.asarray(payload['labels']); metric_started=0.
if payload['track']==2:
    rankings=np.empty(distances.shape,dtype=np.int64)
    for start in range(0,len(distances),payload['query_block_size']):
        check(); rankings[start:start+payload['query_block_size']]=np.argsort(distances[start:start+payload['query_block_size']],axis=1,kind='stable')
    np.save(sys.argv[4],rankings,allow_pickle=False)
    reopened=np.load(sys.argv[4],allow_pickle=False)
    for row in reopened:
        check()
        if not np.array_equal(np.sort(row),np.arange(len(p))): raise ValueError('Invalid ranking permutation')
    if not np.array_equal(reopened,rankings): raise ValueError('Ranking serialization changed')
    rank_seconds=time.monotonic()-predicted
    metric_started=time.monotonic()
    for start in range(0,len(distances),payload['query_block_size']):
        check(); part=metricmod.retrieval_metrics(np.asarray(payload['queries'][start:start+payload['query_block_size']]),labels,
                                               distances[start:start+payload['query_block_size']])
        metric_parts.append(part)
    values={k:np.concatenate([part['per_query'][k] for part in metric_parts]) for k in metric_parts[0]['per_query']}
    metrics={('m'+k if k in {'AP','ADM','INP'} else k):float(v.mean()) for k,v in values.items()}
else:
    metric_started=time.monotonic(); check()
    metrics=metricmod.track1_metrics(labels,p,domains=np.asarray(payload['domains']))
metric_seconds=time.monotonic()-metric_started
ended=time.monotonic(); check()
report={'startup_seconds':loaded-started,'inference_seconds':predicted-loaded,'ranking_seconds':rank_seconds,
        'metrics_seconds':metric_seconds,'metrics':metrics,'total_seconds':ended-started,'images':len(p),'queries':len(payload['queries']) if payload['track']==2 else 0,
        'rss_peak_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        'vram_peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'vram_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'torch':str(torch.__version__),'cuda':str(torch.version.cuda),'device':torch.cuda.get_device_name(),
        'ranking_permutations_verified':permutations,'ranking_roundtrip_verified':payload['track']==2,
        'official_api':'rank_gallery' if payload['track']==2 else 'predict_batch','image_inference_passes':1}
Path(sys.argv[5]).write_text(json.dumps(report,allow_nan=False))
'''


def benchmark(config_path,track):
    """Heavy full-public inference/retrieval; explicitly USER-RUN only."""
    from .protocol import roots,require_record,stage_budget,records,check_campaign_disk
    manifest,candidate,validated,cache,artifacts=_context(config_path,track); run,reports=roots(manifest)
    name='runtime_track'+str(track)
    if (reports/(name+'.json')).exists(): return require_record(manifest,name)
    packaged=require_record(manifest,'package_track'+str(track))
    if packaged['candidate']!=candidate: raise ValueError('Package selected candidate mismatch')
    rows=records(manifest,'public')
    if rows['paths']!=cache['paths'].tolist() or hash_object(rows['queries'].tolist())!=validated['cache_metadata']['official_queries_sha256']:
        raise ValueError('Runtime public cohort/query order mismatch')
    samples=[{'image_path':str((Path(manifest['config']['data']['images'])/p).resolve()),'attribute_names':manifest['attribute_names']} for p in rows['paths']]
    folder=run/'runtime'/('track'+str(track)); folder.mkdir(parents=True,exist_ok=True)
    rankings=folder/'rankings.npy'; probabilities=folder/'probabilities.npy'
    if probabilities.exists() or rankings.exists(): raise FileExistsError('Unrecorded benchmark artifacts; do not repeat interrupted attempt')
    with stage_budget(manifest,'runtime') as clock:
        remaining=min(3600.,clock.remaining_seconds)
        if remaining<=0: raise TimeoutError('Finalization budget exhausted')
        check_campaign_disk(manifest,3*Path(packaged['archive']).stat().st_size+len(rows['paths'])*40*8+
                            (len(rows['queries'])*len(rows['paths'])*16 if track==2 else 0))
        with tempfile.TemporaryDirectory(dir=folder) as scratch:
            scratch=Path(scratch); root=scratch/'package'
            with zipfile.ZipFile(packaged['archive']) as archive:
                for item in archive.infolist():
                    p=Path(item.filename)
                    if p.is_absolute() or '..' in p.parts: raise ValueError('Unsafe archive path')
                archive.extractall(root)
            metrics_source=scratch/'metrics.py'; shutil.copyfile(Path(__file__).parent.parent/'metrics.py',metrics_source)
            payload={'samples':samples,'queries':rows['queries'].tolist(),'labels':rows['labels'].tolist(),'domains':rows['domains'].tolist(),
                     'metrics_source':str(metrics_source),'track':track,'timeout':remaining,
                     'cpu_threads':manifest['config']['runtime']['cpu_threads'],'query_block_size':manifest['config']['runtime']['query_block_size']}
            input_path=scratch/'input.json'; input_path.write_text(json.dumps(payload,allow_nan=False))
            script=scratch/'worker.py'; script.write_text(WORKER); measured=scratch/'measured.json'
            run_bounded([sys.executable,'-I',str(script),str(root),str(input_path),str(probabilities),str(rankings),str(measured)],
                        timeout=min(3600.,clock.remaining_seconds),cwd=scratch)
            measurements=read_json(measured)
        clock.check(); actual=np.load(probabilities,allow_pickle=False); expected=cache['probabilities']
        if actual.shape!=expected.shape or not np.isfinite(actual).all(): raise ValueError('Invalid package benchmark probabilities')
        difference=float(np.max(np.abs(actual-expected)))
        if difference>PUBLIC_GPU_PROBABILITY_TOLERANCE: raise ValueError('New package versus same-protocol public GPU cache parity failed')
        if measurements['total_seconds']>=3600: raise TimeoutError('Official 3600-second local bound exceeded')
        admission=verify_runtime_admission(track,measurements['metrics'],validated['reference'])
        artifacts.update(packaged['artifacts']); artifacts[str(probabilities)]=sha256_file(probabilities)
        if track==2: artifacts[str(rankings)]=sha256_file(rankings)
        result={'status':'completed','protocol_sha256':manifest['protocol_sha256'],'track':track,'candidate':candidate,
                'measurements':measurements,'timeout_seconds':min(3600.,remaining),'archive_sha256':packaged['archive_sha256'],
                'quality_admission_after_export':admission,
                'public_gpu_cache_parity':{'passed':True,'maximum_absolute_probability_difference':difference,
                    'absolute_tolerance':PUBLIC_GPU_PROBABILITY_TOLERANCE,'cache_sha256':sha256_file(validated['cache']),
                    'scope':'New GPU package output versus selected same-protocol effective public cache; distinct from same-CPU export parity'},
                'artifacts':artifacts,'limitations':['Local hardware observations do not guarantee organizer hardware or remote ingestion.','No upload or guaranteed hidden-score gain.']}
        clock.check(); publish(reports/(name+'.json'),result)
    return result
