"""Official starter packaging and local ingestion-contract checks."""
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from .io import sha256_file


def _archive(folder, destination):
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.write(path,path.relative_to(folder))
    return destination


def package_starter(repo,track,destination):
    if track not in (1,2): raise ValueError('track must be 1 or 2')
    source=Path(repo)/f'examples/task{track}/sample_code_submission'
    if not (source/'run.py').is_file(): raise FileNotFoundError(source)
    return _archive(source,destination)


def package_model(repo,track,checkpoint,destination,*,method='l1',prior=None,temperature=1.):
    from .runtime import Predictor
    predictor=Predictor(checkpoint,device='cpu')
    if method in {'weighted_nll','weighted_nll_sum','marginalization'} and predictor.kind!='evidential':
        raise ValueError('Selected scoring rule needs an evidential model')
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        shutil.copy2(Path(repo)/f'examples/task{track}/sample_code_submission/metadata.yaml',root/'metadata.yaml')
        (root/'assets').mkdir()
        shutil.copy2(checkpoint,root/'assets/model.pt')
        (root/'upar').mkdir()
        for name in ['__init__.py','runtime.py','model.py','scoring.py','calibration.py']:
            shutil.copy2(Path(__file__).parent/name,root/'upar'/name)
        config={'track':track,'method':method,'prior':prior,'temperature':temperature,'batch_size':32}
        (root/'assets/inference.json').write_text(json.dumps(config))
        (root/'run.py').write_text('''from pathlib import Path
import json
from upar.runtime import Predictor
from upar.scoring import score_distances
HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'assets/inference.json').read_text())
BATCH_SIZE=CONFIG['batch_size']
_MODEL=None

def load_model():
    global _MODEL
    _MODEL=Predictor(HERE/'assets/model.pt',temperature=CONFIG['temperature'])

def predict_batch(samples):
    if _MODEL is None: load_model()
    return _MODEL.predict_samples(samples)['probabilities'].tolist()

def predict_image(sample):
    return predict_batch([sample])[0]

def rank_gallery(sample):
    if _MODEL is None: load_model()
    gallery=[dict(g,attribute_names=sample['attribute_names']) for g in sample['gallery']]
    output=_MODEL.predict_samples(gallery,batch_size=BATCH_SIZE)
    return {'distances': score_distances(sample['queries'],output['probabilities'],method=CONFIG['method'],uncertainty=output.get('uncertainty'),prior=CONFIG['prior'])}
''')
        return _archive(root,destination)


_SMOKE_SCRIPT = r'''
import importlib.util,json,socket,sys
from pathlib import Path
import numpy as np
from PIL import Image

def denied(*args,**kwargs): raise RuntimeError('Network access disabled in smoke harness')
socket.create_connection=denied
socket.socket.connect=denied
socket.socket.connect_ex=denied
root=Path.cwd()
spec=importlib.util.spec_from_file_location('submission_run',root/'run.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
track=int(sys.argv[1]); names=json.loads(sys.argv[2])
if hasattr(module,'load_model'): module.load_model()
samples=[]
for i in range(3):
    p=root/f'fixture_{i}.png'
    Image.new('RGB',(32,64),(i*50,80,100)).save(p)
    samples.append({'index':i,'image':p.name,'image_path':str(p),'attribute_names':names})
if track==1:
    batch=np.asarray(module.predict_batch(samples) if hasattr(module,'predict_batch') else [module.predict_image(s) for s in samples],dtype=float)
    assert batch.shape==(3,40) and np.isfinite(batch).all() and ((batch>=0)&(batch<=1)).all()
    if hasattr(module,'predict_image'):
        np.testing.assert_allclose(batch[0],module.predict_image(samples[0]),rtol=1e-4,atol=1e-5)
else:
    gallery=[{k:v for k,v in s.items() if k!='attribute_names'} for s in samples]
    out=module.rank_gallery({'gallery':gallery,'queries':[[0]*40,[1]*40],'attribute_names':names})
    assert set(out)=={'distances'}
    values=np.asarray(out['distances'])
    assert values.shape==(2,3) and np.isfinite(values).all()
print(json.dumps({'passed':True,'track':track,'network_disabled':True,'isolation':'Python socket guard; not an OS network namespace','runtime_prefix':sys.prefix,'runtime_executable':sys.executable}))
'''


def smoke_submission(archive,track,*,python=None,timeout=180):
    from .data import ATTRIBUTE_NAMES
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                path=Path(name)
                if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe archive path')
            if not {'run.py','metadata.yaml'} <= set(zf.namelist()): raise ValueError('Missing archive-root entrypoint or metadata')
            zf.extractall(root)
        import os
        environment=dict(os.environ)
        environment.pop('PYTHONPATH',None)
        environment['CUDA_VISIBLE_DEVICES']=''
        environment['OMP_NUM_THREADS']='2'
        run=subprocess.run([os.path.abspath(python or sys.executable),'-c',_SMOKE_SCRIPT,str(track),json.dumps(ATTRIBUTE_NAMES)],cwd=root,text=True,capture_output=True,timeout=timeout,env=environment)
        if run.returncode: raise RuntimeError(f'Submission smoke failed:\n{run.stdout}\n{run.stderr}')
        result=json.loads(run.stdout.strip().splitlines()[-1])
        result.update(archive_sha256=sha256_file(archive),python=python or sys.executable)
        return result


def gate0_report(audit,smoke,tests,remote):
    local=bool(audit.get('passed') and audit.get('images_checked') and tests.get('passed') and all(smoke.get(str(t),{}).get('passed') for t in (1,2)))
    accepted=all(remote.get(str(t),{}).get('accepted') and remote[str(t)].get('submission_id') and remote[str(t)].get('archive_sha256')==smoke.get(str(t),{}).get('archive_sha256') for t in (1,2))
    return {'passed':bool(local and accepted),'local_passed':local,'remote_status':'accepted' if accepted else 'pending','scorer_parity':'published formulas only; private organizer scorer unavailable','requirements':{'full_data_audit':audit.get('passed') and audit.get('images_checked'),'tests':tests.get('passed'),'local_submissions':smoke,'remote_submissions':remote}}
