"""Evaluation from caches, preserving split and annotation provenance."""
from pathlib import Path
import numpy as np
from .cache import load_cache,save_cache
from .data import load_split
from .io import hash_object,sha256_file,write_json
from .metrics import track1_metrics,retrieval_metrics
from .scoring import score_distances


def evaluate_arrays(queries,labels,probabilities,*,uncertainty=None,method='l1',prior=None,block_size=256):
    # Consume distance blocks immediately rather than materializing the complete matrix.
    per_query={}
    valid=[]
    for start in range(0,len(queries),block_size):
        q=queries[start:start+block_size]
        distances=score_distances(q,probabilities,method=method,uncertainty=uncertainty,prior=prior,block_size=block_size)
        part=retrieval_metrics(q,labels,distances)
        for key,value in part['per_query'].items(): per_query.setdefault(key,[]).append(value)
        valid.append(part['valid_query_mask'])
    values={key:np.concatenate(parts) for key,parts in per_query.items()}
    result={('m'+key if key in {'AP','ADM','INP'} else key):float(value.mean()) for key,value in values.items()}
    result['per_query']=values
    result['valid_query_mask']=np.concatenate(valid)
    return result


def evaluate_cache(config,cache_path):
    split=load_split(config['data']['annotations'],config.get('evaluation',{}).get('split','val'))
    cache=load_cache(cache_path,expected_paths=split.paths,expected_metadata={'annotations_sha256':hash_object(split.hashes)})
    method=config.get('retrieval',{}).get('method','l1')
    retrieval=evaluate_arrays(split.queries,split.labels,cache['probabilities'],uncertainty=cache.get('uncertainty'),method=method,prior=config.get('retrieval',{}).get('prior'))
    return {'track1':track1_metrics(split.labels,cache['probabilities'],domains=split.domains),'track2':retrieval,'method':method,'cache':str(cache_path),'cache_sha256':sha256_file(cache_path),'scorer_parity':'published formulas; private organizer scorer unavailable'}


def predict_split(config,checkpoint,destination,*,partition=None):
    from .runtime import Predictor
    split=load_split(config['data']['annotations'],'train' if partition else config.get('evaluation',{}).get('split','val'))
    if partition:
        import json
        manifest=json.loads(Path(partition).read_text())
        if manifest['hashes']!=split.hashes: raise ValueError('Partition provenance mismatch')
        lookup={p:i for i,p in enumerate(split.paths)}
        indices=[lookup[p] for p in manifest['tune']]
        split.paths=[split.paths[i] for i in indices]
        split.domains=split.domains[indices]
    predictor=Predictor(checkpoint,temperature=config.get('calibration',{}).get('temperature',1.))
    samples=[{'image_path':str(Path(config['data']['images'])/p),'attribute_names':split.attribute_names} for p in split.paths]
    arrays=predictor.predict_samples(samples,batch_size=config.get('inference',{}).get('batch_size',32))
    metadata={'checkpoint_sha256':sha256_file(checkpoint),'annotations_sha256':hash_object(split.hashes),'preprocess_sha256':hash_object({'image_size':predictor.image_size,'resize':'PIL bilinear','normalization':'imagenet','temperature':predictor.temperature}),'config_sha256':hash_object(config)}
    save_cache(destination,paths=split.paths,domains=split.domains,attribute_names=split.attribute_names,metadata=metadata,**arrays)
    return {'cache':str(destination),'images':len(samples),'metadata':metadata}
