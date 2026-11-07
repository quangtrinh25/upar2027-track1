"""Charge completed, failed and currently active campaign jobs together."""
from __future__ import annotations

import json
import math
import os
import fcntl
import re
import shutil
import signal
import threading
import time
from contextlib import contextmanager, ExitStack
from pathlib import Path

from .artifacts import write_once_json

STAGE_KEYS = ('preflight_hours','train_hours','inference_hours','head_hours','retrieval_hours','public_package_hours')


def _number(value, name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError(f'Invalid resource {name}')
    return float(value)


def require_completed_producer(protocol, name, task_stage, result_path, *, expected,
                               measured_wall_seconds=0., maximum_hours=None):
    """A published result is usable only after its one producer closes successfully."""
    reports=Path(protocol['artifact_root'])/'reports'
    marker_path=reports/f'{name}_started.json'
    journal_path=reports/f'{name}_journal.json'
    if not marker_path.is_file() or not journal_path.is_file():
        raise RuntimeError('Completed producer marker/journal missing')
    marker=json.loads(marker_path.read_text()); journal=json.loads(journal_path.read_text())
    identity={'stage':name,'task_stage':task_stage,'protocol_sha256':protocol['protocol_sha256']}
    if any(record.get(k)!=v for record in (marker,journal) for k,v in identity.items()):
        raise ValueError('Completed producer identity mismatch')
    if journal.get('status')!='completed' or journal.get('no_retry') is not True or journal.get('error') is not None:
        raise RuntimeError('Required producer did not complete successfully')
    seconds=_number(journal.get('wall_seconds'),'producer wall seconds')
    hours=_number(journal.get('wall_hours'),'producer wall hours')
    cap=_number(marker.get('cap_hours'),'producer cap')
    if (not math.isclose(hours,seconds/3600,rel_tol=0,abs_tol=1e-9) or
        journal.get('cap_hours')!=cap or cap<=0 or hours>cap+1e-9 or
        seconds+1e-9<_number(measured_wall_seconds,'measured wall') or
        (maximum_hours is not None and cap>maximum_hours+1e-9)):
        raise ValueError('Completed producer wall/cap binding differs')
    path=Path(result_path)
    if path.parent.resolve()!=reports.resolve() or not path.is_file():
        raise RuntimeError('Required producer result is missing/outside campaign reports')
    result=json.loads(path.read_text())
    required={'protocol_sha256':protocol['protocol_sha256'],**expected}
    if any(result.get(k)!=v for k,v in required.items()):
        raise ValueError('Completed producer result identity mismatch')
    return result


def resource_snapshot(protocol):
    root = Path(protocol['artifact_root'])
    resources = protocol['budget']['resources']
    total = _number(resources['total_hours'],'total')
    historical = _number(resources['historical_hours'],'historical')
    reserve = _number(resources['reserve_hours'],'reserve')
    charged = {key:0. for key in STAGE_KEYS if key in resources}
    active, seen = [], set()
    for folder in (root/'reports',root/'journals'):
        names = {p.name.removesuffix('_journal.json') for p in folder.glob('*_journal.json')}
        names |= {p.name.removesuffix('_started.json') for p in folder.glob('*_started.json')}
        for name in sorted(names):
            if name in seen:
                raise ValueError('Duplicate campaign job accounting record')
            seen.add(name)
            marker_path, journal_path = folder/(name+'_started.json'),folder/(name+'_journal.json')
            if not marker_path.is_file():
                raise ValueError('Resource journal is missing its attempt marker')
            marker = json.loads(marker_path.read_text())
            if marker.get('protocol_sha256') != protocol['protocol_sha256']:
                raise ValueError('Resource marker protocol mismatch')
            stage = marker.get('task_stage')
            if stage not in charged:
                raise ValueError('Unknown registered resource stage')
            if journal_path.is_file():
                record = json.loads(journal_path.read_text())
                if record.get('protocol_sha256') != protocol['protocol_sha256'] or record.get('task_stage') != stage:
                    raise ValueError('Resource journal protocol/stage mismatch')
                if record.get('status') not in ('completed','failed'):
                    raise ValueError('Resource journal is not a closed job')
                seconds = _number(record.get('wall_seconds'),'wall seconds')
                hours = _number(record.get('wall_hours'),'wall hours')
                if not math.isclose(hours,seconds/3600,rel_tol=0,abs_tol=1e-9):
                    raise ValueError('Resource wall charge is inconsistent')
            else:
                pid = marker.get('pid')
                if type(pid) is not int or pid<=0:
                    raise RuntimeError('Unclosed job has no verified live process; resource decision required')
                try:
                    os.kill(pid,0)
                except OSError:
                    raise RuntimeError('Unclosed job is no longer live; resource decision required') from None
                start = _number(marker.get('started_unix_seconds'),'start time')
                seconds = time.time()-start
                if seconds<0:
                    raise ValueError('Resource marker start time is in the future')
                hours = seconds/3600
                active.append(name)
            charged[stage] += hours
    spent = sum(charged.values())
    remaining_stage = {key:resources[key]-value for key,value in charged.items()}
    return {'historical_hours':historical,'reserve_hours':reserve,'reserve_spendable':False,
            'new_spent_hours':spent,'total_charged_hours':historical+spent,
            'charged_stage_hours':charged,'remaining_stage_hours':remaining_stage,
            'remaining_total_hours':total-reserve-historical-spent,'active_jobs':active}


def require_budget(protocol, task_stage, *, projected_hours=0.):
    projection = _number(projected_hours,'projection')
    snapshot = resource_snapshot(protocol)
    if task_stage not in snapshot['remaining_stage_hours']:
        raise ValueError('Unknown registered resource stage')
    if any(value < -1e-9 for value in snapshot['remaining_stage_hours'].values()):
        raise RuntimeError('A registered stage resource cap has already failed')
    if projection>snapshot['remaining_stage_hours'][task_stage]+1e-9:
        raise RuntimeError('Projected job exceeds the remaining stage resource cap')
    if projection>snapshot['remaining_total_hours']+1e-9:
        raise RuntimeError('Projected job exceeds the carried total minus reserve')
    snapshot['remaining_hours'] = min(snapshot['remaining_total_hours'],snapshot['remaining_stage_hours'][task_stage])
    return snapshot


@contextmanager
def campaign_job(protocol, name, task_stage, *, projected_hours=0., maximum_hours=None, gpu=False):
    """One bounded, immutable user-run attempt; failed work still consumes budget."""
    if not isinstance(name,str) or not re.fullmatch(r'[a-z][a-z0-9_]*',name):
        raise ValueError('Invalid job name')
    root=Path(protocol['artifact_root'])
    reports=root/'reports'
    reports.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        locks=[]
        if gpu:
            locks.append(('gpu_lane.lock','Another GPU job is active'))
        if task_stage=='retrieval_hours':
            locks.append(('retrieval_stage.lock','Another retrieval job is active'))
        for filename,message in locks:
            handle=stack.enter_context((reports/filename).open('a'))
            try:
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(message) from None
        allowance=require_budget(protocol,task_stage,projected_hours=projected_hours)['remaining_hours']
        if maximum_hours is not None:
            allowance=min(allowance,_number(maximum_hours,'maximum job hours'))
        if allowance<=0:
            raise RuntimeError('Registered resource allowance exhausted')
        free=shutil.disk_usage(root).free/(1024**3)
        if free<protocol['budget']['resources'].get('minimum_free_gib',4.):
            raise RuntimeError('Insufficient free space; historical artifacts are preserved')
        started=time.monotonic()
        write_once_json(reports/f'{name}_started.json',{
            'stage':name,'task_stage':task_stage,'protocol_sha256':protocol['protocol_sha256'],
            'pid':os.getpid(),'started_unix_seconds':time.time(),'started_monotonic':started,
            'cap_hours':allowance,'projected_hours':projected_hours,'user_run_only':True})
        status,error='failed',None
        saved={}
        def checkpoint():
            if time.monotonic()-started>=allowance*3600:
                raise TimeoutError('Registered job wall cap exhausted; no automatic retry')
            require_budget(protocol,task_stage)
        def stop(signum,frame):
            raise TimeoutError('Registered job interrupted or wall cap exhausted; no automatic retry')
        try:
            if threading.current_thread() is threading.main_thread():
                if signal.getitimer(signal.ITIMER_REAL)[0]:
                    raise RuntimeError('Refuse overlapping stage alarms')
                for sig in (signal.SIGALRM,signal.SIGTERM):
                    saved[sig]=signal.getsignal(sig)
                    signal.signal(sig,stop)
                signal.setitimer(signal.ITIMER_REAL,allowance*3600)
            yield checkpoint
            checkpoint()
            status='completed'
        except BaseException as exc:
            error=f'{type(exc).__name__}: {exc}'
            raise
        finally:
            if saved:
                signal.setitimer(signal.ITIMER_REAL,0)
                for sig,handler in saved.items():
                    signal.signal(sig,handler)
            wall=time.monotonic()-started
            write_once_json(reports/f'{name}_journal.json',{
                'stage':name,'task_stage':task_stage,'protocol_sha256':protocol['protocol_sha256'],
                'status':status,'wall_seconds':wall,'wall_hours':wall/3600,
                'cap_hours':allowance,'error':error,'no_retry':True})
