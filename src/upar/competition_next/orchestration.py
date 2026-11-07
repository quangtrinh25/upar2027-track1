"""Explicit campaign storage and evidence-only review preparation."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml


def jev_payload_sha256(value):
    from upar.jev import _canonical
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def read_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Campaign configuration must be an object')
    for name in ('repo_root', 'artifact_root'):
        if not isinstance(value.get(name), str) or not Path(value[name]).is_absolute():
            raise ValueError('Campaign repository/storage roots must be absolute')
    return value


def stage_output(config: dict, name: str) -> Path:
    if Path(name).name != name or not name.endswith('.json') or name.startswith('.'):
        raise ValueError('Report name must be a plain JSON filename')
    return Path(config['artifact_root']) / 'reports' / name


def copy_plans(config: dict) -> dict:
    repo = Path(config['repo_root'])
    target = Path(config['artifact_root']).parent / 'docs'
    target.mkdir(parents=True, exist_ok=True)
    result = {}
    for key in ('plan_path', 'master_source_path'):
        source = Path(config[key])
        if not source.is_absolute():
            source = repo / source
        if source.suffix != '.md' or not source.resolve().is_relative_to(repo.resolve()):
            raise ValueError('Plans must be Markdown files from this repository')
        destination = target / source.name
        content = source.read_bytes()
        if destination.exists():
            if destination.read_bytes() != content:
                raise ValueError('An existing different plan is preserved')
        else:
            with destination.open('xb') as handle:
                handle.write(content)
        result[key] = str(destination)
    return result


def build_review_request(protocol: dict, preparation: dict, compatibility=None) -> dict:
    """Send aggregate measurements only; no credential/cache/image data."""
    return {
        'model': 'jev-latest',
        'state': {
            'project': 'UPAR 2027 separately user-authorized competition_next',
            'remote_goal': {'track1': .80, 'track2': .65},
            'remote_incumbent': {'track1': .7282, 'track2': .5241},
            'research': {'gate1': 'FAIL', 'gate2': 'CLOSED'},
            'historical_recipe_closures_unchanged': True,
            'protocol_sha256': protocol['protocol_sha256'],
            'new_partition_counts': copy.deepcopy(protocol['partition_summaries']),
            'fit_only_group_support': copy.deepcopy(protocol['support_audit']),
            'budget': copy.deepcopy(protocol['budget']),
            'gates': copy.deepcopy(protocol['gates']),
            'limitations': copy.deepcopy(protocol['limitations']),
            'preparation_evidence': copy.deepcopy(preparation),
            'ema_offline_compatibility':copy.deepcopy(compatibility) if compatibility is not None else {'status':'unmeasured'},
            'gpu_readiness': 'unmeasured',
            'policy': 'Agent runs light CPU checks only. User runs all heavy jobs. No main training until new measured preflight ready and new-evidence review. No retry, sweep, relaxed gate, public opening or upload now.',
            'prospective_steps': 'WS1 group audit/structured likelihood/unary before agreement. WS2 family ensemble first. FRDL-style matched heads on DEV fitfold only. DINOv3-T D-A after D1, component and diversity gates, D-B conditional. EMA/augmentation separately bounded. One prospective LOCKBOX round after cross-fit, then one Public confirmation.',
        },
        'questions': {
            'next_preparation_action': {
                'type': 'choice',
                'instructions': 'Select action supported by measured preparation evidence and access status, respecting all user constraints. Do not infer GPU readiness or quality gain from unit tests.',
                'criteria': {
                    'continue_cpu_user_preflight_after_access': 'CPU workstreams may proceed; D1 user preflight only with authorized pinned generic weights and compatibility checks.',
                    'continue_cpu_dino_access_blocked': 'Continue authorized CPU work and independent EMA user preflight only if its offline CPU compatibility passed. DINO access is blocked so no DINO GPU command yet.',
                    'pause_for_protocol_defect': 'A concrete material protocol defect prevents even new workstream execution.',
                    'insufficient_evidence': 'Evidence insufficient to recommend the next measured preparation action.',
                },
            },
            'historical_exposure_disclosure': {
                'type': 'choice',
                'instructions': 'Assess the interpretation of a prospective LOCKBOX drawn from previously used TUNE. It must never be called untouched through the entire historical pipeline.',
                'criteria': {
                    'prospective_only_disclosure_adequate': 'Useful prospective protection for new steps, with explicit historical selection exposure.',
                    'protocol_claims_untouched_incorrectly': 'Evidence contains an incorrect claim of fully untouched historical test data.',
                    'insufficient_evidence': 'Cannot determine disclosure adequacy.',
                },
            },
            'main_training_allowed_now': {
                'type': 'noul',
                'instructions': 'May main training start now without a measured READY result for this separate campaign? User requires measured stage resource caps and fresh evidence first.',
                'criteria': {'true': 'All prerequisites actually measured and satisfied.',
                             'false': 'Preflight unmeasured or another prerequisite missing; main training remains unadmitted.'},
            },
        },
    }


def build_preflight_review_request(protocol,arm,evidence,evidence_sha256,resources):
    return {'model':'jev-latest','state':{
        'project':'UPAR competition_next measured preflight',
        'protocol_sha256':protocol['protocol_sha256'],'arm':arm,'evidence_sha256':evidence_sha256,
        'preflight':copy.deepcopy(evidence),'budget':copy.deepcopy(resources),'gates':copy.deepcopy(protocol['gates']),
        'research':{'gate1':'FAIL','gate2':'CLOSED'},'quality_gain':'unmeasured',
        'constraints':'User runs GPU. READY must satisfy all stage/VRAM/cumulative caps. No retry, fallback, resource relaxation or guaranteed remote gain. Completed review cannot be rerolled. Model access and compatibility are required independently.'},
        'questions':{'train_admission':{
            'type':'choice','instructions':'Can this exact arm proceed to its fixed24epoch user-run training based on measured readiness and carried resources? Missing/failed prerequisites cannot be waived.',
            'criteria':{'train':'Measured READY and exactrecipe/source/cumulativebudget all pass; authorize handoff for this fixed arm only.',
                        'stop':'Measured failure or missing requirement; this arm training stays closed.',
                        'insufficient_evidence':'Evidence insufficient for a training handoff; keep training closed.'}}}}
