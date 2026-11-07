"""Typed, advisory-only TypeSafe Jev reviews of explicit evidence packets.

API contract: https://docs.typesafe.ai/api . No training or gate mutations.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests

ENDPOINT = 'https://api.typesafe.ai/v1/systemone'


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError('Jev payload must be finite JSON') from None


def validate_request(request):
    """Support only the Noul/Choice types used by this project."""
    if not isinstance(request, dict) or set(request) != {'model', 'state', 'questions'}:
        raise ValueError('Request needs exactly model, state and questions')
    if not isinstance(request['model'], str) or not request['model'].strip():
        raise ValueError('Request model must be a nonempty string')
    if not isinstance(request['state'], (str, dict, list)):
        raise ValueError('Request state must be a string, object or array')
    questions = request['questions']
    if not isinstance(questions, dict) or not questions:
        raise ValueError('Request needs at least one question')
    for name, question in questions.items():
        if not isinstance(name, str) or not name or not isinstance(question, dict):
            raise ValueError('Question ids must map to objects')
        if set(question) - {'type', 'instructions', 'criteria'}:
            raise ValueError('Unsupported question fields')
        if not isinstance(question.get('instructions'), (str, dict, list)) or not question['instructions']:
            raise ValueError('Each question needs nonempty instructions')
        kind = question.get('type')
        criteria = question.get('criteria')
        if kind == 'choice':
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise ValueError('Choice needs 2 to 255 criteria')
            if any(not isinstance(k, str) or not k for k in criteria):
                raise ValueError('Choice option names must be nonempty strings')
        elif kind == 'noul':
            if criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {'true', 'false'}):
                raise ValueError('Noul criteria may describe true and false only')
        else:
            raise ValueError('This client supports only noul and choice questions')
        if criteria is not None:
            allowed = (str, dict, list, type(None)) if kind == 'choice' else (str, dict, list)
            if any(not isinstance(v, allowed) for v in criteria.values()):
                raise ValueError('Invalid criterion description')
    _canonical(request)


def _probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Response probabilities must be numbers')
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Response probabilities must be finite and in [0, 1]')


def validate_response(request, response):
    validate_request(request)
    if not isinstance(response, dict) or not isinstance(response.get('model'), str) or not response['model']:
        raise ValueError('Response must identify the model that performed the evaluation')
    answers = response.get('answers')
    if not isinstance(answers, dict) or set(answers) != set(request['questions']):
        raise ValueError('Response answer ids must match all request question ids')
    usage = response.get('usage')
    if not isinstance(usage, dict) or any(
        type(usage.get(k)) is not int or usage[k] < 0 for k in ('input_tokens', 'output_tokens')
    ):
        raise ValueError('Response must include nonnegative token usage')
    for name, question in request['questions'].items():
        answer = answers[name]
        if not isinstance(answer, dict) or answer.get('type') != question['type']:
            raise ValueError('Response answer type must match its question')
        if question['type'] == 'noul':
            _probability(answer.get('noul'))
            continue
        probabilities = answer.get('probabilities')
        if not isinstance(probabilities, dict) or set(probabilities) != set(question['criteria']):
            raise ValueError('Choice must provide probabilities for exactly the requested options')
        for probability in probabilities.values():
            _probability(probability)
        if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError('Choice probabilities must sum to one')
        chosen = answer.get('choice')
        if not isinstance(chosen, str) or chosen not in probabilities:
            raise ValueError('Choice answer must be one of the requested options')
        if probabilities[chosen] < max(probabilities.values()) - 1e-8:
            raise ValueError('Choice answer must have the highest probability')
        _probability(answer.get('confidence'))
    _canonical(response)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _record(request):
    return {
        'schema_version': 1,
        'provider': 'TypeSafe',
        'endpoint': ENDPOINT,
        'advisory_only': True,
        'requested_at_utc': _now(),
        'status': 'pending',
        'request': copy.deepcopy(request),
        'request_sha256': hashlib.sha256(_canonical(request).encode()).hexdigest(),
        'client_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def evaluate(request):
    """Perform one HTTPS call; never follow redirects or automatically retry."""
    validate_request(request)
    key = os.environ.get('TYPESAFE_API_KEY', '').strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError('Set a valid TYPESAFE_API_KEY in this process environment')
    record = _record(request)
    try:
        response = requests.post(
            ENDPOINT, json=record['request'],
            headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
            timeout=(10, 120), allow_redirects=False,
        )
    except requests.RequestException:
        raise RuntimeError('TypeSafe network request failed; no automatic retry was made') from None
    if response.status_code != 200:
        # Do not echo a server error body or request headers into logs.
        raise RuntimeError(f'TypeSafe returned HTTP {response.status_code}; no automatic retry was made')
    try:
        value = response.json()
    except ValueError:
        raise ValueError('TypeSafe returned an invalid JSON response') from None
    validate_response(request, value)
    record.update(status='completed', completed_at_utc=_now(), response=value,
                  request_id=response.headers.get('x-request-id'))
    return record


def _replace_record(path, record):
    """Publish the completed/error record atomically after reserving its path."""
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_review(request, output):
    """Reserve a new record before sending; keep failures for audit, too.

    Existing records, including failed/pending attempts, are never reused. An
    interrupted pending request needs inspection before any deliberate retry.
    """
    validate_request(request)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _record(request)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    try:
        record = evaluate(request)
    except (ValueError, RuntimeError) as error:
        record.update(status='failed', completed_at_utc=_now(), error=str(error))
        _replace_record(path, record)
        raise
    _replace_record(path, record)
    return record
