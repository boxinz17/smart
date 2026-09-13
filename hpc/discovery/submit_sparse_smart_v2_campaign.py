#!/usr/bin/env python3
"""Plan or submit case-aligned SparseSMARTv2 pools; never fit or aggregate here."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

RUN_PREFIX = Path('/scratch2')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'PREEMPTED',
            'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'SPECIAL_EXIT'}


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def atomic(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def immutable(path, payload):
    if path.exists():
        require(not path.is_symlink() and path.is_file() and path.read_bytes() == payload,
                f'Immutable campaign input differs: {path}')
        return
    with path.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444)


def load_inputs(root):
    plan, manifest = read(root / 'plan.json'), read(root / 'source-manifest.json')
    fingerprint = hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != 'plan_fingerprint'},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(plan['plan_fingerprint'] == fingerprint, 'Plan fingerprint mismatch')
    require(Path(plan['root']).resolve() == root, 'Plan belongs to another root')
    require(plan['source_manifest_sha256'] == sha(root / 'source-manifest.json'), 'Source manifest identity mismatch')
    require(plan['n_cases'] == len(plan['cases']) and plan['n_tasks'] == len(plan['tasks']) and plan['tasks'], 'Plan counts invalid')
    require((root / 'work-items.tsv').read_text().splitlines() == [str(i) for i in range(plan['n_tasks'])],
            'Full work items must be consecutive canonical task IDs')
    source = (root / 'source').resolve()
    require(manifest['schema_version'] == 1 and Path(manifest['source_root']).resolve() == source,
            'Source manifest identifies a different frozen tree')
    require(isinstance(manifest.get('files'), dict) and manifest['files'], 'Source manifest is empty')
    for name, expected in manifest['files'].items():
        part = Path(name); path = (source / part).resolve()
        require(not part.is_absolute() and '..' not in part.parts and path.is_relative_to(source)
                and sha(path) == expected, f'Frozen source changed: {name}')
    for name in ('sparse_smart_v2_campaign_prepare.sbatch', 'sparse_smart_v2_campaign_pool.sbatch', 'sparse_smart_v2_worker.sh'):
        require((source / 'hpc/discovery' / name).is_file(), f'Missing campaign script: {name}')
    return plan


def chunk_plan(root, plan, maximum):
    groups = OrderedDict()
    cases = {case['case_id'] for case in plan['cases']}
    for tid, task in enumerate(plan['tasks']):
        require(task['task_id'] == tid and task['case_id'] in cases, 'Task table identity invalid')
        groups.setdefault(task['case_id'], []).append(tid)
    require(set(groups) == cases, 'Each planned case must have tasks')
    chunks, current, case_ids = [], [], []
    def flush():
        if current:
            index = len(chunks)
            filename = f'campaign-chunk-{index:04d}.tsv'
            require(index <= 9999, 'Too many campaign chunks')
            payload = ''.join(f'{tid}\n' for tid in current).encode()
            immutable(root / filename, payload)
            chunks.append(dict(chunk_id=index, case_ids=list(case_ids), task_ids=list(current),
                               n_tasks=len(current), task_file=filename, task_file_sha256=sha(root / filename)))
            current.clear(); case_ids.clear()
    for cid, ids in groups.items():
        require(len(ids) <= maximum, f'Case {cid} exceeds --max-tasks-per-chunk; increase the limit')
        if len(current) + len(ids) > maximum:
            flush()
        current.extend(ids); case_ids.append(cid)
    flush()
    value = dict(schema_version=1, root=str(root), plan_sha256=sha(root / 'plan.json'),
                 source_manifest_sha256=sha(root / 'source-manifest.json'), max_tasks_per_chunk=maximum,
                 n_cases=plan['n_cases'], n_tasks=plan['n_tasks'], chunks=chunks)
    immutable(root / 'campaign/chunks.json', canonical(value))
    return value


def ready(root, plan):
    path = root / 'preparation.json'
    if not path.exists():
        return False
    value = read(path)
    require(value.get('success') is True and value.get('status') == 'complete'
            and value.get('plan_sha256') == sha(root / 'plan.json')
            and value.get('source_manifest_sha256') == plan['source_manifest_sha256']
            and value.get('n_cases') == plan['n_cases'] and value.get('n_tasks') == plan['n_tasks'],
            'Existing preparation marker is incomplete or has a different identity')
    return True


def completed_task(root, plan, tid):
    folder = root / 'tasks' / f'{tid:05d}'
    result_path, status_path = folder / 'result.json', folder / 'status.json'
    if not result_path.exists():
        return False
    result = read(result_path)
    require(result.get('task') == plan['tasks'][tid] and result.get('plan_fingerprint') == plan['plan_fingerprint']
            and result.get('source_manifest_sha256') == plan['source_manifest_sha256'],
            f'Existing task identity differs: {tid}')
    if result.get('execution_success') is not True:
        return False
    require(status_path.is_file(), f'Completed task is missing status: {tid}')
    status = read(status_path)
    require(status.get('result_sha256') == sha(result_path) and status.get('status') == 'finished'
            and result.get('status') == 'complete', f'Completed task status/hash invalid: {tid}')
    for key in ('task', 'case', 'plan_fingerprint', 'source_manifest_sha256', 'success', 'execution_success', 'fit_status'):
        require(result.get(key) == status.get(key), f'Completed task status identity differs: {tid}/{key}')
    for filename in ('process-exit-code.txt', 'launcher-exit-code.txt'):
        require((folder / filename).read_text().strip() == '0', f'Completed task has nonzero/missing process status: {tid}')
    for name, expected in result.get('files', {}).items():
        part = Path(name); path = (folder / part).resolve()
        require(not part.is_absolute() and '..' not in part.parts and path.is_relative_to(folder.resolve())
                and sha(path) == expected, f'Completed task artifact changed: {tid}/{name}')
    # A scientific failure or strict initializer exclusion is a finished result,
    # not an implicit request to repeat the same computation.
    return True


def scheduler_state(job_id):
    queue = subprocess.run(['squeue', '--noheader', '--jobs', job_id, '--format=%T'], capture_output=True, text=True)
    require(queue.returncode == 0, f'Cannot inspect active job {job_id}: {queue.stderr.strip()}')
    states = [row.strip() for row in queue.stdout.splitlines() if row.strip()]
    if states:
        return 'active'
    accounting = subprocess.run(['sacct', '--noheader', '--allocations', '--jobs', job_id,
                                  '--format=JobIDRaw,State', '--parsable2'], capture_output=True, text=True)
    require(accounting.returncode == 0, f'Cannot inspect completed job {job_id}: {accounting.stderr.strip()}')
    for row in accounting.stdout.splitlines():
        fields = row.strip().split('|')
        if len(fields) >= 2 and fields[0] == job_id:
            state = fields[1].split()[0].rstrip('+')
            if state in TERMINAL:
                return 'terminal'
    raise ValueError(f'Job {job_id} is absent from the queue and has no conclusive accounting state; no duplicate submission is allowed')


def latest_attempt(root, prefix):
    paths = sorted((root / 'campaign/attempts').glob(prefix + '-attempt-*/submission.json'))
    if not paths:
        return None
    value = read(paths[-1])
    require(value.get('tag') == paths[-1].parent.name, 'Submission record tag does not match its directory')
    return value


def validate_attempt(root, previous, identity, *, kind, chunk=None):
    if previous is None:
        return
    require(all(previous.get(key) == value for key, value in identity.items()) and previous.get('kind') == kind,
            'Prior submission belongs to different frozen campaign inputs')
    require(type(previous.get('attempt')) is int and 1 <= previous['attempt'] <= 9999, 'Invalid prior attempt number')
    prefix = 'prepare' if kind == 'prepare' else f'chunk-{chunk["chunk_id"]:04d}'
    require(previous['tag'] == f'{prefix}-attempt-{previous["attempt"]:04d}', 'Prior attempt tag differs from identity')
    if kind == 'pool':
        name = 'campaign-' + previous['tag'] + '.tsv'
        ids = previous.get('task_ids')
        require(previous.get('chunk_id') == chunk['chunk_id'] and previous.get('task_file') == name,
                'Prior pool belongs to another chunk')
        require(isinstance(ids, list) and ids and len(ids) == len(set(ids))
                and set(ids).issubset(chunk['task_ids']), 'Prior submitted task IDs are not a valid chunk subset')
        path = root / name
        require(not path.is_symlink() and path.read_bytes() == ''.join(f'{tid}\n' for tid in ids).encode()
                and sha(path) == previous.get('task_file_sha256'), 'Prior submitted task list changed')


def attempt_number(previous):
    number = 1 if previous is None else previous['attempt'] + 1
    require(number <= 9999, 'Attempt limit exceeded')
    return number


def base_command(root, args, tag, prepare=False):
    directory = root / 'campaign/attempts' / tag
    command = ['sbatch', '--parsable', '--job-name=sv2-' + hashlib.sha256(str(root).encode()).hexdigest()[:8] + '-' + tag,
               '--account=' + args.account, '--partition=' + args.partition, '--cpus-per-task=1', '--export=ALL',
               '--chdir=' + str(root), '--output=' + str(directory / 'slurm-%j.out'), '--error=' + str(directory / 'slurm-%j.err')]
    if prepare:
        command += ['--ntasks=1', '--mem=' + args.prepare_mem, '--time=' + args.prepare_time]
    else:
        command += ['--ntasks=' + str(args.workers), '--mem-per-cpu=' + args.mem, '--time=' + args.time]
    return command


def submit(root, args, record):
    if args.dry_run:
        return record
    directory = root / 'campaign/attempts' / record['tag']
    directory.mkdir(exist_ok=False)
    path = directory / 'submission.json'
    record.update(status='submitting', submitted_utc=now(), job_id=None)
    atomic(path, record)  # Persist exact intent before a side-effecting scheduler call.
    reply = subprocess.run(record['command'], capture_output=True, text=True)
    (directory / 'submission.stdout').write_text(reply.stdout)
    (directory / 'submission.stderr').write_text(reply.stderr)
    if reply.returncode != 0 or not re.fullmatch(r'[0-9]+(?:;[^\s;]+)?', reply.stdout.strip()):
        record.update(status='submission_uncertain', returncode=reply.returncode)
        atomic(path, record)
        raise ValueError(f'Scheduler submission was rejected or ambiguous for {record["tag"]}; inspect its durable record before retrying')
    record.update(status='submitted', job_id=reply.stdout.strip().split(';')[0], returncode=0)
    atomic(path, record)
    return record


def existing_state(previous, args):
    if previous is None:
        return None
    require(previous.get('status') == 'submitted' and re.fullmatch('[0-9]+', previous.get('job_id') or ''),
            'Prior scheduler submission is ambiguous; reconcile its saved stdout/stderr before retrying')
    if args.dry_run:
        return 'not_queried'
    return scheduler_state(previous['job_id'])


def orchestrate(root, args):
    plan = load_inputs(root)
    manifest = chunk_plan(root, plan, args.max_tasks_per_chunk)
    chosen = list(range(len(manifest['chunks']))) if args.chunks is None else args.chunks
    require(len(chosen) == len(set(chosen)) and all(0 <= i < len(manifest['chunks']) for i in chosen), 'Invalid/duplicate chunk IDs')
    output = dict(schema_version=1, run_dir=str(root), dry_run=args.dry_run, workers=args.workers,
                  n_cases=plan['n_cases'], n_tasks=plan['n_tasks'], chunks=len(manifest['chunks']), actions=[], recorded_utc=now())
    identity = dict(plan_sha256=sha(root / 'plan.json'), source_manifest_sha256=sha(root / 'source-manifest.json'),
                    chunk_plan_sha256=sha(root / 'campaign/chunks.json'))
    dependency = None
    if not ready(root, plan):
        previous = latest_attempt(root, 'prepare')
        validate_attempt(root, previous, identity, kind='prepare')
        state = existing_state(previous, args)
        if state in ('active', 'not_queried'):
            dependency = previous['job_id']
            output['actions'].append(dict(kind='prepare', action='reuse_active_or_unqueried', job_id=dependency))
        else:
            require(previous is None or args.resume, 'Previous preparation ended without readiness; use --resume after inspecting its logs')
            number = attempt_number(previous); tag = f'prepare-attempt-{number:04d}'
            command = base_command(root, args, tag, prepare=True)
            command += [str(root / 'source/hpc/discovery/sparse_smart_v2_campaign_prepare.sbatch'), str(root), tag]
            record = dict(identity, kind='prepare', tag=tag, attempt=number, command=command)
            record = submit(root, args, record)
            output['actions'].append(record)
            dependency = record.get('job_id', 'PREPARE_JOB_ID')
    else:
        output['actions'].append(dict(kind='prepare', action='already_ready'))
    if not args.prepare_only:
        for index in sorted(chosen):
            chunk = manifest['chunks'][index]
            previous = latest_attempt(root, f'chunk-{index:04d}')
            validate_attempt(root, previous, identity, kind='pool', chunk=chunk)
            state = existing_state(previous, args)
            if state in ('active', 'not_queried'):
                output['actions'].append(dict(kind='pool', chunk_id=index, action='active_or_unqueried_skip', job_id=previous['job_id']))
                continue
            pending = [tid for tid in chunk['task_ids'] if not completed_task(root, plan, tid)]
            if not pending:
                output['actions'].append(dict(kind='pool', chunk_id=index, action='already_finished', n_tasks=0))
                continue
            require(previous is None or args.resume, f'Chunk {index} ended with unfinished tasks; use --resume')
            number = attempt_number(previous); tag = f'chunk-{index:04d}-attempt-{number:04d}'
            filename = 'campaign-' + tag + '.tsv'
            immutable(root / filename, ''.join(f'{tid}\n' for tid in pending).encode())
            command = base_command(root, args, tag)
            if dependency:
                command.append('--dependency=afterok:' + dependency)
            command += [str(root / 'source/hpc/discovery/sparse_smart_v2_campaign_pool.sbatch'), str(root), str(args.workers), filename, tag]
            record = dict(identity, kind='pool', tag=tag, chunk_id=index, attempt=number,
                          task_file=filename, task_file_sha256=sha(root / filename), task_ids=pending,
                          n_tasks=len(pending), completed_tasks_skipped=chunk['n_tasks'] - len(pending), command=command)
            output['actions'].append(submit(root, args, record))
    atomic(root / ('campaign-submission-plan.json' if args.dry_run else 'campaign-submission-latest.json'), output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=100)
    parser.add_argument('--max-tasks-per-chunk', type=int, default=10000)
    parser.add_argument('--chunks', type=int, nargs='+')
    parser.add_argument('--time', default='12:00:00')
    parser.add_argument('--mem', default='4G')
    parser.add_argument('--prepare-time', default='01:00:00')
    parser.add_argument('--prepare-mem', default='16G')
    parser.add_argument('--account', default='mkolar_1314')
    parser.add_argument('--partition', default='main')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        require(args.run_dir.is_absolute() and args.run_dir.is_dir(), 'An existing absolute --run-dir is required')
        root = args.run_dir.resolve()
        require(args.dry_run or root.is_relative_to(RUN_PREFIX), 'Actual submission requires a resolved /scratch2 root')
        require(args.workers > 0 and args.max_tasks_per_chunk > 0, 'Workers and chunk size must be positive')
        for value in (args.time, args.prepare_time):
            require(re.fullmatch(r'([0-9]+-)?[0-9]+(:[0-9]{2}){0,2}', value), 'Invalid Slurm time')
        for value in (args.mem, args.prepare_mem):
            require(re.fullmatch(r'[1-9][0-9]*[KMGT]?', value), 'Invalid Slurm memory')
        for value in (args.account, args.partition):
            require(re.fullmatch(r'[A-Za-z0-9_.-]+', value), 'Invalid account/partition')
        (root / 'campaign/attempts').mkdir(parents=True, exist_ok=True)
        with (root / 'campaign/.submission.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            output = orchestrate(root, args)
        print(json.dumps(output, indent=2))
        for action in output['actions']:
            if 'command' in action:
                print(shlex.join(action['command']), file=sys.stderr)
        return 0
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
