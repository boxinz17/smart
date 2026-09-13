"""Campaign orchestration fixtures: no real scheduler jobs or scientific fits."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

CODE = Path(__file__).resolve().parents[2]
HPC = CODE / 'hpc/discovery'


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = module_from(Path(__file__).with_name('test_sparse_smart_v2_launcher.py'), 'legacy_launcher_fixtures')
legacy_campaign = legacy.campaign
launcher = module_from(HPC / 'submit_sparse_smart_v2_campaign.py', 'campaign_submission')


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def refresh(root):
    files = {str(path.relative_to(root / 'source')): legacy.digest(path)
             for path in (root / 'source').rglob('*') if path.is_file()}
    json_write(root / 'source-manifest.json', dict(schema_version=1, source_root=str(root / 'source'), files=files))
    plan = json.loads((root / 'plan.json').read_text())
    plan.pop('plan_fingerprint', None)
    plan['source_manifest_sha256'] = legacy.digest(root / 'source-manifest.json')
    plan['plan_fingerprint'] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    json_write(root / 'plan.json', plan)
    legacy.marker(root)
    return plan


@pytest.fixture
def campaign(legacy_campaign, monkeypatch, tmp_path):
    root, env = legacy_campaign
    for name in ('submit_sparse_smart_v2_campaign.py', 'sparse_smart_v2_campaign_pool.sbatch', 'sparse_smart_v2_campaign_prepare.sbatch'):
        shutil.copyfile(HPC / name, root / 'source/hpc/discovery' / name)
    plan = dict(root=str(root), n_cases=2, n_tasks=6, cases=[{'case_id': 'a'}, {'case_id': 'b'}],
                tasks=[{'task_id': i, 'case_id': 'a' if i < 3 else 'b'} for i in range(6)])
    json_write(root / 'plan.json', plan)
    (root / 'work-items.tsv').write_text('0\n1\n2\n3\n4\n5\n')
    refresh(root)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(launcher, 'RUN_PREFIX', tmp_path)
    return root, env


def arguments(root, *extra):
    return ['--run-dir', str(root), '--max-tasks-per-chunk', '3', *extra]


def journal(root):
    return [json.loads(path.read_text()) for path in sorted((root / 'campaign/attempts').glob('*/submission.json'))]


def install_scheduler(env, monkeypatch, tmp_path):
    calls = tmp_path / 'sbatch-calls.jsonl'
    monkeypatch.setenv('FAKE_SUBMISSION_LOG', str(calls))
    legacy.executable(tmp_path / 'bin/sbatch', '''
import json,os,sys
from pathlib import Path
p=Path(os.environ['FAKE_SUBMISSION_LOG']);rows=p.read_text().splitlines() if p.exists() else []
with p.open('a') as h:h.write(json.dumps(sys.argv[1:])+'\\n')
if os.environ.get('FAKE_SUBMISSION_AMBIGUOUS')=='1':
 print('unrecognized reply');raise SystemExit(0)
print(str(9000+len(rows)))
''')
    legacy.executable(tmp_path / 'bin/squeue', '''
import os
if os.environ.get('FAKE_QUEUE_STATE'):print(os.environ['FAKE_QUEUE_STATE'])
''')
    legacy.executable(tmp_path / 'bin/sacct', '''
import os,sys
print(sys.argv[sys.argv.index('--jobs')+1]+'|'+os.environ.get('FAKE_ACCOUNT_STATE','COMPLETED')+'|')
''')
    return calls


def finish_task(root, tid, *, scientific_success=True):
    plan = json.loads((root / 'plan.json').read_text())
    folder = root / 'tasks' / f'{tid:05d}'
    result = dict(task=plan['tasks'][tid], case=plan['cases'][0 if tid < 3 else 1],
        plan_fingerprint=plan['plan_fingerprint'], source_manifest_sha256=plan['source_manifest_sha256'],
        status='complete', execution_success=True, success=scientific_success,
        fit_status='completed' if scientific_success else 'initialization_spectrum_failed', files={})
    json_write(folder / 'result.json', result)
    status = dict(result, status='finished', result_sha256=legacy.digest(folder / 'result.json'))
    json_write(folder / 'status.json', status)
    for name in ('process-exit-code.txt', 'launcher-exit-code.txt'):
        (folder / name).write_text('0\n')


def test_dry_run_chunks_are_case_aligned_and_independent(campaign, capsys):
    root, env = campaign
    (root / 'preparation.json').unlink()
    assert launcher.main(arguments(root, '--dry-run')) == 0
    output = json.loads((root / 'campaign-submission-plan.json').read_text())
    prep, *pools = output['actions']
    assert len(pools) == 2
    assert '--ntasks=1' in prep['command'] and '--mem=16G' in prep['command']
    assert '--time=01:00:00' in prep['command']
    assert all('--dependency=afterok:PREPARE_JOB_ID' in pool['command'] for pool in pools)
    assert all('--ntasks=100' in pool['command'] and '--cpus-per-task=1' in pool['command'] for pool in pools)
    assert not any(arg.startswith(('--array', '--nodes', '--ntasks-per-node')) for pool in pools for arg in pool['command'])
    chunks = json.loads((root / 'campaign/chunks.json').read_text())['chunks']
    assert [c['case_ids'] for c in chunks] == [['a'], ['b']]
    assert [c['task_ids'] for c in chunks] == [[0, 1, 2], [3, 4, 5]]
    assert not list((root / 'campaign/attempts').iterdir())
    assert not Path(env['FAKE_SCHEDULER_CALLED']).exists()
    assert str(root) in prep['command']  # literal spaces, dollar and braces survived
    assert (root / 'work-items.tsv').read_text() == '0\n1\n2\n3\n4\n5\n'


def test_actual_submission_records_independent_chunks_and_live_resume_skips(campaign, monkeypatch, tmp_path):
    root, env = campaign
    (root / 'preparation.json').unlink()
    calls = install_scheduler(env, monkeypatch, tmp_path)
    assert launcher.main(arguments(root)) == 0
    rows = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(rows) == 3
    assert all('--dependency=afterok:9000' in command for command in rows[1:])
    assert len({r['tag'] for r in journal(root)}) == 3
    monkeypatch.setenv('FAKE_QUEUE_STATE', 'RUNNING')
    assert launcher.main(arguments(root, '--resume')) == 0
    assert len(calls.read_text().splitlines()) == 3
    assert all(r['status'] == 'submitted' for r in journal(root))


def test_resume_skips_verified_finished_failures_and_preserves_attempt_files(campaign, monkeypatch, tmp_path):
    root, env = campaign
    calls = install_scheduler(env, monkeypatch, tmp_path)
    assert launcher.main(arguments(root, '--chunks', '0')) == 0
    first = journal(root)[0]
    finish_task(root, 0)
    finish_task(root, 1, scientific_success=False)
    assert launcher.main(arguments(root, '--chunks', '0', '--resume')) == 0
    rows = journal(root)
    assert len(rows) == 2 and rows[0] == first
    assert rows[1]['task_ids'] == [2] and rows[1]['completed_tasks_skipped'] == 2
    assert (root / rows[0]['task_file']).read_text() == '0\n1\n2\n'
    assert (root / rows[1]['task_file']).read_text() == '2\n'
    assert len(calls.read_text().splitlines()) == 2
    finish_task(root, 2)
    assert launcher.main(arguments(root, '--chunks', '0', '--resume')) == 0
    assert len(calls.read_text().splitlines()) == 2


def test_uncertain_submission_blocks_repeated_work(campaign, monkeypatch, tmp_path):
    root, env = campaign
    calls = install_scheduler(env, monkeypatch, tmp_path)
    monkeypatch.setenv('FAKE_SUBMISSION_AMBIGUOUS', '1')
    assert launcher.main(arguments(root, '--chunks', '0')) == 2
    assert journal(root)[0]['status'] == 'submission_uncertain'
    monkeypatch.delenv('FAKE_SUBMISSION_AMBIGUOUS')
    assert launcher.main(arguments(root, '--chunks', '0', '--resume')) == 2
    assert len(calls.read_text().splitlines()) == 1


def test_unknown_accounting_state_blocks_resume(campaign, monkeypatch, tmp_path):
    root, env = campaign
    calls = install_scheduler(env, monkeypatch, tmp_path)
    assert launcher.main(arguments(root, '--chunks', '0')) == 0
    monkeypatch.setenv('FAKE_ACCOUNT_STATE', 'UNKNOWN')
    assert launcher.main(arguments(root, '--chunks', '0', '--resume')) == 2
    assert len(calls.read_text().splitlines()) == 1


@pytest.mark.parametrize('change', ['source', 'chunk', 'finished_hash', 'finished_process'])
def test_changed_identity_or_corrupt_finished_work_is_not_silently_repeated(campaign, change):
    root, env = campaign
    assert launcher.main(arguments(root, '--dry-run')) == 0
    if change == 'source':
        with (root / 'source/hpc/discovery/env.sh').open('a') as stream:stream.write('\n# changed\n')
    elif change == 'chunk':
        (root / 'campaign-chunk-0000.tsv').chmod(0o644)
        (root / 'campaign-chunk-0000.tsv').write_text('0\n')
    else:
        finish_task(root, 0)
        if change == 'finished_hash':
            with (root / 'tasks/00000/result.json').open('a') as stream:stream.write(' ')
        else:
            (root / 'tasks/00000/process-exit-code.txt').write_text('7\n')
    assert launcher.main(arguments(root, '--dry-run')) == 2
    assert not Path(env['FAKE_SCHEDULER_CALLED']).exists()


def test_campaign_pool_uses_isolated_logs_and_exact_subset_record(campaign):
    root, env = campaign
    assert launcher.main(arguments(root, '--dry-run', '--chunks', '0')) == 0
    action = next(a for a in json.loads((root / 'campaign-submission-plan.json').read_text())['actions'] if a['kind'] == 'pool')
    attempt = root / 'campaign/attempts' / action['tag'];attempt.mkdir()
    json_write(attempt / 'submission.json', action)
    env.update(SLURM_JOB_ID='123', SLURM_NTASKS='100', SLURM_CPUS_PER_TASK='1')
    result = legacy.run_script(root, env, 'sparse_smart_v2_campaign_pool.sbatch', root, 100, action['task_file'], action['tag'])
    assert result.returncode == 0, result.stderr
    assert (attempt / 'pool-exit-code.txt').read_text().strip() == '0'
    assert (attempt / 'parallel-joblog.tsv').is_file()
    assert not (root / 'pool-exit-code.txt').exists()
    assert not (root / 'logs/parallel-joblog.tsv').exists()
    before = json.loads((attempt / 'source-integrity-before.json').read_text())
    assert before['task_file_sha256'] == action['task_file_sha256'] and before['n_selected_tasks'] == 3
    calls = [json.loads(row) for row in (root / 'logs/runner-calls.jsonl').read_text().splitlines()]
    assert [row['task'] for row in calls] == [0, 1, 2]


def test_campaign_pool_rejects_modified_submitted_subset(campaign):
    root, env = campaign
    assert launcher.main(arguments(root, '--dry-run', '--chunks', '0')) == 0
    action = next(a for a in json.loads((root / 'campaign-submission-plan.json').read_text())['actions'] if a['kind'] == 'pool')
    attempt = root / 'campaign/attempts' / action['tag'];attempt.mkdir()
    json_write(attempt / 'submission.json', action)
    (root / action['task_file']).chmod(0o644);(root / action['task_file']).write_text('0\n2\n')
    env.update(SLURM_JOB_ID='123', SLURM_NTASKS='100', SLURM_CPUS_PER_TASK='1')
    result = legacy.run_script(root, env, 'sparse_smart_v2_campaign_pool.sbatch', root, 100, action['task_file'], action['tag'])
    assert result.returncode != 0 and 'exact submitted record' in result.stderr
    assert not Path(env['FAKE_PARALLEL_LOG']).exists()


def test_campaign_preparation_checks_exact_record_and_auditor_tests(campaign, tmp_path):
    root, env = campaign
    (root / 'preparation.json').unlink()
    assert launcher.main(arguments(root, '--dry-run', '--prepare-only')) == 0
    action = json.loads((root / 'campaign-submission-plan.json').read_text())['actions'][0]
    attempt = root / 'campaign/attempts' / action['tag'];attempt.mkdir()
    json_write(attempt / 'submission.json', action)
    commands = tmp_path / 'prepare-calls.jsonl'
    venv = tmp_path / 'prepare-venv'
    legacy.executable(venv / 'bin/python', f'''
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with Path(os.environ['FAKE_PREPARE_CALLS']).open('a') as stream:stream.write(json.dumps(args)+'\\n')
if args[0]=='-c' or args[:2]==['-m','pytest']:raise SystemExit(0)
os.execv({sys.executable!r},[{sys.executable!r}]+args)
''')
    env.update(VENV=str(venv), FAKE_PREPARE_CALLS=str(commands), SLURM_JOB_ID='122',
               SLURM_NTASKS='1', SLURM_CPUS_PER_TASK='1')
    result = legacy.run_script(root, env, 'sparse_smart_v2_campaign_prepare.sbatch', root, action['tag'])
    assert result.returncode == 0, result.stderr
    calls = [json.loads(row) for row in commands.read_text().splitlines()]
    assert calls[0][0] == '-'
    tests = next(row for row in calls if row[:2] == ['-m', 'pytest'])
    assert str(root / 'source/simulation/tests/test_sparse_smart_v2_audit.py') in tests
    assert str(root / 'source/simulation/tests/test_sparse_smart_v2_campaign_launcher.py') in tests
    assert '--import-mode=importlib' in tests and 'no:cacheprovider' in tests
    assert (attempt / 'preparation-exit-code.txt').read_text().strip() == '0'
    assert not (root / 'preparation-exit-code.txt').exists()
    assert json.loads((root / 'preparation.json').read_text())['success'] is True


def test_prior_attempt_identity_is_checked_before_resume(campaign, monkeypatch, tmp_path):
    root, env = campaign
    calls = install_scheduler(env, monkeypatch, tmp_path)
    assert launcher.main(arguments(root, '--chunks', '0')) == 0
    path = next((root / 'campaign/attempts').glob('*/submission.json'))
    record = json.loads(path.read_text());record['plan_sha256'] = 'different'
    json_write(path, record)
    assert launcher.main(arguments(root, '--chunks', '0', '--resume')) == 2
    assert len(calls.read_text().splitlines()) == 1


@pytest.mark.parametrize('extra', [('--workers', '0'), ('--max-tasks-per-chunk', '2'), ('--mem=-1G',),
                                   ('--prepare-time', 'bad'), ('--chunks', '2'), ('--chunks', '0', '0')])
def test_invalid_operational_options_rejected(campaign, extra):
    root, env = campaign
    assert launcher.main(arguments(root, '--dry-run', *extra)) != 0
    assert not Path(env['FAKE_SCHEDULER_CALLED']).exists()


def test_prepare_only_and_default_storage_guard(campaign, monkeypatch):
    root, env = campaign
    (root / 'preparation.json').unlink()
    assert launcher.main(arguments(root, '--dry-run', '--prepare-only')) == 0
    actions = json.loads((root / 'campaign-submission-plan.json').read_text())['actions']
    assert len(actions) == 1 and actions[0]['kind'] == 'prepare'
    monkeypatch.setattr(launcher, 'RUN_PREFIX', Path('/scratch2'))
    if not root.is_relative_to('/scratch2'):
        assert launcher.main(arguments(root)) == 2
    assert not Path(env['FAKE_SCHEDULER_CALLED']).exists()


def test_new_scripts_have_no_other_storage_installation_or_aggregation_actions():
    names = ('submit_sparse_smart_v2_campaign.py', 'sparse_smart_v2_campaign_pool.sbatch', 'sparse_smart_v2_campaign_prepare.sbatch')
    text = '\n'.join((HPC / name).read_text() for name in names)
    assert '/' + 'scratch' + '1' not in text
    assert '--array' not in text and 'scancel' not in text and 'pip install' not in text
    assert 'rsync' not in text and 'tar ' not in text
