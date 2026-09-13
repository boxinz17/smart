"""Scheduler reconciliation checks: no scheduler jobs or scientific fits."""
import importlib.util
from pathlib import Path
import subprocess

import pytest


SOURCE = Path(__file__).resolve().parents[2] / 'hpc/discovery/submit_sparse_smart_v2_campaign.py'
SPEC = importlib.util.spec_from_file_location('scheduler_lookup_under_test', SOURCE)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
JOB = '11958580'
MISSING = 'slurm_load_jobs error: Invalid job id specified'


def responses(monkeypatch, *, queue=(0, '', ''), accounting=(0, f'{JOB}|COMPLETED|\n', '')):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs == {'capture_output': True, 'text': True}
        assert command[command.index('--jobs') + 1] == JOB
        if command[0] == 'squeue':
            reply = queue
        else:
            assert command[0] == 'sacct'
            assert '--allocations' in command
            assert '--format=JobIDRaw,State' in command
            reply = accounting
        return subprocess.CompletedProcess(command, *reply)

    monkeypatch.setattr(launcher.subprocess, 'run', run)
    return calls


@pytest.mark.parametrize('prefix', ['', 'squeue: error: '])
def test_discovery_missing_queue_id_is_reconciled_against_terminal_accounting(monkeypatch, prefix):
    calls = responses(monkeypatch, queue=(1, '', prefix + MISSING + '\n'))
    assert launcher.scheduler_state(JOB) == 'terminal'
    assert [command[0] for command in calls] == ['squeue', 'sacct']


@pytest.mark.parametrize('state', ['RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING'])
def test_active_queue_jobs_do_not_consult_potentially_stale_accounting(monkeypatch, state):
    calls = responses(monkeypatch, queue=(0, state + '\n', ''))
    assert launcher.scheduler_state(JOB) == 'active'
    assert len(calls) == 1


@pytest.mark.parametrize('state', sorted(launcher.TERMINAL) + ['CANCELLED by 1234', 'COMPLETED+'])
def test_empty_successful_queue_needs_terminal_exact_allocation(monkeypatch, state):
    responses(monkeypatch, accounting=(0, f'{JOB}|{state}|\n', ''))
    assert launcher.scheduler_state(JOB) == 'terminal'


@pytest.mark.parametrize('queue', [
    (1, '', 'slurm_load_jobs error: Unable to contact slurm controller (connect failure)'),
    (1, '', 'slurm_load_jobs error: Protocol authentication error'),
    (1, '', 'squeue: error: Access/permission denied'),
    (1, '', ''),
    (1, 'RUNNING\n', MISSING),
    (1, '', MISSING + '\nAdditional scheduler failure'),
    (1, '', 'Invalid job id specified'),
])
def test_unrelated_or_ambiguous_queue_errors_block_without_accounting(monkeypatch, queue):
    calls = responses(monkeypatch, queue=queue)
    with pytest.raises(ValueError, match='Cannot inspect active job'):
        launcher.scheduler_state(JOB)
    assert len(calls) == 1


@pytest.mark.parametrize('output', [
    '',
    f'{JOB}|RUNNING|\n',
    f'{JOB}|PENDING|\n',
    f'{JOB}||\n',
    f'{JOB}|UNKNOWN|\n',
    f'{JOB}.batch|COMPLETED|\n',
    '11958581|COMPLETED|\n',
    f'{JOB}_1|COMPLETED|\n',
    f'{JOB}|COMPLETED|\n{JOB}|RUNNING|\n',
    f'{JOB}|COMPLETED|\n{JOB}|COMPLETED|\n',
])
def test_missing_queue_id_without_unique_terminal_accounting_blocks_duplicates(monkeypatch, output):
    calls = responses(monkeypatch, queue=(1, '', MISSING), accounting=(0, output, ''))
    with pytest.raises(ValueError, match='no conclusive accounting state; no duplicate submission is allowed'):
        launcher.scheduler_state(JOB)
    assert len(calls) == 2


def test_accounting_failure_blocks_even_when_stdout_contains_terminal_record(monkeypatch):
    responses(monkeypatch, queue=(1, '', MISSING),
              accounting=(1, f'{JOB}|COMPLETED|\n', 'Database connection failure'))
    with pytest.raises(ValueError, match='Cannot inspect completed job'):
        launcher.scheduler_state(JOB)


def test_nonmatching_accounting_rows_do_not_override_exact_allocation(monkeypatch):
    responses(monkeypatch, queue=(1, '', MISSING),
              accounting=(0, f'{JOB}.batch|FAILED|\n99999999|RUNNING|\n{JOB}|COMPLETED|\n', ''))
    assert launcher.scheduler_state(JOB) == 'terminal'
