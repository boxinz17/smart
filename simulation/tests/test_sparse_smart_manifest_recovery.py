"""Failed reruns preserve completed provenance; full profiles retain inapplicable cells."""
from concurrent.futures import Future
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_manifest
import run_sparse_smart_external_grid as external
import run_sparse_smart_repairs as repairs
import run_sparse_smart_budget_study as budget
import summarize_sparse_smart_budget_study as summary
import test_summarize_sparse_smart_budget_study as fixtures


class InlinePool:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)
        return future


def options(module, root, monkeypatch):
    if module is external:
        return ('expanded_pilot_manifest.json', 'fit_cell',
                ['--models', '0', '--experiments', '3', '--profile', 'difficult',
                 '--seed-count', '1', '--workers', '1', '--iterations', '1', '--output-root', str(root)])
    if module is repairs:
        baseline = root.parent / 'baseline'
        monkeypatch.setattr(repairs, 'selected_records', lambda path: [(baseline / 'old.json', 'control')])
        return ('repair_manifest.json', 'fit_record',
                ['--baseline-root', str(baseline), '--output-root', str(root), '--workers', '1'])
    return ('budget_study_manifest.json', 'fit_cell',
            ['--models', '0', '--experiments', '3', '--seed-ids', '0', '--workers', '1',
             '--iteration-budgets', '1', '--checkpoint-interval', '1', '--output-root', str(root)])


@pytest.mark.parametrize('module', [external, repairs, budget])
def test_failed_rerun_preserves_completed_manifest_and_saves_attempt(tmp_path, monkeypatch, module):
    root = tmp_path / 'study'
    name, worker, args = options(module, root, monkeypatch)
    monkeypatch.setattr(module, 'ProcessPoolExecutor', InlinePool)
    monkeypatch.setattr(module, worker, lambda task: dict(status='complete', outcome='written'))
    assert module.main(args) == 0
    canonical = root / name
    previous = canonical.read_bytes()
    previous_attempts = {path: path.read_bytes() for path in root.glob('*_attempts/*.json')}

    def fail(task):
        raise ValueError('Implementation differs from existing result')

    monkeypatch.setattr(module, worker, fail)
    assert module.main(args) == 1
    assert canonical.read_bytes() == previous
    for path, content in previous_attempts.items():
        assert path.read_bytes() == content
    new_paths = set(root.glob('*_attempts/*.json')) - previous_attempts.keys()
    assert len(new_paths) == 1
    failed = json.loads(next(iter(new_paths)).read_text())
    assert failed['attempt_status'] == 'failed' and failed['finished']
    assert failed['errors'][0]['exception'] == 'ValueError'


def test_worker_startup_error_and_interrupt_keep_attempt_progress(tmp_path):
    canonical = tmp_path / 'study_manifest.json'
    canonical.write_bytes(b'{"completed":true}\n')
    for error in (RuntimeError('worker startup failed'), KeyboardInterrupt()):
        value = dict(cells=[{'already_finished': True}], errors=[])
        with pytest.raises(type(error)):
            with batch_manifest.BatchManifest(canonical, value):
                raise error
        assert canonical.read_bytes() == b'{"completed":true}\n'
        attempt = json.loads(Path(value['attempt_manifest']).read_text())
        assert attempt['cells'] == [{'already_finished': True}]
        assert attempt['driver_error']['exception'] == type(error).__name__
        assert attempt['attempt_status'] == ('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')


def test_first_failed_budget_attempt_is_resumable_and_summarizable(tmp_path, monkeypatch):
    root = tmp_path / 'study'
    name, worker, args = options(budget, root, monkeypatch)

    def fail(task):
        raise ValueError('temporary worker failure')

    monkeypatch.setattr(budget, worker, fail)
    assert budget.main(args) == 1
    assert not (root / name).exists()
    report = summary.summarize(root, manifest_scope=True, generate_data_fn=fixtures.generator)
    assert report['expected_cells'] == report['missing_cells'] == 1
    assert report['manifest']['errors'][0]['message'] == 'temporary worker failure'
    monkeypatch.setattr(budget, worker, lambda task: dict(status='complete', outcome='written'))
    assert budget.main(args) == 0
    assert json.loads((root / name).read_text())['attempt_status'] == 'completed'


def test_full_profile_inapplicable_no_fit_run_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(budget.external_runner.old_runner, '_load_generator', lambda: fixtures.generator)

    def forbidden():
        raise AssertionError('Inapplicable cell loaded the estimator')

    monkeypatch.setattr(budget.external_runner.old_runner, '_load_sparse_api', forbidden)
    root = tmp_path / 'study'
    assert budget.main(['--models', '0', '--experiments', '2', '--profile', 'full',
        '--setting-index', '0', '--seed-ids', '0', '--workers', '1', '--iteration-budgets', '1', '2',
        '--checkpoint-interval', '1', '--output-root', str(root)]) == 0
    report = summary.summarize(root, manifest_scope=True, generate_data_fn=fixtures.generator)
    assert report['expected_cells'] == report['recorded_cells'] == report['recorded_inapplicable_cells'] == 1
    assert report['missing_cells'] == report['expected_applicable_cells'] == 0
    assert report['expected_inapplicable_cells'] == 1
    assert report['cells'][0]['status'] == 'inapplicable' and report['cells'][0]['caps'] == []
    assert report['per_model'][0]['transitions'] == report['per_setting'][0]['transitions'] == []
    assert 'not failed fits or unresolved optimization caps' in summary.write_outputs(report, root / 'summary').read_text()


def test_full_profile_audits_inapplicable_records_but_excludes_them_from_gain_denominators(tmp_path):
    value = fixtures.fixture()
    fixtures.write(tmp_path, value)
    manifest = fixtures.manifest(tmp_path, value)
    manifest.update(profile='full', setting_index=None, expected_cells=7,
                    expected_applicable=5, expected_inapplicable=2)
    (tmp_path / 'budget_study_manifest.json').write_text(json.dumps(manifest))
    config = budget.RunnerConfig(**value['configuration']['runner'])
    for setting in budget.experiment_settings(0, 2)[:2]:
        path = budget.result_path(tmp_path, model='model1', experiment='exp3', setting=setting, seed_id=0)
        budget.run_setting(setting=setting, model='model1', experiment='exp3', seed_id=0,
            random_seed=value['random_seed'], destination=path, config=config, generate_data_fn=fixtures.generator)
    report = summary.summarize(tmp_path, manifest_scope=True, generate_data_fn=fixtures.generator)
    assert report['expected_cells'] == 7 and report['recorded_cells'] == 3
    assert report['missing_cells'] == 4 and report['recorded_inapplicable_cells'] == 2
    assert all(t['expected_cell_pairs'] == 5 for t in report['per_model'][0]['transitions'])
    assert all(t['unresolved_or_missing_cell_pairs'] == 4 for t in report['per_model'][0]['transitions'])
    bad_path = budget.result_path(tmp_path, model='model1', experiment='exp3',
                                  setting=budget.experiment_settings(0, 2)[0], seed_id=0)
    bad = json.loads(bad_path.read_text())
    bad['evaluation_truth_fingerprint'] = '0' * 64
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        summary.summarize(tmp_path, manifest_scope=True, generate_data_fn=fixtures.generator)
