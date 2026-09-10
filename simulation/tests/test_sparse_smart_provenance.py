"""Portable source identities and identity-payload checks on real resume paths."""

import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sparse_smart_provenance as provenance
import run_sparse_smart as fixed
import run_sparse_smart_tuned as tuned
import run_sparse_smart_external as external
import run_sparse_smart_budget_study as budget


RUNNERS = (fixed, tuned, external, budget)


def _generator_from(path, alias):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generator


def _source_tree(root):
    (root / "simulation").mkdir(parents=True)
    (root / "sparse_smart" / "nested").mkdir(parents=True)
    (root / "simulation" / "run_example.py").write_text("RUNNER_VERSION = 1\n")
    shutil.copyfile(provenance.__file__, root / "simulation" / "sparse_smart_provenance.py")
    (root / "sparse_smart" / "__init__.py").write_text("PACKAGE_VERSION = 1\n")
    (root / "sparse_smart" / "nested" / "math.py").write_text("VALUE = 1\n")
    (root / "generator.py").write_text("def generator(**kwargs):\n    return kwargs\n")
    api = SimpleNamespace(__file__=str(root / "sparse_smart" / "__init__.py"))
    generator = _generator_from(root / "generator.py", "alias_" + root.name)
    return api, generator


def test_identical_relocated_sources_have_same_digest_and_separate_locations(tmp_path, monkeypatch):
    first_root, second_root = tmp_path / "task0", tmp_path / "task1"
    first_api, first_generator = _source_tree(first_root)
    second_api, second_generator = _source_tree(second_root)
    values = []
    for root, api, generator in ((first_root, first_api, first_generator),
                                 (second_root, second_api, second_generator)):
        with monkeypatch.context() as context:
            context.setattr(provenance, "__file__", str(root / "simulation" / "sparse_smart_provenance.py"))
            values.append(provenance.implementation_provenance(
                api, generator, [root / "simulation" / "run_example.py"]))
    first, second = values
    assert first["implementation_fingerprint_scheme"] == provenance.FINGERPRINT_SCHEME
    assert first["implementation_fingerprint"] == second["implementation_fingerprint"]
    assert first["implementation_manifest"] == second["implementation_manifest"]
    assert first["implementation_source_locations"] != second["implementation_source_locations"]
    assert all(str(first_root) in path for path in first["implementation_source_locations"].values())
    assert "simulation/sparse_smart_provenance.py" in first["implementation_source_locations"]
    provenance.validate_resume_implementation(first, second, tmp_path / "record.json")

    (second_root / "sparse_smart" / "nested" / "math.py").write_text("VALUE = 2\n")
    changed = provenance.implementation_provenance(
        second_api, second_generator, [second_root / "simulation" / "run_example.py"])
    assert first["implementation_fingerprint"] != changed["implementation_fingerprint"]
    with pytest.raises(ValueError, match="Implementation differs"):
        provenance.validate_resume_implementation(first, changed, tmp_path / "record.json")


def test_generator_package_dependencies_are_hashed(tmp_path):
    package = tmp_path / "generator_package"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "data.py").write_text("def generator(**kwargs):\n    return kwargs\n")
    dependency = package / "helpers.py"
    dependency.write_text("VALUE = 1\n")
    generator = _generator_from(package / "data.py", "standalone_loader_alias")
    first = provenance.implementation_provenance(None, generator, [])
    assert "generator/generator_package/helpers.py" in first["implementation_source_locations"]
    dependency.write_text("VALUE = 2\n")
    second = provenance.implementation_provenance(None, generator, [])
    assert first["implementation_fingerprint"] != second["implementation_fingerprint"]


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda module: module.__name__)
def test_each_runner_fingerprint_is_portable_and_tracks_content(tmp_path, runner):
    api_a, generator_a = _source_tree(tmp_path / "task0")
    api_b, generator_b = _source_tree(tmp_path / "task1")
    first = runner._implementation_fingerprint(api_a, generator_a)
    assert first == runner._implementation_fingerprint(api_b, generator_b)
    (tmp_path / "task1" / "sparse_smart" / "__init__.py").write_text("PACKAGE_VERSION = 2\n")
    assert first != runner._implementation_fingerprint(api_b, generator_b)


def _data(**kwargs):
    n, p, q = kwargs["n"], kwargs["p"], kwargs["q"]
    return dict(X=np.zeros((n, p)), Y=np.zeros((n, q)), C0=np.eye(p, q), C_star=np.zeros((p, q)))


def _run(runner, destination, *, api=None):
    # Invalid fitted dimensions exercise serialization/resume without fitting.
    setting = fixed.SimulationSetting(6, 12, 10, .01, 5, 3, "inapplicable")
    return runner.run_setting(setting=setting, model="model1", experiment="exp3", seed_id=1,
        random_seed=123, destination=destination, generate_data_fn=_data, sparse_api=api)


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda module: module.__name__)
@pytest.mark.parametrize("field", ["configuration", "setting", "generator_arguments", "schema_version"])
def test_resume_rejects_changed_identity_payload_with_unchanged_digest(tmp_path, runner, field):
    destination = tmp_path / "result.json"
    _, saved = _run(runner, destination)
    original_hash = saved["configuration_fingerprint"]
    if field == "configuration":
        saved[field]["margins"]["d_lower"] += .001
    elif field == "setting":
        saved[field]["n"] += 1
    elif field == "generator_arguments":
        saved[field]["random_seed"] += 1
    else:
        saved[field] += 1
    assert saved["configuration_fingerprint"] == original_hash
    destination.write_text(json.dumps(saved))
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="configuration differs"):
        _run(runner, destination)
    assert destination.read_bytes() == before


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda module: module.__name__)
def test_resume_refuses_legacy_fingerprint_scheme_without_rewriting(tmp_path, runner):
    destination = tmp_path / "result.json"
    _, saved = _run(runner, destination)
    del saved["implementation_fingerprint_scheme"]
    destination.write_text(json.dumps(saved))
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="Legacy or unsupported implementation fingerprint scheme"):
        _run(runner, destination)
    assert destination.read_bytes() == before


@pytest.mark.parametrize("runner", RUNNERS, ids=lambda module: module.__name__)
def test_resume_after_identical_package_relocation_preserves_original_provenance(tmp_path, runner):
    api_a, _ = _source_tree(tmp_path / "task0")
    api_b, _ = _source_tree(tmp_path / "task1")
    destination = tmp_path / "result.json"
    _, first = _run(runner, destination, api=api_a)
    before = destination.read_bytes()
    action, resumed = _run(runner, destination, api=api_b)
    assert action == "skipped" and resumed == first
    assert destination.read_bytes() == before
    assert str(tmp_path / "task0") in resumed["implementation_source_locations"]["sparse_smart/__init__.py"]


def test_nonstandard_nonfinite_identity_is_not_normalized_into_a_valid_null(tmp_path):
    destination = tmp_path / "result.json"
    _, saved = _run(fixed, destination)
    assert saved["configuration"]["runner"]["support_limit"] is None
    saved["configuration"]["runner"]["support_limit"] = float("nan")
    destination.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="configuration differs"):
        _run(fixed, destination)
