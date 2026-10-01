"""The transfer builder freezes current source without copying old fit data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

CODE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("_cheap_snapshot_tests",
    CODE / "hpc/discovery/build_sparse_smart_v2_cheap_bic_snapshot.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_snapshot_copies_current_bytes_and_freezes_remote_identity(tmp_path, monkeypatch):
    code = tmp_path / "repo"
    code.mkdir()
    (code / "module.py").write_text("CURRENT = True\n")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(dict(schema_version=1, files={"module.py": "0" * 64})))
    reference = tmp_path / "old-plan.json"
    reference.write_text(json.dumps(dict(root=builder.EXHAUSTIVE_ROOT, reference_root="/scratch2/old-paper")))
    monkeypatch.setattr(builder, "CODE", code)
    monkeypatch.setattr(builder, "source_paths", lambda manifest: ["module.py"])
    monkeypatch.setattr(builder, "git_state", lambda: dict(commit="abc", status_porcelain=" M module.py\n", diff_sha256="d" * 64))
    frozen_split = dict(split_fingerprint="f" * 64, development_seed_ids=list(range(5)))
    monkeypatch.setattr(builder, "split_manifest", lambda plan: frozen_split)
    local, remote = tmp_path / "pilot", Path("/scratch2/test-pilot")
    output = builder.build(local, remote, baseline_manifest=baseline, reference_plan_path=reference)
    assert (local / "source/module.py").read_text() == "CURRENT = True\n"
    manifest = json.loads((local / "source-manifest.json").read_text())
    assert manifest["source_root"] == "/scratch2/test-pilot/source"
    assert manifest["files"]["module.py"] == hashlib.sha256(b"CURRENT = True\n").hexdigest()
    assert json.loads((local / "split-manifest.json").read_text()) == frozen_split
    config = json.loads((local / "pilot.json").read_text())
    assert config["accepted_update_cap"] == 200 and config["workers"] == 100
    assert config["pool_time"] == "06:00:00"
    assert config["exhaustive_plan_sha256"] == builder.sha(reference)
    with tarfile.open(output["archive"], "r:gz") as archive:
        names = set(archive.getnames())
        assert "source/module.py" in names and "pilot.json" in names
        assert not any(name.endswith((".npz", ".npy")) or name.startswith("/") for name in names)
    with pytest.raises(ValueError, match="fresh"):
        builder.build(local, remote, baseline_manifest=baseline, reference_plan_path=reference)


def test_snapshot_rejects_existing_or_non_scratch_remote_before_reading_inputs(tmp_path):
    with pytest.raises(ValueError, match="below /scratch2"):
        builder.build(tmp_path / "pilot", Path("/scratch1/pilot"))
    (tmp_path / "pilot.tar.gz").write_bytes(b"keep")
    with pytest.raises(ValueError, match="fresh"):
        builder.build(tmp_path / "pilot", Path("/scratch2/pilot"))
    assert (tmp_path / "pilot.tar.gz").read_bytes() == b"keep"


def test_source_inventory_rejects_raw_results_and_missing_current_files(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "REQUIRED_ADDITIONS", ())
    source = tmp_path / "result/data.npz"
    source.parent.mkdir()
    source.write_bytes(b"raw")
    with pytest.raises(ValueError, match="raw result/data"):
        builder.source_paths(dict(schema_version=1, files={"result/data.npz": "a" * 64}), code=tmp_path)
    with pytest.raises(ValueError, match="missing"):
        builder.source_paths(dict(schema_version=1, files={"missing.py": "a" * 64}), code=tmp_path)
