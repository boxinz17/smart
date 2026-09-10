"""Portable, verified paper aggregates without manuscript or PDF dependencies."""

from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_reference import DEFAULT_REFERENCE, read_reference, verify_reference


@pytest.fixture
def bundle(tmp_path):
    destination = tmp_path / "portable" / "paper_reference"
    destination.mkdir(parents=True)
    original = Path(DEFAULT_REFERENCE).parent
    for name in ("__init__.py", "v1_simulation_curves.csv", "provenance.json"):
        shutil.copyfile(original / name, destination / name)
    return destination


def metadata(bundle):
    return json.loads((bundle / "provenance.json").read_text())


def write_metadata(bundle, value):
    (bundle / "provenance.json").write_text(json.dumps(value, indent=2) + "\n")


def test_stored_reference_has_every_paper_grid_cell_once():
    rows, verification = read_reference()
    methods = {"RRR", "SRRR", "SOFAR", "RSSVD", "SMART_fixed", "SMART"}
    sample_sizes = {
        "0": (200, 400, 600, 800, 1000),
        "1": (300, 500, 700, 1000, 1200),
        "2": (500, 700, 1000, 1200, 1500),
    }
    expected = set()
    for model, sizes in sample_sizes.items():
        grids = {
            "exp1": sizes,
            "exp2": (1, 3, 5, 7, 9, 11),
            "exp3": (0, 3, 5, 7, 10, 15, 20),
            "exp4": (0, .01, .02, .05, .1, .5),
        }
        expected.update((model, experiment, float(x), method)
                        for experiment, values in grids.items()
                        for x in values for method in methods)
    identifiers = [(row["model_id"], row["experiment"], float(row["x"]), row["method"])
                   for row in rows]
    assert len(rows) == len(expected) == verification["row_count"] == 432
    assert set(identifiers) == expected
    assert all(count == 1 for count in Counter(identifiers).values())
    assert all(isinstance(value, str) for row in rows for value in row.values())
    assert {row["paper_repetitions"] for row in rows} == {"100"}
    assert {row["source_kind"] for row in rows} == {"digitized_pdf_vector"}
    assert all(math.isfinite(float(row[field])) and float(row[field]) >= 0
               for row in rows for field in ("mean", "se"))


def test_copied_bundle_verifies_csv_and_provenance_without_pdfs(bundle):
    path = bundle / "v1_simulation_curves.csv"
    rows, verification = read_reference(path)
    assert len(rows) == 432 and verification["csv_hash_verified"] is True
    assert verification["csv_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert verification["provenance_sha256"] == hashlib.sha256(
        (bundle / "provenance.json").read_bytes()).hexdigest()
    assert verification["metric"] == "norm(C_hat - C_star, fro) / sqrt(p*q)"
    assert len(verification["source_pdfs"]) == 3
    assert {source["verification"] for source in verification["source_pdfs"]} == {"not_requested"}
    assert not list(bundle.parent.rglob("*.pdf"))
    assert verify_reference(path) == verification


def test_isolated_bundle_loads_with_stdlib_python_only(bundle):
    script = """
import json
import sys
sys.path.insert(0, sys.argv[1])
from paper_reference import read_reference, verify_reference
rows, verification = read_reference()
assert verify_reference() == verification
assert not any(name in sys.modules for name in ('pdfplumber', 'numpy', 'pandas', 'matplotlib'))
print(json.dumps({'rows': len(rows), 'hash_verified': verification['csv_hash_verified'],
                  'pdf_states': [source['verification'] for source in verification['source_pdfs']]}))
"""
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("PYTHONPATH", "PYTHONHOME")}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, "-S", "-c", script, str(bundle.parent)],
                            cwd=bundle.parent, env=environment, capture_output=True,
                            text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "rows": 432, "hash_verified": True, "pdf_states": ["not_requested"] * 3,
    }


def test_csv_tampering_is_rejected_before_using_changed_results(bundle):
    path = bundle / "v1_simulation_curves.csv"
    original = path.read_bytes()
    assert b"0.02208100" in original
    path.write_bytes(original.replace(b"0.02208100", b"0.12208100", 1))
    with pytest.raises(ValueError, match=r"(?i)(checksum|sha256|hash)"):
        read_reference(path)


def test_duplicate_cell_rejected_even_with_consistent_csv_checksum(bundle):
    path = bundle / "v1_simulation_curves.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[-1] = rows[0].copy()
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    value = metadata(bundle)
    value["csv"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_metadata(bundle, value)
    with pytest.raises(ValueError, match=r"(?i)(duplicate|cell|grid)"):
        read_reference(path)


def test_missing_provenance_is_a_clear_asset_error(bundle):
    (bundle / "provenance.json").unlink()
    with pytest.raises(ValueError, match=r"(?i)provenance"):
        read_reference(bundle / "v1_simulation_curves.csv")


@pytest.mark.parametrize("bad_checksum", ["a" * 63, "g" * 64, None])
def test_malformed_csv_checksum_is_rejected(bundle, bad_checksum):
    value = metadata(bundle)
    value["csv"]["sha256"] = bad_checksum
    write_metadata(bundle, value)
    with pytest.raises(ValueError, match=r"(?i)(checksum|sha256|hash)"):
        verify_reference(bundle / "v1_simulation_curves.csv")


def test_corrupt_declared_row_count_is_rejected(bundle):
    value = metadata(bundle)
    value["row_count"] = 431
    write_metadata(bundle, value)
    with pytest.raises(ValueError, match=r"(?i)(row|count)"):
        read_reference(bundle / "v1_simulation_curves.csv")


def test_requested_but_missing_pdfs_are_explicitly_unavailable(bundle, tmp_path):
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()
    rows, verification = read_reference(bundle / "v1_simulation_curves.csv",
                                        manuscript_root=manuscript)
    assert len(rows) == 432 and verification["csv_hash_verified"] is True
    assert {source["verification"] for source in verification["source_pdfs"]} == {"unavailable"}


def test_present_pdf_is_hash_verified_and_missing_ones_remain_unavailable(bundle, tmp_path):
    manuscript = tmp_path / "manuscript"
    value = metadata(bundle)
    first = value["sources"][0]
    path = manuscript / first["pdf"]
    path.parent.mkdir(parents=True)
    content = b"%PDF-1.7\nSynthetic checksum fixture; no PDF parsing is required.\n"
    path.write_bytes(content)
    first["sha256"] = hashlib.sha256(content).hexdigest()
    write_metadata(bundle, value)
    verification = verify_reference(bundle / "v1_simulation_curves.csv",
                                    manuscript_root=manuscript)
    by_pdf = {source["pdf"]: source for source in verification["source_pdfs"]}
    assert by_pdf[first["pdf"]]["verification"] == "verified"
    assert by_pdf[first["pdf"]]["sha256"] == first["sha256"]
    assert [source["verification"] for source in verification["source_pdfs"]].count("unavailable") == 2

    path.write_bytes(content + b"changed bytes\n")
    with pytest.raises(ValueError, match=r"(?i)(PDF|checksum|sha256|hash)"):
        read_reference(bundle / "v1_simulation_curves.csv", manuscript_root=manuscript)


def test_unrequested_present_pdfs_do_not_block_portable_loading(bundle, tmp_path, monkeypatch):
    value = metadata(bundle)
    manuscript = tmp_path / "manuscript"
    path = manuscript / value["sources"][0]["pdf"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"This deliberately does not match the source PDF hash.")
    monkeypatch.chdir(manuscript)
    rows, verification = read_reference(bundle / "v1_simulation_curves.csv")
    assert len(rows) == 432
    assert {source["verification"] for source in verification["source_pdfs"]} == {"not_requested"}
