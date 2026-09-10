"""Read the saved V1 figure aggregates without PDF software or source PDFs."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

DEFAULT_REFERENCE = Path(__file__).resolve().parent / "v1_simulation_curves.csv"
METRIC = "norm(C_hat - C_star, fro) / sqrt(p*q)"
_METHODS = {"RRR", "SRRR", "SOFAR", "RSSVD", "SMART_fixed", "SMART"}
_DIMENSIONS = {0: (100, 50), 1: (150, 100), 2: (300, 200)}
_SAMPLE_GRIDS = {0: (200, 400, 600, 800, 1000),
                 1: (300, 500, 700, 1000, 1200),
                 2: (500, 700, 1000, 1200, 1500)}
_GRIDS = {"exp2": (1, 3, 5, 7, 9, 11), "exp3": (0, 3, 5, 7, 10, 15, 20),
          "exp4": (0, .01, .02, .05, .1, .5)}
_PARAMETERS = {"exp1": "n", "exp2": "r_hat", "exp3": "r_s", "exp4": "sigma0"}
_FIELDS = {"model_id", "model", "p", "q", "experiment", "varying_parameter", "x",
           "method", "method_label", "mean", "se", "paper_repetitions",
           "is_horizontal_reference", "source_kind", "source_pdf", "pdf_page"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path, description):
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"Missing or unreadable paper reference {description}: {path}") from error


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _checksum(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_reference(reference_path=DEFAULT_REFERENCE, *, manuscript_root=None):
    """Return CSV string-valued rows and an explicit verification record.

    The CSV checksum, metadata, and full figure grid are always checked.
    PDFs are optional: when ``manuscript_root`` is supplied, existing PDFs
    must match their recorded hashes; absent PDFs are reported as unavailable.
    Checksums establish bundle consistency, not an independent re-extraction
    or authenticity guarantee. No PDF parser is imported here.
    """
    path = Path(reference_path)
    raw = _read(path, "CSV")
    provenance_bytes = _read(path.with_name("provenance.json"), "provenance")
    try:
        provenance = json.loads(provenance_bytes)
    except (ValueError, UnicodeError) as error:
        raise ValueError("Invalid paper reference provenance JSON") from error
    _require(isinstance(provenance, dict), "Invalid paper reference provenance")
    artifact = provenance.get("csv")
    _require(isinstance(artifact, dict) and _checksum(artifact.get("sha256")),
             "Missing or invalid paper reference CSV checksum")
    _require(artifact.get("file") == path.name, "Paper reference CSV filename differs from provenance")
    digest = _sha256(raw)
    _require(digest == artifact["sha256"], "Paper reference CSV hash differs from provenance")
    _require(provenance.get("metric") == METRIC, "Paper reference metric mismatch")
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
        _require(set(reader.fieldnames or ()) == _FIELDS, "Invalid paper reference CSV columns")
        rows = list(reader)
    except (csv.Error, UnicodeError) as error:
        raise ValueError("Invalid paper reference CSV") from error
    _require(len(rows) == 432 and provenance.get("row_count") == len(rows),
             "Paper reference row count mismatch")
    expected = {(model, experiment, method, float(x))
                for model in _DIMENSIONS for experiment in _PARAMETERS for method in _METHODS
                for x in (_SAMPLE_GRIDS[model] if experiment == "exp1" else _GRIDS[experiment])}
    seen = set()
    for row in rows:
        try:
            model, experiment = int(row["model_id"]), row["experiment"]
            key = (model, experiment, row["method"], float(row["x"]))
            _require(key in expected and key not in seen, "Missing, duplicate, or unexpected paper reference cell")
            _require((int(row["p"]), int(row["q"])) == _DIMENSIONS[model]
                     and row["model"] == ("ModelI", "ModelII", "ModelIII")[model],
                     "Paper reference model dimensions mismatch")
            _require(row["varying_parameter"] == _PARAMETERS[experiment]
                     and int(row["paper_repetitions"]) == 100, "Paper reference experiment metadata mismatch")
            _require(all(math.isfinite(float(row[name])) and float(row[name]) >= 0 for name in ("mean", "se")),
                     "Invalid paper reference mean or SE")
            _require(row["source_kind"] == "digitized_pdf_vector" and int(row["pdf_page"]) == 1
                     and row["source_pdf"] == f"v1/fig_smart/simulation_model{model+1}.pdf",
                     "Invalid paper reference source metadata")
            _require(row["is_horizontal_reference"] in ("true", "false") and bool(row["method_label"]),
                     "Invalid paper reference curve metadata")
            seen.add(key)
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("Invalid paper reference CSV row") from error
    _require(seen == expected, "Incomplete paper reference grid")
    sources = provenance.get("sources")
    _require(isinstance(sources, list) and len(sources) == 3, "Incomplete paper reference provenance")
    expected_pdfs = {f"v1/fig_smart/simulation_model{model+1}.pdf" for model in _DIMENSIONS}
    checked = []
    for source in sources:
        _require(isinstance(source, dict) and source.get("pdf") in expected_pdfs
                 and _checksum(source.get("sha256")) and source.get("page") == 1,
                 "Invalid paper reference PDF provenance")
        expected_pdfs.remove(source["pdf"])
        status = "not_requested"
        if manuscript_root is not None:
            pdf = Path(manuscript_root) / source["pdf"]
            status = "unavailable"
            if pdf.exists():
                _require(pdf.is_file() and _sha256(_read(pdf, "PDF")) == source["sha256"],
                         f"Paper PDF hash differs: {source['pdf']}")
                status = "verified"
        checked.append(dict(pdf=source["pdf"], sha256=source["sha256"], verification=status))
    verification = dict(csv_sha256=digest, csv_hash_verified=True, row_count=len(rows),
                        provenance_sha256=_sha256(provenance_bytes), metric=METRIC, source_pdfs=checked)
    return rows, verification


def verify_reference(reference_path=DEFAULT_REFERENCE, *, manuscript_root=None):
    """Verify the saved bundle, optionally also checking available source PDFs."""
    return read_reference(reference_path, manuscript_root=manuscript_root)[1]
