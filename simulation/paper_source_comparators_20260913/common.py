"""Small provenance helpers shared by the paper comparator campaign."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

METHOD = "PaperSourceComparators"
METHODS = ("target_rrr", "target_ridge_rrr", "source_subspace_rrr",
           "source_subspace_ridge_rrr", "ridge_to_source", "source_target_mixture",
           "nuclear_contrast", "initializer_only")
RANK_INDEPENDENT = ("ridge_to_source", "nuclear_contrast")

def require(condition, message):
    if not condition:
        raise ValueError(message)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(canonical(value) + b"\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def immutable(path, value):
    path = Path(path)
    contents = canonical(value) + b"\n"
    if path.exists():
        require(not path.is_symlink() and path.read_bytes() == contents, f"frozen input changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            f.write(contents)

def load_plan(root, full_source=False):
    root = Path(root).resolve()
    plan = read(root / "plan.json")
    require(plan.get("method") == METHOD and plan.get("schema_version") == 1, "unsupported plan")
    require(plan["root"] == str(root), "plan root differs")
    require(plan["plan_fingerprint"] == digest({k:v for k,v in plan.items() if k != "plan_fingerprint"}),
            "plan fingerprint differs")
    require(sha(root / "source-manifest.json") == plan["source_manifest_sha256"], "source manifest differs")
    manifest = read(root / "source-manifest.json")
    source = root / "source"
    require(Path(manifest["source_root"]).resolve() == source.resolve(), "source root differs")
    if full_source:
        for name, expected in manifest["files"].items():
            path = (source / name).resolve()
            require(not Path(name).is_absolute() and path.is_relative_to(source.resolve())
                    and sha(path) == expected, f"source changed: {name}")
    return plan

def verify_imports(root):
    manifest = read(Path(root) / "source-manifest.json")
    source = Path(manifest["source_root"]).resolve()
    prefixes = ("paper_source_comparators_20260913", "reviewer_revision_20260913", "sparse_smart_v2", "sparse_smart", "smart")
    for name, module in list(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            continue
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            require(path.is_relative_to(source), f"import outside frozen source: {name}")
            require(sha(path) == manifest["files"].get(path.relative_to(source).as_posix()),
                    f"import hash differs: {name}")

def require_slurm(root):
    require(bool(os.environ.get("SLURM_JOB_ID")), "scientific work requires Slurm")
    require(Path(root).resolve().is_relative_to(Path("/scratch2")), "output must reside on scratch2")
