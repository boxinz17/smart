"""Package completed audited summaries and selected receipts, without arrays."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def export(root: Path, destination: Path, *, include_receipts: bool = False) -> dict:
    root = root.resolve()
    rank = (root / "rank-plan.json").exists()
    audit_name = "rank-audit.json" if rank else "audit.json"
    audit = json.loads((root / audit_name).read_text())
    if not audit.get("audit_passed") or not audit.get("all_planned_tasks_complete"):
        raise ValueError("A complete passing independent audit is required")
    files = set(root.glob("*.json")) | set(root.glob("*.csv"))
    files.update(path for path in (root / "report").rglob("*") if path.is_file())
    if include_receipts:
        tasks = root / ("rank_tasks" if rank else "tasks")
        files.update(tasks.glob("*/result.json"))
    manifest = {"remote_root": str(root), "scope": "Audited compact evidence; no numerical arrays or unselected candidate archives", "selected_result_receipts_included": include_receipts, "files": {}}
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(files):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Unexpected nonregular evidence file: {path}")
            name = str(path.relative_to(root))
            data = path.read_bytes()
            manifest["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(data))
        data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        entry = tarfile.TarInfo("compact-export-manifest.json")
        entry.size = len(data)
        entry.mode = 0o644
        archive.addfile(entry, io.BytesIO(data))
    return {"archive": str(destination), "n_files": len(files), "bytes": destination.stat().st_size, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--include-receipts", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.destination, include_receipts=args.include_receipts), sort_keys=True))
