"""Keep batch attempts inspectable without replacing a completed manifest on failure."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from run_sparse_smart import _atomic_json_dump


def manifest_for_resume(canonical):
    """Prefer a completed canonical manifest, otherwise the latest saved attempt."""
    canonical = Path(canonical)
    if canonical.exists():
        return canonical
    attempts = sorted(canonical.with_name(canonical.stem + "_attempts").glob("*.json"))
    return attempts[-1] if attempts else None


class BatchManifest:
    """Update one unique attempt; publish the canonical file only after completion.

    Failed and interrupted attempts retain their progress and errors at their
    own paths. A completed prior manifest is never modified by those attempts.
    """

    def __init__(self, canonical, manifest):
        self.canonical = Path(canonical)
        self.manifest = manifest
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = self.canonical.with_name(self.canonical.stem + "_attempts") / f"{stamp}-{uuid4().hex}.json"
        self.finished = False

    def __enter__(self):
        self.manifest.update(attempt_manifest=str(self.path.resolve()), attempt_status="running")
        self.update()
        return self

    def update(self):
        _atomic_json_dump(self.manifest, self.path)

    def finish(self, complete):
        self.manifest.update(attempt_status="completed" if complete else "failed")
        self.manifest.setdefault("finished", datetime.now(timezone.utc).isoformat())
        self.update()
        if complete:
            _atomic_json_dump(self.manifest, self.canonical)
        self.finished = True

    def __exit__(self, exc_type, error, traceback):
        if error is not None:
            self.manifest.update(
                attempt_status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                driver_error=dict(exception=type(error).__name__, message=str(error)),
                finished=datetime.now(timezone.utc).isoformat())
            self.update()
        elif not self.finished:
            self.finish(False)
        return False
