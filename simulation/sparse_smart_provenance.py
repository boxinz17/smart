"""Portable source-content identities and strict SparseSMART resume checks.

Absolute paths are retained as provenance, but never enter the implementation
digest. A copied checkout therefore has the same identity; changed source does
not. Legacy digests without this explicit scheme remain historical evidence
and are not silently upgraded during resume.
"""

from collections.abc import Mapping
import hashlib
import inspect
import json
from pathlib import Path


FINGERPRINT_SCHEME = "sparse-smart-source-content-v1"


def _manifest_digest(manifest):
    payload = dict(scheme=FINGERPRINT_SCHEME, manifest=manifest)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def implementation_provenance(api, generator, runner_files):
    """Hash logical source roles and bytes, recording locations separately.

    The generator's package is included when it lives in a Python package;
    a standalone generator is identified by its source filename and callable
    name. No checkout prefix or dynamic import-module alias affects the hash.
    External scientific dependencies remain governed by the shared runtime.
    """
    sources = {}

    def add(name, path):
        path = Path(path).resolve()
        if name in sources and sources[name] != path:
            raise ValueError(f"Ambiguous implementation source identity: {name}")
        sources[name] = path

    add("simulation/sparse_smart_provenance.py", __file__)
    for path in runner_files:
        add(f"simulation/{Path(path).name}", path)
    module_file = getattr(api, "__file__", None)
    if module_file:
        package = Path(module_file).resolve().parent
        for path in package.rglob("*.py"):
            add(f"sparse_smart/{path.relative_to(package).as_posix()}", path)
        api_kind = "source-package"
    else:
        api_kind = "not-used" if api is None else "injected-api"

    generator_name = None
    if generator is not None:
        generator_name = getattr(generator, "__qualname__", type(generator).__qualname__)
        try:
            source = inspect.getsourcefile(generator)
        except TypeError:
            source = None
        if source is None:
            raise ValueError("The generator must have inspectable source for implementation provenance")
        source = Path(source).resolve()
        package = source.parent
        if (package / "__init__.py").is_file():
            while (package.parent / "__init__.py").is_file():
                package = package.parent
            for path in package.rglob("*.py"):
                add(f"generator/{package.name}/{path.relative_to(package).as_posix()}", path)
        else:
            add(f"generator/{source.name}", source)

    manifest = dict(api=api_kind, generator=generator_name, files=[
        dict(name=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for name, path in sorted(sources.items())
    ])
    return dict(
        implementation_fingerprint_scheme=FINGERPRINT_SCHEME,
        implementation_fingerprint=_manifest_digest(manifest),
        implementation_manifest=manifest,
        implementation_source_locations={name: str(path) for name, path in sorted(sources.items())},
    )


def validate_resume_identity(existing, identity, destination, *, digest, extra_identity=None):
    """Recompute the saved identity digest before comparing to the request."""
    valid = isinstance(existing, Mapping) and all(key in existing for key in identity)
    if valid:
        saved_identity = {key: existing[key] for key in identity}
        claimed = existing.get("configuration_fingerprint")
        try:
            # The diagnostic serializer maps nonfinite values to null; that
            # must not make a malformed saved identity look like a valid null.
            json.dumps(saved_identity, allow_nan=False)
            valid = digest(saved_identity) == claimed == digest(identity)
        except (TypeError, ValueError):
            valid = False
        valid = valid and all(existing.get(key) == value
                              for key, value in (extra_identity or {}).items())
    if not valid:
        raise ValueError(f"Existing result configuration differs at {destination}; "
                         "its stored identity payload must match its fingerprint. "
                         "Use --force where supported or choose a fresh output root")


def validate_resume_implementation(existing, current, destination):
    """Reject legacy schemes, altered manifests, and genuine source changes."""
    if existing.get("implementation_fingerprint_scheme") != FINGERPRINT_SCHEME:
        raise ValueError(f"Legacy or unsupported implementation fingerprint scheme at {destination}; "
                         "cannot safely resume without explicit recomputation. "
                         "Use --force where supported or choose a fresh output root")
    manifest = existing.get("implementation_manifest")
    if (not isinstance(manifest, Mapping)
            or _manifest_digest(manifest) != existing.get("implementation_fingerprint")
            or existing.get("implementation_fingerprint") != current["implementation_fingerprint"]):
        raise ValueError(f"Implementation differs from {destination}; source manifest or content changed. "
                         "Use --force where supported or choose a fresh output root")
