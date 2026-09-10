# Shared Python environment

Local development, GitHub Actions, and Discovery use Python **3.12** and one
scientific dependency set:

| Dependency | Version |
|---|---|
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.0 |
| joblib | 1.6.0 |
| threadpoolctl | 3.6.0 |

[`.python-version`](../.python-version) selects the Python minor version;
patch releases within 3.12 are allowed. Discovery currently loads its available
`python/3.12.8` module. [The shared constraints](../python-constraints.txt)
also pin the installed test/build tools and supporting simulation dependencies.
Constraints restrict versions but do not install optional packages by themselves.

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c python-constraints.txt setuptools wheel
python -m pip install --no-build-isolation -c python-constraints.txt \
  -e './smart[test]' -e './bi-smart[test]' -e './sparse-smart[test]'
python -m pip check
python environment/check_runtime.py
```

For a standalone SparseSMART environment, omit the `smart` and `bi-smart`
editable installs. Its wheel metadata pins the same five runtime dependencies
and requires Python 3.12. The other two packages are needed for their own
tests and the simulation data generators.

Use `.venv/bin/python` for new local runs, or activate `.venv` first. The runtime
check reads installed distribution metadata, verifies the five scientific pins,
and exits with an actionable error if they differ. It performs no fitting.
`--python-only` checks the interpreter before dependencies are installed.
The check does not claim to validate optional single-cell or R dependencies.

Both CI workflows use the shared files. SparseSMART retains Linux and macOS
coverage with a single Python/scikit-learn combination. The general workflow
uses the same versions for its SMART, BI-SMART, and simulation tests.

Discovery's bootstrap and worker activation use the same constraints and runtime
check. The [cluster guide](../hpc/discovery/README.md) explains deployment and
Slurm use. Editing the local configuration does not deploy it or submit a job.
Existing saved simulation artifacts retain their original version fingerprints.

## Historical verification snapshot

The verification recorded on **2026-09-09** in repository commit `d4b6e7b`
passed 325 SparseSMART tests, 110 BI-SMART tests, 8 SMART tests, and 353
simulation tests (796 total). The SparseSMART wheel built for that snapshot
also passed its 325 tests and all three example modes in this environment.
These counts describe the source and wheel checked then; they are not current
test totals or verification of subsequent source changes. The saved
[v0.5.0 wheel](../sparse-smart/dist/sparse_smart-0.5.0-py3-none-any.whl) is archival
and predates later audit fixes. Use the editable source installation above
for current development and runs.

Run the general suites in separate processes using their existing CI commands;
the simulation fixtures use pytest's default import mode. These checks do not
constitute a remote CI or cluster run.
