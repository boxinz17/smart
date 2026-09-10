"""Check Discovery's numerical policy without importing or fitting estimators."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("discovery_runtime_check", REPO / "environment/check_runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


@pytest.fixture
def numerical_runtime(tmp_path, monkeypatch):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor : 0\nflags : sse2 avx avx2 fma\n"
                       "processor : 1\nflags : sse2 avx avx2 fma\n")
    pools = [dict(user_api="blas", internal_api="openblas", architecture="Haswell",
                  num_threads=1, version="fixture", filepath=f"/fixture/{name}.libs/libopenblas.so")
             for name in ("numpy", "scipy")]
    cpu = SimpleNamespace(__cpu_baseline__=["X86_V2"],
        __cpu_dispatch__=["X86_V3", "X86_V4", "AVX512_ICL", "AVX512_SPR"],
        __cpu_features__={"SSE2": True, "AVX2": True, "X86_V2": True,
                          "X86_V3": False, "X86_V4": False, "AVX512_ICL": False, "AVX512_SPR": False})
    imported = []

    def import_module(name):
        imported.append(name)
        return {
            "numpy": SimpleNamespace(__file__="/fixture/numpy/__init__.py"),
            "numpy._core._multiarray_umath": cpu,
            "scipy.linalg": SimpleNamespace(__file__="/fixture/scipy/linalg/__init__.py"),
            "threadpoolctl": SimpleNamespace(threadpool_info=lambda: pools),
        }[name]

    monkeypatch.setattr(runtime, "import_module", import_module)
    monkeypatch.setenv("OPENBLAS_CORETYPE", "Haswell")
    monkeypatch.setenv("NPY_DISABLE_CPU_FEATURES", "X86_V3,X86_V4,AVX512_ICL,AVX512_SPR")
    for name in runtime.NUMERICAL_ENVIRONMENT:
        if name.endswith("NUM_THREADS") or name == "VECLIB_MAXIMUM_THREADS":
            monkeypatch.setenv(name, "1")
    return cpuinfo, pools, imported, cpu


def test_fixed_core_report_checks_both_libraries_and_preserves_active_evidence(numerical_runtime):
    cpuinfo, pools, imported, _ = numerical_runtime
    report = runtime.check_blas_runtime("Haswell", cpuinfo_path=cpuinfo)
    assert imported == ["numpy", "numpy._core._multiarray_umath", "scipy.linalg", "threadpoolctl"]
    assert report["blas_core"] == "Haswell"
    assert report["cpu_flags_verified"] == ["avx2", "fma"]
    assert report["processors_checked"] == 2
    assert report["environment"]["OPENBLAS_CORETYPE"] == "Haswell"
    assert report["environment"]["OPENBLAS_NUM_THREADS"] == "1"
    assert report["environment"]["NPY_DISABLE_CPU_FEATURES"] == "X86_V3,X86_V4,AVX512_ICL,AVX512_SPR"
    assert report["numpy_simd"] == dict(baseline=["X86_V2"],
        dispatch=["X86_V3", "X86_V4", "AVX512_ICL", "AVX512_SPR"], active_features=["AVX2", "SSE2", "X86_V2"])
    assert report["threadpools"] == pools
    assert json.loads(json.dumps(report)) == report


@pytest.mark.parametrize("contents", ["", "flags : avx2\n", "flags : fma\n",
                                      "flags : avx2 fma\nflags : avx2\n"])
def test_unsupported_cpu_is_rejected_before_any_numerical_import(numerical_runtime, contents):
    cpuinfo, _, imported, _ = numerical_runtime
    cpuinfo.write_text(contents)
    with pytest.raises(ValueError, match="requires CPU flags"):
        runtime.check_blas_runtime("Haswell", cpuinfo_path=cpuinfo)
    assert imported == []


def test_missing_cpuinfo_is_rejected_before_numerical_import(numerical_runtime):
    cpuinfo, _, imported, _ = numerical_runtime
    cpuinfo.unlink()
    with pytest.raises(OSError):
        runtime.check_blas_runtime("Haswell", cpuinfo_path=cpuinfo)
    assert imported == []


def test_unset_or_different_requested_core_is_not_silently_accepted(numerical_runtime, monkeypatch):
    cpuinfo, _, imported, _ = numerical_runtime
    monkeypatch.setenv("OPENBLAS_CORETYPE", "Cooperlake")
    with pytest.raises(ValueError, match="OPENBLAS_CORETYPE must be Haswell"):
        runtime.check_blas_runtime("Haswell", cpuinfo_path=cpuinfo)
    assert imported == []


def test_numpy_dispatch_policy_must_be_exported_before_import(numerical_runtime, monkeypatch):
    cpuinfo, _, imported, _ = numerical_runtime
    monkeypatch.setenv("NPY_DISABLE_CPU_FEATURES", "AVX512_ICL,AVX512_SPR")
    with pytest.raises(ValueError, match="NPY_DISABLE_CPU_FEATURES must be"):
        runtime.check_blas_runtime("Haswell", cpuinfo_path=cpuinfo)
    assert imported == []


@pytest.mark.parametrize("mutation", [
    lambda cpu: setattr(cpu, "__cpu_baseline__", ["X86_V3"]),
    lambda cpu: cpu.__cpu_dispatch__.append("FUTURE_GROUP"),
    lambda cpu: cpu.__cpu_dispatch__.remove("X86_V4"),
    lambda cpu: cpu.__cpu_features__.update(X86_V2=False),
    lambda cpu: cpu.__cpu_features__.update(X86_V3=True),
    lambda cpu: cpu.__cpu_features__.update(AVX512_SPR=True),
    lambda cpu: cpu.__cpu_features__.pop("AVX512_ICL"),
])
def test_different_numpy_build_or_enabled_dispatch_is_rejected(numerical_runtime, mutation):
    cpuinfo, _, imported, cpu = numerical_runtime
    mutation(cpu)
    with pytest.raises(ValueError, match="NumPy"):
        runtime.check_blas_runtime("Haswell", cpuinfo_path=cpuinfo)
    assert imported == ["numpy", "numpy._core._multiarray_umath"]


@pytest.mark.parametrize("mutation", [
    lambda pools: pools[0].update(architecture="SkylakeX"),
    lambda pools: pools[1].update(architecture="Cooperlake"),
    lambda pools: pools[1].update(num_threads=2),
    lambda pools: pools[1].update(num_threads=True),
    lambda pools: pools[1].update(internal_api="mkl"),
    lambda pools: pools.pop(),
    lambda pools: pools.clear(),
    lambda pools: pools[1].update(filepath="/other/libopenblas.so"),
])
def test_missing_backend_wrong_kernel_or_thread_count_fails_closed(numerical_runtime, mutation):
    cpuinfo, pools, _, _ = numerical_runtime
    mutation(pools)
    with pytest.raises(ValueError, match="loaded.*pool|BLAS pool"):
        runtime.check_blas_runtime("Haswell", cpuinfo_path=cpuinfo)


def test_metadata_only_runtime_check_remains_portable_and_skips_blas(tmp_path, monkeypatch):
    (tmp_path / ".python-version").write_text(f"{sys.version_info.major}.{sys.version_info.minor}\n")
    (tmp_path / "python-constraints.txt").write_text("".join(f"{name}==fixture\n" for name in runtime.RUNTIME_PACKAGES))
    monkeypatch.setattr(runtime, "version", lambda name: "fixture")
    monkeypatch.setattr(runtime, "check_blas_runtime", lambda core: pytest.fail("Unexpected numerical import"))
    assert runtime.check_runtime(tmp_path) == {
        "Python": runtime.platform.python_version(), **dict.fromkeys(runtime.RUNTIME_PACKAGES, "fixture")}
    with pytest.raises(ValueError, match="cannot be combined"):
        runtime.check_runtime(tmp_path, python_only=True, blas_core="Haswell")


def test_requested_runtime_policy_is_added_to_existing_version_report(tmp_path, monkeypatch):
    (tmp_path / ".python-version").write_text(f"{sys.version_info.major}.{sys.version_info.minor}\n")
    (tmp_path / "python-constraints.txt").write_text("".join(f"{name}==fixture\n" for name in runtime.RUNTIME_PACKAGES))
    monkeypatch.setattr(runtime, "version", lambda name: "fixture")
    policies = []

    def check(core):
        policies.append(core)
        return {"blas_core": core}

    monkeypatch.setattr(runtime, "check_blas_runtime", check)
    assert runtime.check_runtime(tmp_path, blas_core="Haswell")["numerical_runtime"] == {"blas_core": "Haswell"}
    assert policies == ["Haswell"]


def test_actual_discovery_environment_forces_policy_before_first_python(tmp_path):
    source = tmp_path / "source with spaces"
    (source / "environment").mkdir(parents=True)
    for name in (".python-version", "python-constraints.txt", "environment/check_runtime.py"):
        (source / name).write_text("fixture\n")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/activate").write_text("# No real environment activation required.\n")
    fake_python = venv / "bin/python"
    fake_python.write_text(f"#!{sys.executable}\n" +
        "import json, os, sys\nfrom pathlib import Path\n" +
        "Path(os.environ['RUNTIME_CAPTURE']).write_text(json.dumps({'args':sys.argv[1:], " +
        "'environment':{key:os.environ.get(key) for key in " + repr(runtime.NUMERICAL_ENVIRONMENT) + "}}))\n")
    fake_python.chmod(0o755)
    capture = tmp_path / "runtime.json"
    environment = dict(os.environ, SMART_SOURCE_ROOT=str(source), VENV=str(venv),
        RUNTIME_ENV_SCRIPT=str(REPO / "hpc/discovery/env.sh"), RUNTIME_CAPTURE=str(capture),
        MPLCONFIGDIR=str(tmp_path / "matplotlib"), OPENBLAS_CORETYPE="Cooperlake",
        NPY_DISABLE_CPU_FEATURES="AVX512_ICL")
    for key in runtime.NUMERICAL_ENVIRONMENT:
        if key.endswith("NUM_THREADS") or key == "VECLIB_MAXIMUM_THREADS":
            environment[key] = "64"
    result = subprocess.run(["bash", "-c", 'module() { return 0; }; source "$RUNTIME_ENV_SCRIPT"'],
        env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    report = json.loads(capture.read_text())
    assert report["args"] == [str(source / "environment/check_runtime.py"), "--root", str(source),
                              "--blas-core", "Haswell"]
    assert report["environment"]["OPENBLAS_CORETYPE"] == "Haswell"
    assert report["environment"]["NPY_DISABLE_CPU_FEATURES"] == "X86_V3,X86_V4,AVX512_ICL,AVX512_SPR"
    for key, value in report["environment"].items():
        if key.endswith("NUM_THREADS") or key == "VECLIB_MAXIMUM_THREADS":
            assert value == "1"
