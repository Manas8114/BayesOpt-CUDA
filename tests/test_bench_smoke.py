"""
tests/test_bench_smoke.py
A fast smoke-test that runs the benchmark script in --quick mode and
verifies the output JSON is well-formed with the expected keys.
"""

import json
import os
import subprocess
import sys


def test_benchmark_runs():
    """Verify bench_kernels.py --quick exits cleanly and produces valid JSON."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script = os.path.join(project_root, "benchmarks", "bench_kernels.py")

    result = subprocess.run(
        [sys.executable, script, "--quick",
         "--n-warmup", "3", "--n-repeat", "5"],
        capture_output=True, text=True, cwd=project_root
    )
    assert result.returncode == 0, (
        f"bench_kernels.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # Find the saved JSON path from stdout
    json_path = None
    for line in result.stdout.splitlines():
        if "bench_" in line and ".json" in line:
            # Extract path after "->"
            parts = line.split("->")
            if len(parts) == 2:
                json_path = parts[1].strip()
                break

    assert json_path is not None, f"Could not find JSON path in output:\n{result.stdout}"
    assert os.path.isfile(json_path), f"JSON file not found: {json_path}"

    with open(json_path) as f:
        payload = json.load(f)

    # Validate structure
    assert "meta" in payload
    assert "results" in payload
    assert len(payload["results"]) > 0

    meta = payload["meta"]
    assert "gpu" in meta
    assert "n_warmup" in meta
    assert "n_repeat" in meta

    for r in payload["results"]:
        assert "kernel" in r
        assert "n" in r and "m" in r and "d" in r
        assert "naive" in r
        assert "speedup_mean" in r
        # Timing values must be non-negative
        assert r["naive"]["mean_ms"] > 0
        opt_key = "tiled" if "tiled" in r else "fused"
        assert r[opt_key]["mean_ms"] > 0
