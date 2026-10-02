"""Print a compact status of a run directory on Titan (reads result.json files only).

    python code/single_cell_v2/titan/check_run.py runs/RUNTAG
"""

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
results = sorted(root.glob("*/*/result.json"))
print(f"{root}: {len(results)} complete tasks")
for path in results:
    r = json.loads(path.read_text())
    failed = [k for k, m in r["methods"].items() if not m.get("success")]
    seconds = {k: round(m.get("elapsed_seconds", 0)) for k, m in r["methods"].items()
               if m.get("elapsed_seconds", 0) >= 30}
    print(f"{path.parent.parent.name}/{path.parent.name}: mode={r['mode']} config={r['config_sha256'][:12]} "
          f"total={r['elapsed_seconds']:.0f}s ranks={r['ranks']} genes={r['preprocessing']['n_genes']} "
          f"sparse_smart={r['sparse_smart']['status_counts']} failed={failed or 'none'} "
          f"test_scored={sum('test' in m for m in r['methods'].values())} slow={seconds}")
