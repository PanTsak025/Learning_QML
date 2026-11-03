"""One-off repair: the MLP back-fill appended rows with the NEW column layout (30 columns,
including cpu_seconds) to result files whose header has the OLD layout (29 columns).
This rewrites every results CSV in the new layout. Old rows get cpu_seconds = empty (it
did not exist when they ran); no values are changed or lost. A backup of every file is
kept as <name>.bak before rewriting.

    python repair_results.py
"""
import csv, glob, shutil
import pandas as pd
from run_experiments import COLUMNS

OLD = [c for c in COLUMNS if c != "cpu_seconds"]
for path in sorted(glob.glob("results/main*/runs_*.csv")):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    if header == COLUMNS:
        print(f"ok (already new layout): {path}")
        continue
    if header != OLD:
        raise SystemExit(f"unexpected header in {path}: {header}")
    old = [r for r in body if len(r) == len(OLD)]
    new = [r for r in body if len(r) == len(COLUMNS)]
    other = [r for r in body if len(r) not in (len(OLD), len(COLUMNS))]
    if other:
        raise SystemExit(f"{path}: {len(other)} rows with an unexpected number of fields; not changed")
    shutil.copy(path, path + ".bak")
    out = pd.concat([pd.DataFrame(old, columns=OLD), pd.DataFrame(new, columns=COLUMNS)],
                    ignore_index=True).reindex(columns=COLUMNS)
    out.to_csv(path, index=False)
    check = pd.read_csv(path)
    print(f"repaired: {path}  ({len(old)} old-layout + {len(new)} new-layout rows -> {len(check)} rows)")