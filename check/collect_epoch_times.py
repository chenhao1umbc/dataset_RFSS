"""
Epoch times and concurrency of the final training runs, from the run logs under final/<run>/log.txt.

For each run the log gives the seconds per epoch (t=...). The log's modification time is the end of its last epoch, so the start of
every epoch follows from the cumulative times; the concurrency of an epoch is the number of other final training runs running at the
midpoint of that epoch. Scoring jobs, learning-rate screens and the seed-0 pilots (their logs were not kept) are not counted.

Usage:
    uv run python check/collect_epoch_times.py

Output: check/epoch_times.json (read by paper/make_journal_assets.py for the cost table).
"""

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "check" / "epoch_times.json"
DEVICES = {"stft_blstm": "CPU"}


def read_run(log: Path) -> dict:
    text = log.read_text()
    times = [float(t) for t in re.findall(r"t=([0-9.]+)s", text)]
    end = log.stat().st_mtime
    starts = end - np.cumsum(times[::-1])[::-1]
    return {"run": log.parent.name, "params": int(re.search(r"Model parameters: ([0-9,]+)", text).group(1).replace(",", "")),
            "epoch_seconds": times, "epoch_start": starts.tolist(), "epoch_end": (starts + np.array(times)).tolist()}


def main():
    runs = [read_run(p) for p in sorted((ROOT / "final").glob("*/log.txt")) if "aborted" not in p.parent.name]
    for run in runs:
        mids = [(s + e) / 2 for s, e in zip(run["epoch_start"], run["epoch_end"])]
        run["concurrent_others"] = [sum(any(s <= m <= e for s, e in zip(o["epoch_start"], o["epoch_end"])) for o in runs if o is not run) for m in mids]
        run["device"] = DEVICES.get(run["run"].rsplit("_", 2)[0], "MPS")
    for run in runs:
        del run["epoch_start"], run["epoch_end"]
    OUTPUT.write_text(json.dumps(runs, indent=1))
    for r in runs:
        t, c = np.array(r["epoch_seconds"]), r["concurrent_others"]
        print(f"{r['run']:34s} epochs {len(t):2d} min/med/max {t.min():6.0f} {np.median(t):6.0f} {t.max():6.0f}  others {min(c)}-{max(c)} (median {np.median(c):.0f})")


if __name__ == "__main__":
    main()
