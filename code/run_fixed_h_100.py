"""Execute a fresh, paired 100-repetition comparison without changing estimators."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SEED = 202611080
SIZES = (500, 2000)
REPETITIONS = 100
SHARD_SIZE = 10


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def seed_plan():
    records = []
    for size in SIZES:
        for rep in range(REPETITIONS):
            base = SEED + size * 100 + rep
            records.append(dict(auctions=size, repetition=rep+1, data_seed=base,
                                fold_seed=base+31415926,
                                gan_seeds=[base+1000000+(k+1)*10000 for k in range(3)],
                                generation_seeds=[base+1000000+(k+1)*10000+102 for k in range(3)]))
    all_seeds = []
    for r in records:
        all_seeds.extend([r["data_seed"], r["fold_seed"], *r["gan_seeds"], *r["generation_seeds"]])
    if len(set(all_seeds)) != len(all_seeds):
        raise AssertionError("Unexpected seed collision in the new run")
    old_seeds = set()
    for n in SIZES:
        for rep in range(10):
            b = 20261108+n*100+rep
            old_seeds.update([b, b+31415926])
            for k in range(3):
                old_seeds.update([b+1000000+(k+1)*10000, b+1000000+(k+1)*10000+102])
    if old_seeds.intersection(all_seeds):
        raise AssertionError("New seeds overlap the previous 10-repetition run")
    return records


def main(output, workers):
    if output.exists():
        raise FileExistsError(f"Refusing to reuse {output}")
    plan = seed_plan()
    output.mkdir(parents=True)
    (output/"logs").mkdir()
    source = output/"source_snapshot"/"code"
    source.mkdir(parents=True)
    for name in ("fixed_h_small_simulation.py", "run_fixed_h_100.py", "calibrated_wgan_pilot.py",
                 "orthogonal_wgan_dml.py", "wgan_gp_baseline.py", "crossfit_wgan_pilot.py",
                 "baseline_direct_inversion.py"):
        shutil.copy2(ROOT/"code"/name, source/name)
    old = ROOT/"output/experiments/fixed_h_small_2026-11-08"
    preservation = hashes(old)
    (output/"previous_run_sha256.json").write_text(json.dumps(preservation, indent=2)+"\n")
    (output/"seed_plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    master = dict(created_utc=datetime.now(timezone.utc).isoformat(), base_seed=SEED,
                  sample_sizes=SIZES, repetitions_per_size=REPETITIONS,
                  shard_size=SHARD_SIZE, workers=workers,
                  design="fresh independent repetitions; paired methods; unchanged fixed-h specification",
                  old_results_preserved=False, completed=False,
                  source_sha256=hashes(source))
    (output/"run_manifest.json").write_text(json.dumps(master, indent=2)+"\n")
    jobs = [(n, start) for start in range(0, REPETITIONS, SHARD_SIZE) for n in SIZES]
    def run(job):
        n, start = job
        cell = output/f"n{n}"/f"part_{start+1:03d}_{start+SHARD_SIZE:03d}"
        log = output/"logs"/f"n{n}_{start+1:03d}_{start+SHARD_SIZE:03d}.log"
        cmd = [sys.executable, str(source/"fixed_h_small_simulation.py"), "--size", str(n),
               "--reps", str(SHARD_SIZE), "--start-rep", str(start), "--seed", str(SEED),
               "--output-dir", str(cell)]
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        with log.open("x") as handle:
            result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            raise RuntimeError(f"Shard n={n}, reps={start+1}-{start+SHARD_SIZE} failed; see {log}")
        print(f"Finished n={n} repetitions={start+1}-{start+SHARD_SIZE}", flush=True)
        return cell
    started = time.perf_counter()
    cells = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = [executor.submit(run, job) for job in jobs]
        for future in as_completed(pending):
            cells.append(future.result())
    subprocess.run([sys.executable, str(source/"fixed_h_small_simulation.py"), "--aggregate",
                    *[str(p) for p in sorted(cells)], "--output-dir", str(output/"summary")], check=True)
    master.update(completed=True, elapsed_seconds=time.perf_counter()-started,
                  old_results_preserved=preservation == hashes(old))
    (output/"run_manifest.json").write_text(json.dumps(master, indent=2)+"\n")
    if not master["old_results_preserved"]:
        raise AssertionError("Previous run changed unexpectedly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    main(args.output_dir, args.workers)
