"""Run the independent audit once the already-started simulation exits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--runner-pid", type=int, required=True)
args = parser.parse_args()
while True:
    try:
        os.kill(args.runner_pid, 0)
    except ProcessLookupError:
        break
    time.sleep(30)
manifest = json.loads((args.run_dir/"run_manifest.json").read_text())
if manifest.get("completed"):
    verifier = Path(__file__).with_name("verify_fixed_h_100.py")
    with (args.run_dir/"completion_audit.log").open("x") as log:
        result = subprocess.run([sys.executable, str(verifier), "--run-dir", str(args.run_dir)],
                                stdout=log, stderr=subprocess.STDOUT)
    print(f"Independent verification exit code: {result.returncode}", flush=True)
    sys.exit(result.returncode)
print("Simulation exited without a completed manifest; no audit was run.", flush=True)
sys.exit(1)
