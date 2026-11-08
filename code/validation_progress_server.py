"""Read-only localhost dashboard for fresh-seed Wald/score validation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
from urllib.parse import urlparse

METHODS = {"Calibrated WGAN-DML", "Empirical-local DML", "Smooth plug-in"}
SIZES = (500, 2000)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def process_alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def complete_rows(path):
    """Discard an unterminated last line and all incomplete/malformed records."""
    try:
        content = path.read_text()
        content = content[:content.rfind("\n")+1]
        candidates = list(csv.DictReader(io.StringIO(content)))
    except (OSError, csv.Error, UnicodeError):
        return []
    rows = []
    for row in candidates:
        try:
            if (None in row or any(value is None for value in row.values())
                    or row.get("method") not in METHODS
                    or row.get("status") not in {"ok", "failed"}
                    or not row.get("inversion_status")
                    or not row.get("data_seed", "").isdigit()
                    or not row.get("fold_seed", "").isdigit()
                    or int(row["auctions"]) not in SIZES
                    or not 1 <= int(row["repetition"]) <= 100):
                continue
        except (ValueError, KeyError, TypeError):
            continue
        rows.append(row)
    return rows


def snapshot(run, runner_pid):
    manifest = read_json(run/"run_manifest.json") or {}
    row_index = {}
    for path in sorted(run.glob("n*/part_*/raw_results.csv")):
        for row in complete_rows(path):
            row_index[row["auctions"], row["repetition"], row["method"]] = row
    rows = list(row_index.values())
    groups = defaultdict(set)
    for row in rows:
        groups[row["auctions"], row["repetition"]].add(row["method"])
    counts = {str(n): sum(methods == METHODS for (size, _), methods in groups.items()
                         if size == str(n)) for n in SIZES}
    total = sum(counts.values())
    now = datetime.now(timezone.utc)
    try:
        created = datetime.fromisoformat(manifest["created_utc"])
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
    except (KeyError, ValueError, TypeError):
        created = now
    elapsed = max(0.0, (now-created).total_seconds())
    complete = manifest.get("completed") is True
    if complete:
        elapsed = manifest.get("elapsed_seconds", elapsed)
    if complete:
        status = "complete"
    elif manifest.get("failed") or manifest.get("status") == "failed":
        status = "stopped"
    elif process_alive(runner_pid):
        status = "running"
    elif not manifest:
        status = "waiting"
    else:
        status = "stopped"
    shards = []
    for start in range(1, 101, 10):
        for size in SIZES:
            end = start+9
            cell = run/f"n{size}"/f"part_{start:03d}_{end:03d}"
            record = read_json(cell/"protocol.json") or read_json(cell/"completed.json") or {}
            log = run/"logs"/f"n{size}_{start:03d}_{end:03d}.log"
            try:
                lines = [line.strip() for line in log.read_text(errors="replace").splitlines() if line.strip()]
            except OSError:
                lines = []
            finished = record.get("completed") is True
            count = sum(methods == METHODS for (n, rep), methods in groups.items()
                        if n == str(size) and start <= int(rep) <= end)
            active = log.exists() or cell.exists()
            shards.append(dict(size=size, start=start, end=end, completed=count,
                status="complete" if finished else (status if active else "waiting"),
                detail="已完成" if finished else (lines[-1] if lines else "等待启动")))
    metrics = []
    if complete:
        try:
            with (run/"summary/summary.csv").open(newline="") as handle:
                metrics = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            pass
    point_failures = sum(row["status"] != "ok" for row in rows)
    score_failures = sum(row["status"] != "ok" or row["inversion_status"] != "ok" for row in rows)
    return dict(status=status, completed=total, planned=200, counts=counts,
        result_rows=len(rows), planned_result_rows=600,
        wald_failures=point_failures, score_failures=score_failures,
        elapsed_seconds=elapsed,
        estimated_remaining_seconds=None if complete or total < 6 else elapsed*(200-total)/total,
        updated_utc=now.isoformat(), shards=shards, metrics=metrics,
        verified=read_json(run/"summary/verification.json"),
        base_seed=manifest.get("base_seed"),
        seed_independence_verified=manifest.get("seed_independence_verified"),
        old_results_preserved=manifest.get("prior_results_preserved", manifest.get("old_results_preserved")) if complete else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--port", type=int, default=8877)
    args = parser.parse_args()
    page = Path(__file__).resolve().parents[1]/"progress/validation.html"
    downloads = {"summary.csv": "summary/summary.csv", "raw_results.csv": "summary/raw_results.csv",
                 "verification.json": "summary/verification.json", "protocol": "protocol.json"}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path).path
            if route in ("/", "/index.html", "/validation.html"):
                body = page.read_bytes(); content_type = "text/html; charset=utf-8"
            elif route == "/api/progress":
                body = json.dumps(snapshot(args.run_dir, args.runner_pid), ensure_ascii=False).encode()
                content_type = "application/json; charset=utf-8"
            elif route.startswith("/download/") and route.rsplit("/", 1)[-1] in downloads:
                name = route.rsplit("/", 1)[-1]
                path = args.run_dir/downloads[name]
                if not path.is_file():
                    self.send_error(404, "Results are not ready"); return
                body = path.read_bytes()
                content_type = "application/json; charset=utf-8" if name.endswith(".json") else "text/plain; charset=utf-8"
            else:
                self.send_error(404); return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    print(f"Validation progress: http://127.0.0.1:{args.port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
