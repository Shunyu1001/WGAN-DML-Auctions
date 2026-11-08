"""Read-only local progress page for the separately running simulation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import io
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from urllib.parse import urlparse


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def snapshot(run, runner_pid):
    manifest = read_json(run/"run_manifest.json") or {}
    groups = defaultdict(set)
    rows = []
    for path in sorted(run.glob("n*/part_*/raw_results.csv")):
        try:
            content = path.read_text()
            # The writer can be observed between truncation and the final byte.
            # A result row is eligible only after its terminating newline exists.
            content = content[:content.rfind("\n")+1]
            local = list(csv.DictReader(io.StringIO(content)))
            methods = {"Calibrated WGAN-DML", "Empirical-local DML", "Smooth plug-in"}
            local = [r for r in local if all(v is not None for v in r.values())
                     and r.get("method") in methods and r.get("status") in {"ok", "failed"}
                     and r.get("data_seed", "").isdigit() and r.get("fold_seed", "").isdigit()]
            rows.extend(local)
            for row in local:
                groups[row.get("auctions"), row.get("repetition")].add(row["method"])
        except (OSError, csv.Error):
            continue
    counts = {str(n): sum(len(c) == 3 for (size, _), c in groups.items() if size == str(n))
              for n in (500, 2000)}
    total = sum(counts.values())
    created = datetime.fromisoformat(manifest["created_utc"]) if manifest.get("created_utc") else datetime.now(timezone.utc)
    elapsed = max(0, (datetime.now(timezone.utc)-created).total_seconds())
    complete = manifest.get("completed") is True
    if complete:
        elapsed = manifest.get("elapsed_seconds", elapsed)
    alive = process_alive(runner_pid)
    status = "complete" if complete else ("running" if alive else "stopped")
    shards = []
    for path in sorted((run/"logs").glob("*.log")):
        name = path.stem
        size, start, end = name.removeprefix("n").split("_")
        cell = run/f"n{size}"/f"part_{start}_{end}"
        protocol = read_json(cell/"protocol.json") or {}
        try:
            lines = [s.strip() for s in path.read_text(errors="replace").splitlines() if s.strip()]
        except OSError:
            lines = []
        finished = protocol.get("completed") is True
        completed = sum(len(c) == 3 for (n, rep), c in groups.items()
                        if n == size and rep and int(start) <= int(rep) <= int(end))
        shards.append(dict(size=int(size), start=int(start), end=int(end), completed=completed,
                           status="complete" if finished else status,
                           detail="已完成" if finished else (lines[-1] if lines else "等待启动")))
    metrics = []
    if complete:
        try:
            with (run/"summary/summary.csv").open(newline="") as handle:
                metrics = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            pass
    verification = read_json(run/"summary/verification.json")
    return dict(status=status, completed=total, planned=200, counts=counts,
                result_rows=len(rows), failed_results=sum(r.get("status") != "ok" for r in rows),
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=None if complete or total < 6 else elapsed*(200-total)/total,
                updated_utc=datetime.now(timezone.utc).isoformat(), shards=shards,
                metrics=metrics, verified=verification,
                old_results_preserved=manifest.get("old_results_preserved") if complete else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--port", type=int, default=8876)
    args = parser.parse_args()
    page = Path(__file__).resolve().parents[1]/"progress/index.html"
    downloads = {"summary.csv": "summary/summary.csv", "raw_results.csv": "summary/raw_results.csv",
                 "paired_comparisons.csv": "summary/paired_comparisons.csv", "protocol": "PROTOCOL.md"}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path).path
            if route in ("/", "/index.html"):
                body = page.read_bytes(); content_type = "text/html; charset=utf-8"
            elif route == "/api/progress":
                body = json.dumps(snapshot(args.run_dir, args.runner_pid), ensure_ascii=False).encode()
                content_type = "application/json; charset=utf-8"
            elif route.startswith("/download/") and route.rsplit("/", 1)[-1] in downloads:
                name = route.rsplit("/", 1)[-1]
                path = args.run_dir/downloads[name]
                if not path.is_file():
                    self.send_error(404, "Results are not ready"); return
                body = path.read_bytes(); content_type = "text/plain; charset=utf-8"
            else:
                self.send_error(404); return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers(); self.wfile.write(body)

        def log_message(self, *_):
            pass

    print(f"Progress page: http://127.0.0.1:{args.port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
