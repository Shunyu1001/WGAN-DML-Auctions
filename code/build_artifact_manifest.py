#!/usr/bin/env python3
"""Create or verify checksums for the manuscript's committed result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "replication" / "artifact_manifest.json"
ARTIFACT_GLOBS = (
    "paper/tables/*.csv",
    "paper/tables/*.json",
    "paper/tables/*.tex",
    "paper/figures/*.png",
)


def artifact_paths() -> list[Path]:
    paths = {path for pattern in ARTIFACT_GLOBS for path in ROOT.glob(pattern)}
    return sorted(paths, key=lambda path: path.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_manifest() -> dict[str, object]:
    files = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in artifact_paths()
    ]
    return {
        "algorithm": "sha256",
        "scope": list(ARTIFACT_GLOBS),
        "files": files,
    }


def write_manifest() -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(current_manifest(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")


def check_manifest() -> None:
    if not MANIFEST.exists():
        raise SystemExit(
            "Artifact manifest is missing; run "
            "`python3 code/build_artifact_manifest.py --write`."
        )
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current = current_manifest()
    if recorded != current:
        recorded_by_path = {item["path"]: item for item in recorded.get("files", [])}
        current_by_path = {item["path"]: item for item in current["files"]}
        changed = sorted(
            path
            for path in recorded_by_path.keys() | current_by_path.keys()
            if recorded_by_path.get(path) != current_by_path.get(path)
        )
        detail = "\n".join(f"  - {path}" for path in changed)
        raise SystemExit(
            "Artifact manifest does not match the committed result files:\n"
            f"{detail}\nRegenerate it only after intentionally reproducing results."
        )
    print(f"Verified {len(current['files'])} result artifacts against SHA-256 manifest.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="write the manifest")
    action.add_argument("--check", action="store_true", help="verify the manifest")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.write:
        write_manifest()
    else:
        check_manifest()
