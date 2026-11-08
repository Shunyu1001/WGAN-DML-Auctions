"""Package the local PSRM confidence-set revision without replacing archives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "output/psrm_reporting_update_2026-11-08"
EXPECTED_CHANGED = {
    "paper/main.tex", "paper/sections/appendix.tex",
    "paper/sections/conclusion.tex", "paper/sections/declarations.tex",
    "paper/sections/inference.tex", "paper/sections/introduction.tex",
    "paper/sections/simulations.tex",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(path: Path):
    return sorted(p for p in path.rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts
                  and p.suffix not in {".zip", ".pyc"})


def main():
    baseline = json.loads((REVISION / "baseline_hashes.json").read_text())
    changed = sorted(name for name, value in baseline.items()
                     if sha(ROOT / name) != value)
    if set(changed) != EXPECTED_CHANGED:
        raise AssertionError(f"Unexpected changes to baseline paper: {changed}")
    new_paper = sorted(str(p.relative_to(ROOT)) for p in files(ROOT / "paper")
                       if str(p.relative_to(ROOT)) not in baseline)
    expected_new = {
        "paper/figures/disconnected_sets_diagnostic.png",
        "paper/tables/fresh_score_validation.tex",
        "paper/tables/confidence_set_flagged.tex",
        "paper/tables/score_set_sensitivity.tex",
    }
    if set(new_paper) != expected_new:
        raise AssertionError(f"Unexpected new paper files: {new_paper}")

    table_checks = json.loads((REVISION / "table_generation_checks.json").read_text())
    assert table_checks["passed"]
    assert all(sha(ROOT / name) == value
               for name, value in table_checks["source_sha256"].items())
    log = (REVISION / "build/main.log").read_text()
    warning_patterns = [r"Overfull", r"Underfull", r"undefined",
                        r"multiply defined", r"LaTeX Error", r"! Emergency stop"]
    warnings = [pattern for pattern in warning_patterns if re.search(pattern, log)]
    if warnings:
        raise AssertionError(f"Unresolved build warnings: {warnings}")
    match = re.search(r"Output written on .*?\((\d+) pages", log)
    assert match

    pdf = ROOT / "output/pdf/WGAN-DML-Auctions-PSRM-Score-Sets-2026-11-08.pdf"
    if pdf.exists():
        raise FileExistsError(pdf)
    shutil.copy2(REVISION / "build/main.pdf", pdf)
    source_zip = REVISION / "PSRM-Score-Sets-Source-2026-11-08.zip"
    bundle_zip = REVISION / "PSRM-Score-Sets-Revision-Bundle-2026-11-08.zip"
    for path in (source_zip, bundle_zip):
        if path.exists():
            raise FileExistsError(path)

    paper_files = [p for p in files(ROOT / "paper")
                   if p.suffix in {".tex", ".bib", ".png"}
                   and p.name not in {"adviser_memo.tex"}]
    with zipfile.ZipFile(source_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paper_files:
            archive.write(path, path.relative_to(ROOT / "paper"))

    manifest = {
        "revision": "PSRM confidence-set reporting update, 2026-11-08",
        "scope": "Local PSRM working paper only; no GitHub or Overleaf sync in this operation",
        "frozen_thesis": "Excluded from every input and output path; not modified",
        "pages": int(match.group(1)),
        "changed_existing_paper_files": changed,
        "new_paper_files": new_paper,
        "all_other_baseline_paper_files_unchanged": True,
        "source_results_unchanged": True,
        "source_results_sha256": table_checks["source_sha256"],
        "build_warning_patterns_found": warnings,
        "pdf": {"path": str(pdf.relative_to(ROOT)), "sha256": sha(pdf)},
        "source_zip": {"path": str(source_zip.relative_to(ROOT)), "sha256": sha(source_zip)},
        "evidence": {
            "fresh_validation": "100 repetitions each at n=500 and n=2000; seed 2026110800",
            "interpretation": "Modest n=500 coverage gains; no established GAN coverage advantage",
            "confidence_sets": "All accepted components retained; interval envelope reported separately",
            "chronology": "Score inversion fixed before validation; envelope/flags added after diagnosis",
            "exact_reserve_inference": "Not claimed; maintained target is fixed-bandwidth theta_0,h",
        },
    }
    manifest_path = REVISION / "revision_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    included = set(paper_files)
    included.update(files(ROOT / "code"))
    included.add(ROOT / "requirements.txt")
    included.add(pdf)
    included.add(manifest_path)
    included.add(REVISION / "table_generation_checks.json")
    included.add(REVISION / "baseline_hashes.json")
    included.add(REVISION / "compile.log")
    for directory in (
        ROOT / "output/reports/confidence_sets_2026-11-08",
        ROOT / "output/experiments/fixed_h_score_validation_2026-11-08/summary",
        ROOT / "output/experiments/fixed_h_score_validation_2026-11-08/source_snapshot",
        ROOT / "output/experiments/disconnected_sets_2026-11-08",
    ):
        included.update(files(directory))
    fresh_root = ROOT / "output/experiments/fixed_h_score_validation_2026-11-08"
    for name in ("protocol.json", "run_manifest.json", "seed_plan.json"):
        included.add(fresh_root / name)
    with zipfile.ZipFile(bundle_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(included):
            archive.write(path, path.relative_to(ROOT))
    print(json.dumps({"pdf": str(pdf), "pages": manifest["pages"],
                      "source_zip": str(source_zip), "bundle_zip": str(bundle_zip),
                      "changed": changed, "new": new_paper,
                      "all_checks_passed": True}, indent=2))


if __name__ == "__main__":
    main()
