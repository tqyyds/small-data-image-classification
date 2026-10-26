"""Check the public allowlist, local documentation links and headline claims."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
from PIL import Image

from .verify_results import ROOT, verify


def main() -> None:
    paths = json.loads((ROOT / "public-files.json").read_text(encoding="utf-8"))
    if len(paths) != len(set(paths)) or "public-files.json" not in paths:
        raise ValueError("Public allowlist is duplicated or omits itself")
    allowed_roots = {"src", "tests", "docs", "scripts", "optional_transfer_cnn", "results", ".github"}
    forbidden_parts = {"data", "outputs", ".local", ".git", ".venv", "__pycache__"}
    forbidden_suffixes = {".pt", ".pth", ".pkl", ".joblib", ".npy", ".npz", ".zip", ".pdf"}
    expected_images = {f"results/figures/{name}.png" for name in
                       ("model_comparison", "bird_per_class", "task1_confusion", "task2_confusion")}
    secret_patterns = [r"github_pat_[A-Za-z0-9_]{30,}", r"gh[pousr]_[A-Za-z0-9]{30,}",
                       r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
                       r"[A-Za-z]:[\\/]Users[\\/][^\s]+"]
    for relative in paths:
        path = ROOT / relative
        parts = Path(relative).parts
        if (Path(relative).is_absolute() or ".." in parts or
            (len(parts) > 1 and parts[0] not in allowed_roots) or
            any(part in forbidden_parts or part.startswith(".env") for part in parts) or
            path.suffix.lower() in forbidden_suffixes or not path.is_file() or
            not path.resolve().is_relative_to(ROOT)):
            raise ValueError(f"Unsafe or missing public file: {relative}")
        if path.stat().st_size > 5_000_000:
            raise ValueError(f"Unexpectedly large public file: {relative}")
        if path.suffix == ".png":
            if relative not in expected_images:
                raise ValueError(f"Unreviewed image: {relative}")
            with Image.open(path) as image:
                image.verify()
            continue
        text = path.read_text(encoding="utf-8-sig")
        if any(re.search(pattern, text) for pattern in secret_patterns):
            raise ValueError(f"Potential credential/local-path exposure: {relative}")
        if path.suffix == ".md":
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
                if target.startswith(("https://", "http://", "#", "mailto:")):
                    continue
                target_path = unquote(target.split("#", 1)[0])
                resolved = (path.parent / target_path).resolve()
                if not resolved.is_file() or resolved.relative_to(ROOT).as_posix() not in paths:
                    raise ValueError(f"Broken/unpublished Markdown target in {relative}: {target}")
    actual_images = {relative for relative in paths if relative.endswith(".png")}
    if actual_images != expected_images:
        raise ValueError("Public figure set is incomplete")
    result = verify()
    saved = json.loads((ROOT / "results/verification.json").read_text())
    if result != saved:
        raise ValueError("Stored verification result is stale")
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    historical = pd.read_csv(ROOT / "results/historical/model_comparison.csv")
    expected = historical.drop_duplicates().copy()
    rows = {row for entry in manifest["experiments"] for row in entry.get("table_rows", [])}
    expected["evidence_level"] = ["prediction_verified" if i in rows else "summary_only" for i in expected.index]
    pd.testing.assert_frame_equal(expected.reset_index(drop=True),
                                  pd.read_csv(ROOT / "results/model_comparison.csv"),
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for key in ("task1_core_efficientnet_b0_official_logistic_regression",
                "task2_kaggle_cnn_plus_provided_official_random_forest",
                "task1_resnet50_transfer_tta", "task2_resnet50_transfer_tta"):
        metric = next(row["macro_f1"] for row in result["experiments"] if row["id"] == key)
        if f"{metric:.4f}" not in readme:
            raise ValueError(f"README headline missing verified result: {key}")
    print(f"PASS: {len(paths)} allowlisted files, four PNGs, local Markdown links and headline claims.")


if __name__ == "__main__":
    main()
