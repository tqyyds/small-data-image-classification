"""Package only the reviewed public-file allowlist; never include local data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_base = dist / (args.name or f"small_data_image_classification_{timestamp}")
    archive_path = archive_base.with_suffix(".zip")
    if archive_path.exists():
        archive_path.unlink()

    allowlist = json.loads((root / "public-files.json").read_text(encoding="utf-8"))
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as zip_file:
        for relative in allowlist:
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Invalid public allowlist entry: {relative}")
            zip_file.write(path, relative)
    print(archive_path)


if __name__ == "__main__":
    main()
