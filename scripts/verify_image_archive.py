"""Verify the exact evaluation images and benchmark inputs, extracted or zipped."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Check the ZIP without extracting it")
    parser.add_argument("--root", type=Path, default=ROOT, help="Extracted repository root")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "results/image_archive_manifest.json").read_text())
    errors = []
    archive = zipfile.ZipFile(args.archive) if args.archive else None
    try:
        for record in manifest["files"]:
            try:
                stream = archive.open(record["path"]) if archive else (args.root / record["path"]).open("rb")
                with stream:
                    actual = digest(stream)
                if actual != record["sha256"]:
                    errors.append(record["path"] + ": SHA256 mismatch")
            except (FileNotFoundError, KeyError):
                errors.append(record["path"] + ": missing")
    finally:
        if archive:
            archive.close()
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(manifest['files'])} files.")


if __name__ == "__main__":
    main()
