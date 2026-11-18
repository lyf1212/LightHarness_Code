"""Find the 32 original inputs by filename and SHA256 in user-supplied datasets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True,
                        help="Dataset directory; repeat for multiple datasets")
    args = parser.parse_args()
    records = json.loads((ROOT / "results/scene_manifest.json").read_text())
    wanted = {r["source_filename"] for r in records}
    candidates = {name: [] for name in wanted}
    for root in args.root:
        if not root.is_dir():
            parser.error(f"Dataset directory not found: {root}")
        for path in root.rglob("*"):
            if path.name in wanted and path.is_file():
                candidates[path.name].append(path)
    missing = []
    for record in records:
        target = (ROOT / "configs" / record["image"]).resolve()
        choices = ([target] if target.exists() else []) + candidates[record["source_filename"]]
        match = next((p for p in choices if hashlib.sha256(p.read_bytes()).hexdigest() == record["source_sha256"]), None)
        if match is None:
            missing.append(record["id"])
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != record["source_sha256"]:
            raise FileExistsError(f"Refusing to overwrite a different image: {target}")
        if match.resolve() != target:
            shutil.copy2(match, target)
    if missing:
        raise SystemExit("Inputs missing or hashes differ:\n" + "\n".join(missing))
    print("Verified and staged all 32 original inputs in data/benchmark.")


if __name__ == "__main__":
    main()
