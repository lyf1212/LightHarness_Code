"""Check out the vision-model source revisions used by this experiment."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run(args, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", "sam3", "vggt", "lumen"], default="all")
    args = parser.parse_args()
    sources = json.loads((ROOT / "third_party/sources.json").read_text())
    for name, item in sources.items():
        if args.model not in ("all", name):
            continue
        target = ROOT / "third_party" / name
        if target.exists():
            head = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
            if head != item["commit"]:
                raise SystemExit(f"Existing {target} is at a different revision; move it before fetching.")
            print(f"Already at recorded revision: {name}")
            continue
        run("git", "clone", "--no-checkout", item["url"], str(target))
        run("git", "-C", str(target), "checkout", "--detach", item["commit"])
        run("git", "-C", str(target), "submodule", "update", "--init", "--recursive")


if __name__ == "__main__":
    main()
