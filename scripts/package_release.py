"""Build a code archive from the explicit release roots, excluding local work files."""
import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRS = ("lightagent", "tests", "configs", "docs", "assets", "results", "evaluation", "examples", "requirements", "scripts")
FILES = ("README.md", "README_zh-CN.md", "requirements.txt", ".gitignore", ".env.example", "pytest.ini", "release_validation.json", "source_manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "LightProxy_release.zip")
    args = parser.parse_args()
    paths = [ROOT / n for n in FILES]
    for name in DIRS:
        for p in (ROOT / name).rglob("*"):
            if not p.is_file():
                continue
            parts = p.relative_to(ROOT).parts
            if any(v in parts for v in ("__pycache__", ".pytest_cache", "rerender", "images", "models")):
                continue
            if p.suffix in (".pyc", ".pyo", ".log"):
                continue
            paths.append(p)
    paths.extend((ROOT / "third_party/notices").glob("*"))
    paths.append(ROOT / "third_party/sources.json")
    paths = sorted(set(paths))
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}" for p in paths]
    manifest = ROOT / "MANIFEST.sha256"
    manifest.write_text("\n".join(lines) + "\n")
    paths.append(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in paths:
            if path.is_symlink():
                raise ValueError(f"Release cannot contain symlinks: {path}")
            z.write(path, "LightProxy_release/" + path.relative_to(ROOT).as_posix())
    print(f"{args.output}: {len(paths)} files, {args.output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
