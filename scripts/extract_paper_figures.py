"""Extract the figures from the supplied manuscript PDF without changing their content."""
import argparse
import hashlib
import json
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
REGIONS = {
    "teaser": (0, [58, 244, 554, 380]),
    "method": (3, [58, 71, 554, 302]),
    "qualitative": (5, [58, 72, 554, 462]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path)
    args = parser.parse_args()
    out = ROOT / "assets/paper"
    out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(args.paper)
    records = []
    for name, (page, box) in REGIONS.items():
        clip = fitz.Rect(box)
        doc[page].get_pixmap(matrix=fitz.Matrix(4, 4), clip=clip, alpha=False).save(out / f"{name}.png")
        cropped = fitz.open()
        target = cropped.new_page(width=clip.width, height=clip.height)
        target.show_pdf_page(target.rect, doc, page, clip=clip)
        cropped.save(out / f"{name}.pdf", garbage=4, deflate=True)
        records.append({"name": name, "page_1_based": page + 1, "clip_points": box})
    (out / "provenance.json").write_text(json.dumps({
        "source": args.paper.name,
        "sha256": hashlib.sha256(args.paper.read_bytes()).hexdigest(),
        "figures": records,
        "operation": "PDF crop and 288-dpi rasterization; no content edits",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
