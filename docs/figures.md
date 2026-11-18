# Figures

| File | Source | Reproduction |
| --- | --- | --- |
| `assets/paper/teaser.png` / `.pdf` | Figure 1 of the supplied LightHarness manuscript | PDF crop, no content edits |
| `assets/paper/method.png` / `.pdf` | Figure 3 of the same manuscript | PDF crop, no content edits |
| `assets/paper/qualitative.png` / `.pdf` | Figure 4 of the same manuscript | PDF crop, no content edits |
| `assets/main_results.png` / `.pdf` / `.svg` | Main-table CSV | `python scripts/plot_results.py` |
| `assets/ablation_results.png` / `.pdf` / `.svg` | 16-image ablation CSV | `python scripts/plot_results.py` |
| `assets/method/lightproxy_method_v3.*` | Earlier editable method diagram from the experiment | `python assets/method/draw_framework_v3.py` |
| `assets/conceptual_overview.png` | Earlier conceptual comparison illustration | Preserved source image; illustrative, not a measured result |

The README uses the current manuscript figures. The earlier method diagram is included as editable SVG/PDF with its plotting script, image assets, and experiment-relative provenance. Its example uses round 6, while the current paper method figure uses another example with round 7.

`assets/paper/provenance.json` records the source PDF's SHA256, page numbers, and crop rectangles. To extract again from that same PDF:

```bash
python -m pip install pymupdf==1.25.5
python scripts/extract_paper_figures.py /path/to/manuscript.pdf
```

The crops retain labels and aspect ratios. Their resolution is 288 dpi; the accompanying PDFs retain vector content where present in the source. The qualitative montage follows the manuscript's selection, and its labels are not rewritten to match a different metric snapshot. The quantitative plots read numbers directly from CSV. No result images were synthesized or retouched for this release.
