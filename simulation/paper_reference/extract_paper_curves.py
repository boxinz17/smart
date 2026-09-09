#!/usr/bin/env python3
"""Read the plotted means and standard-error bars in the existing V1 PDFs.

Requires pdfplumber. This does not load seed results, run an estimator, or render
new paper figures. The color/legend mapping and 1e-2 y-axis multiplier were
visually verified against all three original PDFs. Axis transforms themselves
are recovered from their vector tick marks and printed numeric tick labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import pdfplumber


REPO_ROOT = Path(__file__).resolve().parents[3]
METHOD_COLORS = {
    (0.0, 0.0, 1.0): "RRR",
    (1.0, 0.0, 0.0): "SRRR",
    (1.0, 0.6470588235, 0.0): "SOFAR",
    (0.5019607843, 0.0, 0.5019607843): "RSSVD",
    (0.5019607843, 0.5019607843, 0.5019607843): "SMART (fixed ranks)",
    (0.0, 0.5019607843, 0.0): "SMART",
}
PARAMETERS = {1: "n", 2: "r_hat", 3: "r_s", 4: "sigma0"}
EXPECTED_GRID = {
    1: [200, 400, 600, 800, 1000],
    2: [1, 3, 5, 7, 9, 11],
    3: [0, 3, 5, 7, 10, 15, 20],
    4: [0, 0.01, 0.02, 0.05, 0.1, 0.5],
}
SAMPLE_GRIDS = {
    1: [200, 400, 600, 800, 1000],
    2: [300, 500, 700, 1000, 1200],
    3: [500, 700, 1000, 1200, 1500],
}
MODEL_DIMENSIONS = {1: (100, 50), 2: (150, 100), 3: (300, 200)}
FIELDS = [
    "model_id", "model", "p", "q", "experiment", "varying_parameter", "x",
    "method", "method_label", "mean", "se",
    "paper_repetitions", "is_horizontal_reference", "source_kind", "source_pdf",
    "pdf_page",
]


def rgb(color):
    """PDF uses a grayscale scalar for the gray SMART line."""
    if isinstance(color, (int, float)):
        return (float(color),) * 3
    return tuple(color)


def close(a, b, atol=2e-6):
    return abs(a - b) <= atol


def numeric_words(page):
    return [
        word for word in page.extract_words()
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", word["text"])
    ]


def affine_calibration(pairs):
    """Least-squares coordinate -> data transform, checked at every tick."""
    assert len(pairs) >= 3, pairs
    mean_coordinate = sum(c for c, _ in pairs) / len(pairs)
    mean_value = sum(v for _, v in pairs) / len(pairs)
    slope = sum((c - mean_coordinate) * (v - mean_value) for c, v in pairs)
    slope /= sum((c - mean_coordinate) ** 2 for c, _ in pairs)
    intercept = mean_value - slope * mean_coordinate
    residual = max(abs(slope * c + intercept - v) for c, v in pairs)
    assert residual < 2e-5, (pairs, residual)
    return {"slope": slope, "intercept": intercept,
            "max_tick_residual": residual, "coordinate_value_pairs": pairs}


def calibrate_axis(page, box, words):
    left, top, right, bottom = box
    xpairs, ypairs = [], []
    for line in page.lines:
        if not close(line["linewidth"], 0.8):
            continue
        if (close(line["top"], bottom) and close(line["bottom"], bottom + 3.5)
                and close(line["x0"], line["x1"])
                and left <= line["x0"] <= right):
            candidates = [word for word in words
                          if 3 < word["top"] - bottom < 20
                          and abs((word["x0"] + word["x1"]) / 2 - line["x0"]) < 2]
            assert len(candidates) == 1, ("x tick", line, candidates)
            xpairs.append((line["x0"], float(candidates[0]["text"])))
        if (close(line["x1"], left) and close(line["x0"], left - 3.5)
                and close(line["top"], line["bottom"])
                and top - 1e-6 <= line["top"] <= bottom + 1e-6):
            candidates = [word for word in words
                          if 2 < left - word["x1"] < 20
                          and abs(word["top"] - line["top"]) < 8]
            # Some panels intentionally omit the zero tick label.
            assert len(candidates) <= 1, ("y tick", line, candidates)
            if candidates:
                ypairs.append((line["top"], float(candidates[0]["text"]) * 1e-2))
    return {"x": affine_calibration(xpairs), "y": affine_calibration(ypairs),
            "box_points": list(box), "printed_y_multiplier": 0.01}


def transform(value, calibration):
    return calibration["slope"] * value + calibration["intercept"]


def extract_pdf(path: Path, model: int):
    rows, axes_metadata = [], []
    with pdfplumber.open(path) as pdf:
        assert len(pdf.pages) == 1
        page = pdf.pages[0]
        boxes = sorted([
            (r["x0"], r["top"], r["x1"], r["bottom"]) for r in page.rects
            if 200 < r["width"] < page.width / 2 and 100 < r["height"] < page.height / 2
        ], key=lambda box: (round(box[1]), box[0]))
        assert len(boxes) == 4, boxes
        words = numeric_words(page)
        for experiment, box in enumerate(boxes, start=1):
            left, top, right, bottom = box
            calibration = calibrate_axis(page, box, words)
            curves = [(index, curve) for index, curve in enumerate(page.curves)
                      if close(curve["linewidth"], 1.5) and not curve["fill"]
                      and all(left <= x <= right and top <= y <= bottom
                              for x, y in curve["pts"])]
            assert len(curves) == 6, (model, experiment, len(curves))
            assert {rgb(c["stroking_color"]) for _, c in curves} == set(METHOD_COLORS)
            curve_metadata = []
            for curve_index, curve in curves:
                color = rgb(curve["stroking_color"])
                method = METHOD_COLORS[color]
                values = [transform(x, calibration["x"]) for x, _ in curve["pts"]]
                grid = SAMPLE_GRIDS[model] if experiment == 1 else EXPECTED_GRID[experiment]
                assert len(values) == len(grid)
                assert all(abs(a - b) < 2e-5 for a, b in zip(values, grid)), values
                curve_metadata.append({"method": method, "curve_index": curve_index,
                                       "color_rgb": color, "vertices_points": curve["pts"]})
                for parameter_value, (x, y) in zip(grid, curve["pts"]):
                    bars = [line for line in page.lines
                            if close(line["linewidth"], 1.5)
                            and rgb(line["stroking_color"]) == color
                            and close(line["x0"], x) and close(line["x1"], x)
                            and close((line["top"] + line["bottom"]) / 2, y)]
                    assert len(bars) == 1, (model, experiment, method, parameter_value, bars)
                    bar = bars[0]
                    mean = transform(y, calibration["y"])
                    se = abs(calibration["y"]["slope"]) * (bar["bottom"] - bar["top"]) / 2
                    assert math.isfinite(mean) and mean >= 0
                    assert math.isfinite(se) and se >= 0
                    horizontal = ((experiment in (3, 4) and not method.startswith("SMART"))
                                  or (experiment in (2, 3) and method == "SMART"))
                    p, q = MODEL_DIMENSIONS[model]
                    rows.append({
                        "model_id": model - 1,
                        "model": f"Model{['I', 'II', 'III'][model - 1]}", "p": p, "q": q,
                        "experiment": f"exp{experiment}", "varying_parameter": PARAMETERS[experiment],
                        "x": parameter_value,
                        "method": "SMART_fixed" if method == "SMART (fixed ranks)" else method,
                        "method_label": method, "mean": f"{mean:.8f}",
                        "se": f"{se:.8f}", "paper_repetitions": 100,
                        "is_horizontal_reference": str(horizontal).lower(),
                        "source_kind": "digitized_pdf_vector",
                        "source_pdf": str(path.relative_to(REPO_ROOT)), "pdf_page": 1,
                    })
            axes_metadata.append({"experiment": experiment, "calibration": calibration,
                                  "curves": curve_metadata})
    return rows, {"pdf": str(path.relative_to(REPO_ROOT)),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "page": 1, "axes": axes_metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    rows, sources = [], []
    for model in (1, 2, 3):
        data, provenance = extract_pdf(REPO_ROOT / f"v1/fig_smart/simulation_model{model}.pdf", model)
        rows.extend(data)
        sources.append(provenance)
    assert len(rows) == 432
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "v1_simulation_curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "description": "Plotted aggregate means and symmetric standard-error bars read from original V1 PDF vectors.",
        "metric": "norm(C_hat - C_star, fro) / sqrt(p*q)",
        "method_mapping": "Color/legend mapping and 1e-2 axis multiplier visually verified in all three original PDFs.",
        "paper_specification": "v1/main.tex, Simulation examples, Figures simulation_model1/2/3.",
        "limitations": [
            "These are digitized figure aggregates, not the underlying replicate outputs.",
            "CSV values are rounded to eight decimal places. More digits do not imply statistical accuracy.",
            "Bars are interpreted as standard errors from the experimental-design text in v1/main.tex.",
            "Repeated horizontal lines are references at the default configuration, not separate runs at each x value.",
            "The x=0 entry in Experiment 3 is retained exactly as plotted; no meaning is inferred for that sentinel.",
            "Figure aggregates do not permit paired tests against newly run seed-level results.",
        ],
        "row_count": len(rows), "sources": sources,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Extracted {len(rows)} plotted points from three PDFs into {args.output_dir}")


if __name__ == "__main__":
    main()
