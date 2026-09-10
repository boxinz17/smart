# V1 paper simulation reference

`v1_simulation_curves.csv` contains the plotted means and standard errors from
the three existing `v1/fig_smart/simulation_model*.pdf` figures used by
`v1/main.tex`. It has 432 rows: three models, four experiments, six methods,
and all plotted x coordinates. No other estimator was run to obtain these
references.

These values are already saved and versioned in the software repository.
**Cluster runs should read this CSV; they do not need the manuscript PDF,
figure PDFs, or `pdfplumber`.** Transfer the `paper_reference/` directory with
the source checkout. `provenance.json` now binds the CSV bytes with a SHA-256
checksum, in addition to retaining the original figure fingerprints.

The measurement is `||C_hat - C_star||_F / sqrt(p*q)`, as defined in the
manuscript's simulation section. `mean` and `se` are in this unscaled metric;
the printed figure axes use a multiplier of `1e-2`.
Compare these numbers with the simulation runner's coefficient error `avg_err`,
not its validation MSE or training objective.

## Use on Discovery without PDFs

From the software repository's `simulation/` directory:

```sh
python -m paper_reference
```

This checks the CSV checksum, all 432 rows, the complete model/experiment/method
grid, numeric values, and source metadata using the Python standard library.
It does not open any PDF or fit an estimator. The existing comparison scripts
use the saved reference by default. External-data comparison summaries also
check source PDFs if available; absent PDFs are recorded as `unavailable`,
while a mismatched available PDF still raises an error. Their metadata records
`paper_reference_verification`; `paper_pdf_hashes_verified` lists only files
actually checked on that machine.

For example, read the paper comparison for the single Model I noise-0.5 trial:

```python
from paper_reference import read_reference

rows, verification = read_reference()
for row in rows:
    if (row["model_id"] == "0" and row["experiment"] == "exp4"
            and float(row["x"]) == 0.5):
        print(row["method"], row["mean"], row["se"])
```

To additionally verify the source PDFs on a workstation that has the manuscript,
use `python -m paper_reference --manuscript-root /path/to/SMART_BoxinJinchi`.
Verification distinguishes recorded source hashes from hashes checked locally.
The saved metadata is an extraction audit trail; CSV consistency checks do not
constitute a fresh extraction or verification of unavailable PDF files.

## Reading the CSV

- `model_id`: 0, 1, 2, corresponding to Model I, II, III.
- `experiment`: `exp1`, `exp2`, `exp3`, or `exp4`.
- `x`: the displayed sample size, fitted rank, source truncation, or source
  noise standard deviation, respectively.
- `method`: `RRR`, `SRRR`, `SOFAR`, `RSSVD`, `SMART_fixed`, or `SMART`.
- `mean`, `se`: digitized figure mean and standard-error half-width, rounded
  to eight decimal places. `method_label` preserves the displayed legend.
- `is_horizontal_reference`: whether the manuscript says that the curve is
  repeated as a default-configuration reference, rather than rerun at each x.
- `paper_repetitions`: 100, as stated in the manuscript.
- `source_kind`, `source_pdf`, and `pdf_page`: distinguish these figure
  aggregates from newly computed replicate outputs.

These are figure aggregates, not the raw replicate results. Extra displayed
digits preserve the vector conversion; they do not establish additional
statistical accuracy. In particular, the CSV cannot supply paired seed-level
tests against a new estimator's results. The standard-error interpretation is
from the experimental-design text in `v1/main.tex`.

The original x=0 point in Experiment 3 is preserved as plotted. This extraction
does not assign a meaning to that sentinel.

## Reproduce the extraction

This is optional maintenance when paper figures change, not a cluster setup
step. From the manuscript repository root, using Python with `pdfplumber`
installed:

```sh
python code/simulation/paper_reference/extract_paper_curves.py
```

The extractor identifies the four axis rectangles, matches their vector ticks
to numeric labels, and calibrates linear coordinate transforms. The legend
colors and `1e-2` y multiplier were visually checked on complete renders of all
three original figures. It reads each mean from a polyline vertex and each SE
from the matching symmetric vertical error bar. Assertions require all expected
methods, grids, ticks, and bars to match. There is no pixel-based tracing or
interpolation between displayed points.

`provenance.json` retains the original PDF SHA-256 hashes, axis tick coordinates
and transforms, curve indices/colors/vertices, and interpretation limits. This
provides an audit trail if the manuscript figures subsequently change. PDF
originals are read only.

## Match experimental settings before comparing

Experiment 1 has model-specific sample-size grids:

| Model | Sample sizes |
|---|---|
| I | 200, 400, 600, 800, 1000 |
| II | 300, 500, 700, 1000, 1200 |
| III | 500, 700, 1000, 1200, 1500 |

Experiment 4 uses source noise values 0, 0.01, 0.02, 0.05, 0.1, and 0.5.

The text states a default sample size of 200. However, the target-only
horizontal reference levels in Experiments 3 and 4 equal Experiment 1's first
sample-size values for each model (200, 300, and 500), whereas the rank-5
target-only values in Experiment 2 differ for Models II and III. The PDFs alone
do not resolve the underlying sample sizes for those references. Consult the
original seed/configuration mapping before making those comparisons; do not
infer a common sample size solely from the horizontal curves. Model I
Experiments 1 and 4 give the unambiguous first comparison requested here.
