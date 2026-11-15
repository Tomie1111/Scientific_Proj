# possum least-squares analysis

Research question: How well can total length and skull width separately predict
a possum's head length?

This project contains Python code and analysis outputs. No report or slides are
included. The project brief was read as background; the requested scope and model
definitions determine this implementation.

## setup and run

Tested with Python 3.14.7. Use a virtual environment so dependencies stay local:

```bash
cd /Users/pannawatthawonwong/Scienctific_Proj
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py
```

The default input is `possum.csv` beside the script, even when the current working
directory is elsewhere. For the supplied data, just run `python analysis.py`.
You can also name the existing CSV explicitly and choose an output folder:

```bash
python analysis.py possum.csv --output-dir outputs
```

To analyze a different CSV, replace `possum.csv` with that file's actual path.
Paths such as `/path/to/possum.csv` are examples, not existing files on your Mac.
An explicitly supplied relative path is relative to your current working
directory. A repeated run replaces files with the same names in the output
folder. Keep the input outside that folder. Figures save automatically; no plot
window opens. `requirements.txt` pins the four analysis packages used in this run;
the exact Python and package versions are also saved with each run.

## files and outputs

- `possum.csv`: original supplied data, preserved unchanged. Its external source
  and collection year are not established by the CSV; no provenance was invented.
- `analysis.py`: main script, with inspection, fitting, prediction, metrics,
  plotting and verification functions in one file.
- `requirements.txt`: analysis dependencies.
- `outputs/model_comparison.csv`: all six equations, sample sizes and fit metrics.
- `outputs/coefficients.csv`: full-precision numeric coefficients (displayed
  equations use ten significant digits; calculations use unrounded values).
- `outputs/example_predictions.csv`: six predictions and observed-range checks.
- `outputs/fitted_values_and_residuals.csv`: common observations and each model's
  fitted values/residuals. `csv_row_including_header` links back to the source.
- `outputs/*_fit.png`: six individual scatterplots and fitted functions.
- `outputs/*_comparison.png`: two overlays of the three functions.
- `outputs/*_residuals.png`: six residual-versus-fitted plots.
- `outputs/data_inspection.json`: counts, column names, missing values, duplicate
  count, numeric checks, included sample size, ranges and source SHA-256 hash.
- `outputs/model_details.json`: coefficients, parameter counts, metrics and
  exponential initialization/convergence details.
- `outputs/package_versions.json`, `outputs/verification.json`: run versions and
  numerical checks.
- `outputs/table_captions.txt`: captions for the machine-readable CSV tables.
- `outputs/analysis_output.txt`: plain-text tables, inspection and fitting notes.

## data and fitting methods

`hdlngth` is head length in mm (outcome). `totlngth` is total length in cm and
`skullw` is skull width in mm. The predictors are fitted **separately**, never
together in a multiple regression. Units are kept as supplied.

The file has 104 rows and 14 columns. There are no duplicate rows. Only `age`
(two values) and `footlgth` (one value) are missing. All three selected columns
are numeric and complete, so all six models use the same 104 observations.
The code only excludes rows missing one of these three measurements. It does
not drop duplicates or unusual observations automatically.

For linear fitting, a row of the design matrix is `[1, x]` and coefficients are
`[a, b]`. For quadratic fitting, a row is `[1, x, x**2]` and coefficients are
`[a, b, c]`. `numpy.linalg.lstsq` minimizes the squared differences between
`design @ coefficients` and observed head length.

For exponential fitting, first solve `ln(y) = alpha + B*x` by least squares,
where `alpha = ln(A)`. This provides starting values. Then
`scipy.optimize.least_squares` minimizes `sum((exp(alpha + B*x) - y)**2)` in
the original mm units, and `A = exp(alpha)`. Fitting logarithms alone minimizes
squared log errors, a different objective that generally gives different
coefficients. This is numerical nonlinear optimization from that starting
estimate; convergence is checked, without claiming a global-optimum proof.

Every model uses the original head-length scale for comparison:

```text
residual = observed y - fitted y
SSE = sum(residual**2)                           (mm^2)
RMSE = sqrt(SSE/n)                              (mm)
SST = sum((y - mean(y))**2)                      (mm^2)
R^2 = 1 - SSE/SST
```

R² describes improvement over predicting the sample mean.
All metrics describe **in-sample fit**;
no held-out evaluation was requested, so they do not establish accuracy for
unseen possums or causal effects.

## observed results

The skull-width quadratic has the strongest in-sample fit among the six models:
R² = 0.623515 and RMSE = 2.181983 mm. Its R² means it
accounts for about 62.4% of the sample's variation in head length. The
total-length quadratic has R² = 0.484030 and RMSE = 2.554405 mm; its improvement
over total-length linear fitting is small (linear RMSE = 2.570246 mm).
For each predictor, the exponential has slightly worse in-sample metrics than
the linear function. The nonlinear shape does not automatically improve fit.

Example predictions, in function order linear / quadratic / exponential:

- Total length 88 cm: **93.125 / 93.317 / 93.094 mm**.
- Skull width 57 mm: **92.698 / 93.393 / 92.663 mm**.

Both inputs are within the observed ranges: total length 75–96.5 cm and skull
width 50–68.6 mm. These are fitted point predictions, not prediction intervals.
Outside those ranges, extrapolation requires caution because the assumed curves
may not describe possum anatomy there. In particular, the skull-width quadratic
turns downward near the upper end of the range, and sparse large skull widths
make that shape worth discussing with the residual plots. No observation was
removed to improve the curves.

## verification

Checks recompute metrics using scalar
sums, confirm finite coefficients/predictions/metrics, verify that quadratic
training SSE is no greater than linear training SSE within numerical tolerance,
check that exponential refinement does not increase original-scale SSE, and
compare the source hash before and after execution. The saved plots were also
visually inspected. These checks validate calculations on this file; they do not
independently verify measurement quality or unseen-data performance.
