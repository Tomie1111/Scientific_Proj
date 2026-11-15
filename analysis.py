"""six separate least-squares models for possum head length.

Run: python analysis.py [optional/path/to/possum.csv]
Only the three selected columns determine which observations are included.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save PNGs without needing a graphical desktop.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
SELECTED = ["hdlngth", "totlngth", "skullw"]
PREDICTORS = {
    "totlngth": ("Total length (cm)", 88.0),
    "skullw": ("Skull width (mm)", 57.0),
}
COLORS = {"Linear": "#176b96", "Quadratic": "#c35b16", "Exponential": "#27805b"}


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def inspect_data(csv_path):
    """Inspect raw data, check numbers, then select ONE common set of rows."""
    raw = pd.read_csv(csv_path)
    missing_columns = set(SELECTED) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    # errors='raise' prevents bad text from silently becoming missing data.
    selected = raw[SELECTED].apply(pd.to_numeric, errors="raise")
    common = selected.dropna(subset=SELECTED).copy()
    if len(common) <= 3:
        raise ValueError("Need more than three complete observations.")
    if not np.isfinite(common.to_numpy()).all():
        raise ValueError("Selected data contain infinity; inspect the source.")
    if (common["hdlngth"] <= 0).any():
        raise ValueError("Exponential initialization requires positive head lengths.")
    if common["hdlngth"].var() == 0:
        raise ValueError("R-squared is undefined when all head lengths are equal.")

    inspection = {
        "title": "CSV inspection",
        "source": str(csv_path),
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "row_count": len(raw),
        "column_names": raw.columns.tolist(),
        "missing_values": {c: int(v) for c, v in raw.isna().sum().items()},
        "duplicate_rows_beyond_first": int(raw.duplicated().sum()),
        "raw_selected_dtypes": {c: str(raw[c].dtype) for c in SELECTED},
        "checked_numeric_dtypes": {c: str(selected[c].dtype) for c in SELECTED},
        "all_selected_columns_numeric": all(
            pd.api.types.is_numeric_dtype(selected[c]) for c in SELECTED
        ),
        "rows_used_by_every_model": len(common),
        "excluded_csv_rows_including_header": (raw.index[~raw.index.isin(common.index)] + 2).tolist(),
        "policy": "Drop only missing selected values. Retain duplicates and unusual observations.",
        "observed_ranges": {
            c: {"min": float(common[c].min()), "max": float(common[c].max())}
            for c in SELECTED
        },
    }
    return common, inspection


def fit_polynomial(x, y, degree):
    """Each matrix row is [1, x] or [1, x, x**2]."""
    if degree == 1:
        # Linear regression
        design = np.column_stack([np.ones_like(x), x])
    else:
        # Quadratic regression
        design = np.column_stack([np.ones_like(x), x, x**2])
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != degree + 1:
        raise ValueError("Design matrix lacks full column rank.")
    return {"function": "Linear" if degree == 1 else "Quadratic",
            "coefficients": dict(zip(["a", "b", "c"][:degree + 1], beta.tolist())),
            "k": degree + 1}


# Exponential regression
def fit_exponential(x, y):
    """Start with log(y), then minimize ORIGINAL-SCALE squared residuals."""
    design = np.column_stack([np.ones_like(x), x])
    log_beta, _, rank, _ = np.linalg.lstsq(design, np.log(y), rcond=None)
    if rank != 2:
        raise ValueError("Exponential design matrix lacks full column rank.")

    def residuals(parameters):
        alpha, B = parameters
        return np.exp(alpha + B*x) - y

    result = least_squares(residuals, log_beta, method="lm",
                           ftol=1e-12, xtol=1e-12, gtol=1e-12,
                           max_nfev=10000)
    if not result.success:
        raise RuntimeError(f"Exponential fit failed: {result.message}")
    alpha, B = result.x
    return {
        "function": "Exponential", "k": 2,
        "coefficients": {"A": float(np.exp(alpha)), "B": float(B)},
        "initial_log_coefficients": {"ln_A": float(log_beta[0]), "B": float(log_beta[1])},
        "initial_original_scale_SSE": float(np.sum(residuals(log_beta)**2)),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_evaluations": int(result.nfev),
    }


def predict(model, x):
    """Use fitted coefficients in exactly the equations stated in the project."""
    p = model["coefficients"]
    x = np.asarray(x, dtype=float)
    if model["function"] == "Exponential":
        return p["A"] * np.exp(p["B"] * x)
    y_hat = p["a"] + p["b"] * x
    if model["function"] == "Quadratic":
        y_hat = y_hat + p["c"] * x**2
    return y_hat


def equation(model):
    p = model["coefficients"]
    if model["function"] == "Exponential":
        return f"y = {p['A']:.10g} * exp({p['B']:.10g} * x)"
    text = f"y = {p['a']:.10g} {p['b']:+.10g} * x"
    if model["function"] == "Quadratic":
        text += f" {p['c']:+.10g} * x^2"
    return text


def metrics(y, y_hat, k):
    """All errors are in the original head-length scale (mm)."""
    n = len(y)
    residual = y - y_hat
    # SSE
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - np.mean(y))**2))
    return {
        "SSE": sse,
        # RMSE
        "RMSE": float(np.sqrt(sse/n)),
        # R²
        "R_squared": 1 - sse/sst,
        # Adjusted R²
        "adjusted_R_squared": 1 - (sse/(n-k))/(sst/(n-1)),
    }


def verify_model(model, y, y_hat):
    """Recompute formulas by scalar sums, independently of metrics()."""
    n, k = len(y), model["k"]
    mean_y = sum(float(v) for v in y) / n
    sse = sum((float(a) - float(b))**2 for a, b in zip(y, y_hat))
    sst = sum((float(v) - mean_y)**2 for v in y)
    expected = [sse, (sse/n)**0.5, 1-sse/sst,
                1-(sse/(n-k))/(sst/(n-1))]
    actual = list(model["metrics"].values())
    if not (np.isfinite(y_hat).all() and np.isfinite(actual).all()
            and np.isfinite(list(model["coefficients"].values())).all()):
        raise AssertionError("Non-finite prediction, metric, or coefficient.")
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    if model["function"] == "Exponential":
        tolerance = 1e-8 * max(1, sse)
        if sse > model["initial_original_scale_SSE"] + tolerance:
            raise AssertionError("Exponential refinement increased original-scale SSE.")


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def create_plots(x, y, models, predictor, xlabel, output):
    # linspace is sorted, dense, and stays inside the observed range.
    grid = np.linspace(x.min(), x.max(), 500)
    for model in models:
        name = model["function"]
        stem = f"{predictor}_{name.lower()}"
        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.scatter(x, y, s=25, alpha=0.65, color="#4b5563", label="Observed possums")
        ax.plot(grid, predict(model, grid), color=COLORS[name], lw=2, label=f"{name} fit")
        ax.set(title=f"Head length vs {xlabel.split(' (')[0].lower()} - {name}",
               xlabel=xlabel, ylabel="Head length (mm)")
        ax.legend()
        ax.grid(alpha=0.2)
        save_plot(fig, output / f"{stem}_fit.png")

        fitted = predict(model, x)
        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.scatter(fitted, y-fitted, s=25, alpha=0.7, color=COLORS[name])
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set(title=f"{predictor} {name} residuals",
               xlabel="Fitted head length (mm)",
               ylabel="Residual: observed - fitted head length (mm)")
        ax.grid(alpha=0.2)
        save_plot(fig, output / f"{stem}_residuals.png")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(x, y, s=25, alpha=0.55, color="#4b5563", label="Observed possums")
    for model, style in zip(models, ["-", "--", ":"]):
        name = model["function"]
        ax.plot(grid, predict(model, grid), color=COLORS[name], ls=style, lw=2, label=name)
    ax.set(title=f"Three functions for {xlabel.split(' (')[0].lower()}",
           xlabel=xlabel, ylabel="Head length (mm)")
    ax.legend()
    ax.grid(alpha=0.2)
    save_plot(fig, output / f"{predictor}_comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=ROOT / "possum.csv",
                        help="CSV path (default: possum.csv beside this script)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs",
                        help="Output folder (default: outputs beside this script)")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source = args.csv.resolve()
    if not source.is_file():
        parser.error(
            f"CSV file not found: {source}\n"
            "For the supplied data, run: python analysis.py\n"
            "To use another CSV, supply its actual existing file path."
        )
    # Refuse to overwrite an input whose name matches an output artifact.
    if source.parent == output:
        raise ValueError("Keep the input CSV outside the output folder.")
    data, inspection = inspect_data(source)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "data_inspection.json", inspection)
    versions = {"title": "execution versions", "Python": platform.python_version(),
                "pandas": pd.__version__, "NumPy": np.__version__,
                "SciPy": scipy.__version__, "Matplotlib": matplotlib.__version__}
    write_json(output / "package_versions.json", versions)

    y = data["hdlngth"].to_numpy(dtype=float)
    models_all, comparisons, coefficients, examples = [], [], [], []
    row_outputs = data.copy()
    row_outputs.insert(0, "csv_row_including_header", data.index + 2)
    checks = {"title": "numerical verification", "models_checked": 0,
              "metric_formulas_and_finiteness": "PASS"}
    for predictor, (xlabel, example_x) in PREDICTORS.items():
        x = data[predictor].to_numpy(dtype=float)
        models = [fit_polynomial(x, y, 1), fit_polynomial(x, y, 2), fit_exponential(x, y)]
        for model in models:
            model["predictor"] = predictor
            fitted = predict(model, x)
            model["metrics"] = metrics(y, fitted, model["k"])
            verify_model(model, y, fitted)
            checks["models_checked"] += 1
            comparisons.append({"predictor": predictor, "function": model["function"],
                                "n": len(y), "equation": equation(model), **model["metrics"]})
            for parameter, value in model["coefficients"].items():
                coefficients.append({"predictor": predictor, "function": model["function"],
                                     "parameter": parameter, "value": value})
            example_y = float(predict(model, example_x))
            if not np.isfinite(example_y):
                raise AssertionError("Example prediction is not finite.")
            examples.append({"predictor": predictor, "function": model["function"],
                             "input_value": example_x, "input_unit": "cm" if predictor == "totlngth" else "mm",
                             "predicted_head_length_mm": example_y,
                             "observed_min": float(x.min()), "observed_max": float(x.max()),
                             "within_observed_range": bool(x.min() <= example_x <= x.max())})
            stem = f"{predictor}_{model['function'].lower()}"
            row_outputs[f"{stem}_fitted_mm"] = fitted
            row_outputs[f"{stem}_residual_mm"] = y - fitted

        linear_sse, quad_sse = [m["metrics"]["SSE"] for m in models[:2]]
        tolerance = 1e-8 * max(1, linear_sse)
        if quad_sse > linear_sse + tolerance:
            raise AssertionError("Quadratic SSE exceeds linear SSE beyond tolerance.")
        checks[f"{predictor}_quadratic_SSE_no_greater_than_linear"] = "PASS"
        create_plots(x, y, models, predictor, xlabel, output)
        models_all.extend(models)

    captions = {
        "model_comparison.csv": "model comparison: in-sample fit on head length (mm); SSE in mm^2, RMSE in mm",
        "coefficients.csv": "fitted coefficients: x uses the predictor's stated units; y is head length (mm)",
        "example_predictions.csv": "separate example predictions of head length (mm)",
        "fitted_values_and_residuals.csv": "common observations, fitted head lengths and residuals (mm)",
    }
    tables = [pd.DataFrame(comparisons), pd.DataFrame(coefficients),
              pd.DataFrame(examples), row_outputs]
    log = ["possum least-squares analysis", json.dumps(inspection, indent=2)]
    for (filename, caption), table in zip(captions.items(), tables):
        table.to_csv(output / filename, index=False, float_format="%.15g")
        log.extend(["\n" + caption, table.to_string(index=False, float_format=lambda v: f"{v:.6f}")])
    (output / "table_captions.txt").write_text(
        "\n".join(f"{name}: {caption}" for name, caption in captions.items()) + "\n")
    checks["source_CSV_unchanged"] = (
        hashlib.sha256(source.read_bytes()).hexdigest() == inspection["source_sha256"]
    )
    if not checks["source_CSV_unchanged"]:
        raise AssertionError("Source CSV changed during analysis.")
    write_json(output / "verification.json", checks)
    write_json(output / "model_details.json", {"title": "fitted model details", "models": models_all})
    log.extend([
        "\nfitting notes",
        "Linear/quadratic: explicit design matrices solved by numpy.linalg.lstsq.",
        "Exponential: log least squares initializes alpha=ln(A), B; scipy.optimize.least_squares",
        "then minimizes sum((y-exp(alpha+B*x))^2) in original head-length units.",
        "Log fitting alone minimizes squared log errors, a different objective.",
        "R^2 measures improvement over predicting the sample mean. Adjusted R^2 penalizes",
        "parameter count using k=2,3,2. For nonlinear exponential fits it is a descriptive extension.",
        "All comparisons are in-sample; these metrics do not establish unseen-data accuracy.",
        "Inputs outside observed ranges require caution: fitted shapes may not continue there.",
        "AI helped write/comment the code and interpret outputs. The script was run on the supplied",
        "CSV; scalar formula checks, finite-value checks, nested-SSE checks and source hash checks passed.",
        json.dumps(checks, indent=2), json.dumps(versions, indent=2),
    ])
    (output / "analysis_output.txt").write_text("\n".join(log) + "\n")
    # Keep console output short; the full row-level table is saved above.
    print(log[0])
    print(json.dumps(inspection, indent=2))
    for filename, table in list(zip(captions, tables))[:3]:
        print("\n" + captions[filename])
        print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\nAll numerical checks passed. These are in-sample fit metrics.")
    print(f"Saved 14 PNG plots, tables, fitting notes and verification records to {output}")


if __name__ == "__main__":
    main()
