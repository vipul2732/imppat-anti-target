"""Tanimoto Gaussian process for hERG pIC50.

This file has no per-compound assay standard deviation. A single noise
variance of 0.25, corresponding to a standard deviation of 0.5 pIC50
units, is placed on every training label. That is a fixed allowance for
mixed ChEMBL assays, not a measured error.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "results" / "splits" / "compound_folds.csv"
PAIRS = ROOT / "results" / "pairs" / "neighbor_pairs.csv"
BASELINE = ROOT / "results" / "baseline" / "regression_metrics.csv"
OUT = ROOT / "results" / "gp"
NOISE_VARIANCE = 0.25
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class Tanimoto(Kernel):
    def __init__(self):
        self.theta_ = np.empty(0)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)
        gram = X @ Y.T
        denom = np.sum(X * X, axis=1)[:, None] + np.sum(Y * Y, axis=1)[None, :] - gram
        sim = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
        if eval_gradient:
            return sim, np.empty((sim.shape[0], sim.shape[1], 0))
        return sim

    @property
    def theta(self):
        return np.empty(0)

    @theta.setter
    def theta(self, value):
        return

    @property
    def bounds(self):
        return np.empty((0, 2))

    def diag(self, X):
        return np.ones(len(X))

    def is_stationary(self):
        return False


def fingerprints(smiles):
    rows = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        arr = np.zeros((2048,), dtype=np.float64)
        if mol is not None:
            DataStructs.ConvertToNumpyArray(FPGEN.GetFingerprint(mol), arr)
        rows.append(arr)
    return np.vstack(rows)


def fit_gp(x, y):
    kernel = ConstantKernel(float(np.var(y)), constant_value_bounds="fixed") * Tanimoto()
    model = GaussianProcessRegressor(kernel=kernel, alpha=NOISE_VARIANCE, optimizer=None, normalize_y=False)
    model.fit(x, y - y.mean())
    model._y_mean = float(y.mean())
    return model


def metrics(y, pred):
    err = pred - y
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {"n": int(len(y)), "rmse": float(np.sqrt(np.mean(err ** 2))), "r2": float(1 - np.sum(err ** 2) / ss_tot)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(FOLDS)
    print("fingerprints", flush=True)
    x = fingerprints(data["SMILES"])
    y = data["PIC50"].to_numpy(dtype=float)
    names = data["SourceID"].tolist()
    metric_rows = []
    cluster_pred = None
    for split in ["cluster_fold", "scaffold_fold"]:
        print(split, flush=True)
        pred = np.full(len(y), np.nan)
        folds = data[split].to_numpy()
        for fold in sorted(pd.unique(folds)):
            test = folds == fold
            print(f"  fold {fold} train {(~test).sum()}", flush=True)
            model = fit_gp(x[~test], y[~test])
            pred[test] = model.predict(x[test]) + model._y_mean
        row = metrics(y, pred)
        row["split"] = split
        metric_rows.append(row)
        if split == "cluster_fold":
            cluster_pred = pred
    dev = data["USED_AS"].to_numpy() == "Dev-set"
    print("preset test", flush=True)
    model = fit_gp(x[dev], y[dev])
    preset = model.predict(x[~dev]) + model._y_mean
    row = metrics(y[~dev], preset)
    row["split"] = "preset_test"
    metric_rows.append(row)

    pairs = pd.read_csv(PAIRS)
    lookup = dict(zip(names, cluster_pred))
    left = pairs["id_i"].map(lookup).to_numpy()
    right = pairs["id_j"].map(lookup).to_numpy()
    predicted = np.abs(left - right)
    observed = pairs["abs_delta"].to_numpy()
    pair_row = {
        "n_pairs": int(len(pairs)),
        "delta_rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
    }
    for flag, threshold in [("one_log", 1.0), ("two_log", 2.0)]:
        truth = pairs[flag].to_numpy().astype(bool)
        pair_row[f"{flag}_recall"] = float(((truth) & (predicted >= threshold)).sum() / truth.sum())

    gp_metrics = pd.DataFrame(metric_rows)
    baseline = pd.read_csv(BASELINE)
    comparison = gp_metrics.merge(baseline, on="split", suffixes=("_gp", "_rf"))
    gp_metrics.to_csv(OUT / "regression_metrics.csv", index=False)
    comparison.to_csv(OUT / "comparison_with_baseline.csv", index=False)
    pd.DataFrame([pair_row]).to_csv(OUT / "pair_delta_metrics.csv", index=False)
    pd.DataFrame({"SourceID": names, "y_true": y, "y_pred_cluster": cluster_pred}).to_csv(
        OUT / "cluster_predictions.csv", index=False
    )
    print(comparison.to_string(index=False))
    print(pair_row)


if __name__ == "__main__":
    main()
