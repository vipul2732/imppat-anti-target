"""Error of hERG pIC50 predictions against similarity to the training set.

Cluster-fold predictions are the saved out-of-fold models. For the preset
ToxTree split, the Gaussian process and the random forest are refit on the
8,380 development compounds and scored on the 499 test compounds.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel

from gp_herg import NOISE_VARIANCE, Tanimoto, fit_gp

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "results" / "splits" / "compound_folds.csv"
CLUSTER_PRED = ROOT / "results" / "gp" / "cluster_predictions.csv"
RF_PRED = ROOT / "results" / "baseline" / "oof_predictions.csv"
OUT = ROOT / "results" / "applicability"
BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprints(smiles):
    bitvects = []
    rows = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        arr = np.zeros((2048,), dtype=np.float64)
        if mol is None:
            bitvects.append(None)
        else:
            fp = FPGEN.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, arr)
            bitvects.append(fp)
        rows.append(arr)
    return bitvects, np.vstack(rows)


def max_similarity(query, library):
    scores = np.zeros(len(query))
    library = [fp for fp in library if fp is not None]
    for i, fp in enumerate(query):
        if fp is None or not library:
            scores[i] = np.nan
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, library)
        scores[i] = max(sims)
    return scores


def binned(similarity, y, pred, split, model):
    err = np.abs(pred - y)
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mask = (similarity >= lo) & (similarity < hi)
        if mask.sum() == 0:
            continue
        residual = pred[mask] - y[mask]
        rows.append(
            {
                "split": split,
                "model": model,
                "sim_lo": lo,
                "sim_hi": min(hi, 1.0),
                "n": int(mask.sum()),
                "rmse": float(np.sqrt(np.mean(residual ** 2))),
                "median_abs_error": float(np.median(err[mask])),
            }
        )
    return rows


def plot(table, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for (split, model), block in table.groupby(["split", "model"]):
        x = (block["sim_lo"] + block["sim_hi"]) / 2
        ax.plot(x, block["rmse"], marker="o", label=f"{model}, {split}")
    ax.set_xlabel("Tanimoto similarity to nearest training compound")
    ax.set_ylabel("RMSE of pIC50")
    ax.set_title("hERG prediction error by distance to the training set")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(FOLDS)
    print("fingerprints", flush=True)
    bitvects, x = fingerprints(data["SMILES"])
    y = data["PIC50"].to_numpy(dtype=float)
    folds = data["cluster_fold"].to_numpy()
    dev = data["USED_AS"].to_numpy() == "Dev-set"

    gp = pd.read_csv(CLUSTER_PRED).set_index("SourceID")["y_pred_cluster"]
    rf = pd.read_csv(RF_PRED)
    rf = rf.loc[rf["split"] == "cluster_fold"].set_index("SourceID")["y_pred"]
    gp_cluster = data["SourceID"].map(gp).to_numpy()
    rf_cluster = data["SourceID"].map(rf).to_numpy()

    sim_cluster = np.zeros(len(data))
    for fold in sorted(np.unique(folds)):
        test = np.flatnonzero(folds == fold)
        train = [bitvects[i] for i in np.flatnonzero(folds != fold)]
        sim_cluster[test] = max_similarity([bitvects[i] for i in test], train)
        print(f"cluster similarity fold {fold}", flush=True)

    print("preset models", flush=True)
    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=0
    )
    rf_model.fit(x[dev], y[dev])
    gp_model = fit_gp(x[dev], y[dev])
    test = np.flatnonzero(~dev)
    rf_test = rf_model.predict(x[test])
    gp_test = gp_model.predict(x[test]) + gp_model._y_mean
    sim_test = max_similarity([bitvects[i] for i in test], [bitvects[i] for i in np.flatnonzero(dev)])

    rows = []
    rows += binned(sim_cluster, y, gp_cluster, "cluster_holdout", "gaussian_process")
    rows += binned(sim_cluster, y, rf_cluster, "cluster_holdout", "random_forest")
    rows += binned(sim_test, y[test], gp_test, "preset_test", "gaussian_process")
    rows += binned(sim_test, y[test], rf_test, "preset_test", "random_forest")
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "error_by_similarity.csv", index=False)
    plot(table, OUT / "error_by_similarity.png")
    print(table.to_string(index=False))
    print("preset test max similarity median", float(np.median(sim_test)))
    print("cluster holdout max similarity median", float(np.median(sim_cluster)))
    print("cluster holdout fraction with training neighbor >= 0.80", float(np.mean(sim_cluster >= 0.80)))


if __name__ == "__main__":
    main()
