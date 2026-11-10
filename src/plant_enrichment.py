"""Which plants carry close analogs of measured anti-target compounds.

Computes the enrichment table only; figures.py draws Figure 5.

A plant's score is the number of its constituents that are close at 0.80
under the two-fingerprint rule. The null keeps each plant's constituent
count fixed and reshuffles which compounds it has, so a plant with many
recorded phytochemicals is not enriched by size alone. Compounds reported
from many plants are also reported separately, because they carry no
information about a particular source.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMPPAT = ROOT / "input"
PAPER = ROOT / "results" / "paper"
OUT = PAPER
TARGETS = ("hERG", "CYP", "PXR")
CUTOFF = 0.80
N_DRAWS = 20000
UBIQUITOUS = 20


def main():
    similarity = pd.read_csv(PAPER / "imppat_similarity.csv")
    catalog = pd.read_csv(
        IMPPAT / "imppat.csv",
        usecols=["IMPPAT Phytochemical identifier", "Indian medicinal plant", "Plant part"],
    )
    catalog = catalog.rename(
        columns={"IMPPAT Phytochemical identifier": "imppat_id", "Indian medicinal plant": "plant", "Plant part": "part"}
    )
    scored = similarity.loc[similarity["imppat_id"].notna(), ["imppat_id"] + [f"consensus_{name}" for name in TARGETS] + ["n_plants", "phytochemical_name"]]
    scored = scored.drop_duplicates("imppat_id")
    links = catalog.merge(scored, on="imppat_id", how="inner")
    membership = links.drop_duplicates(["plant", "imppat_id"])
    print(f"plants {membership['plant'].nunique()} compounds {membership['imppat_id'].nunique()} links {len(membership)}", flush=True)

    compounds = scored.set_index("imppat_id")
    rng = np.random.default_rng(0)
    rows = []
    for name in TARGETS:
        flag = (compounds[f"consensus_{name}"] >= CUTOFF)
        restricted = flag & (compounds["n_plants"] <= UBIQUITOUS)
        for label, indicator in (("all", flag), ("restricted", restricted)):
            values = membership["imppat_id"].map(indicator).fillna(False).to_numpy()
            frame = pd.DataFrame({"plant": membership["plant"].to_numpy(), "hit": values})
            observed = frame.groupby("plant")["hit"].sum()
            sizes = frame.groupby("plant")["hit"].size()
            pool = indicator.reindex(membership["imppat_id"].unique()).fillna(False).to_numpy()
            total = int(pool.sum())
            population = len(pool)
            # exact hypergeometric tail by simulation over the shared compound pool
            unique_sizes = np.unique(sizes.to_numpy())
            tail = {}
            for size in unique_sizes:
                draws = rng.hypergeometric(total, population - total, size, size=N_DRAWS)
                tail[size] = draws
            pvalues = []
            expected = []
            for plant, count in observed.items():
                size = int(sizes[plant])
                draws = tail[size]
                pvalues.append(float((draws >= count).mean()))
                expected.append(float(draws.mean()))
            result = pd.DataFrame({
                "plant": observed.index,
                "n_constituents": sizes.reindex(observed.index).to_numpy(),
                "n_close": observed.to_numpy(),
                "expected": expected,
                "p_value": pvalues,
            })
            result = result.loc[result["n_close"] > 0].copy()
            result = result.sort_values("p_value")
            m = len(result)
            result["q_value"] = np.minimum.accumulate((result["p_value"].to_numpy() * m / np.arange(1, m + 1))[::-1])[::-1] if m else []
            result["anti_target"] = name
            result["compound_set"] = label
            rows.append(result)
            significant = int((result["q_value"] < 0.05).sum())
            print(f"{name} {label}: close compounds {total}, plants with a hit {len(result)}, q<0.05 {significant}", flush=True)

    table = pd.concat(rows, ignore_index=True)
    table.to_csv(OUT / "plant_enrichment.csv", index=False)


    detail = links.loc[
        (links[[f"consensus_{name}" for name in TARGETS]] >= CUTOFF).any(axis=1)
    ].copy()
    detail = detail.sort_values(["plant", "phytochemical_name"])
    detail.to_csv(OUT / "plant_close_compounds.csv", index=False)
    print(table.loc[table.compound_set == "restricted"].sort_values("q_value").head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
