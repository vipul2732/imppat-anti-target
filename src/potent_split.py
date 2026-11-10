"""Keep close phytochemicals whose nearest measured neighbor is potent.

Potent means pIC50 or pEC50 of at least 5. For CYP the value is the maximum
across measured isoforms. Plant enrichment is then repeated on that subset,
still excluding compounds recorded from more than 20 plants.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMPPAT = ROOT / "input"
PAPER = ROOT / "results" / "paper"
TARGETS = ("hERG", "CYP", "PXR")
CUTOFF = 0.80
POTENT = 5.0
UBIQUITOUS = 20
N_DRAWS = 20000


def max_potency(text):
    labeled = re.findall(r":(\d+(?:\.\d+)?)", str(text))
    numbers = labeled or re.findall(r"\d+(?:\.\d+)?", str(text))
    values = [float(item) for item in numbers]
    return max(values) if values else np.nan


def main():
    similarity = pd.read_csv(PAPER / "imppat_similarity.csv")
    for name in TARGETS:
        similarity[f"potency_{name}"] = similarity[f"neighbor_potency_{name}"].map(max_potency)
        close = similarity[f"consensus_{name}"] >= CUTOFF
        potent = close & (similarity[f"potency_{name}"] >= POTENT)
        print(
            f"{name}: close {int(close.sum())} potent {int(potent.sum())} "
            f"median {similarity.loc[close, f'potency_{name}'].median():.2f}",
            flush=True,
        )

    catalog = pd.read_csv(
        IMPPAT / "imppat.csv",
        usecols=["IMPPAT Phytochemical identifier", "Indian medicinal plant"],
    ).rename(columns={"IMPPAT Phytochemical identifier": "imppat_id", "Indian medicinal plant": "plant"})
    scored = similarity.dropna(subset=["imppat_id"]).drop_duplicates("imppat_id")
    membership = catalog.merge(scored, on="imppat_id", how="inner").drop_duplicates(["plant", "imppat_id"])
    compounds = scored.set_index("imppat_id")
    rng = np.random.default_rng(0)
    rows = []
    for name in TARGETS:
        indicator = (
            (compounds[f"consensus_{name}"] >= CUTOFF)
            & (compounds[f"potency_{name}"] >= POTENT)
            & (compounds["n_plants"] <= UBIQUITOUS)
        )
        frame = pd.DataFrame({
            "plant": membership["plant"].to_numpy(),
            "hit": membership["imppat_id"].map(indicator).fillna(False).to_numpy(),
        })
        observed = frame.groupby("plant")["hit"].sum()
        sizes = frame.groupby("plant")["hit"].size()
        pool = indicator.reindex(membership["imppat_id"].unique()).fillna(False).to_numpy()
        total, population = int(pool.sum()), len(pool)
        tail = {
            int(size): rng.hypergeometric(total, population - total, int(size), size=N_DRAWS)
            for size in np.unique(sizes.to_numpy())
        }
        records = []
        for plant, count in observed.items():
            if count <= 0:
                continue
            draws = tail[int(sizes[plant])]
            records.append((plant, int(sizes[plant]), int(count), float(draws.mean()), float((draws >= count).mean())))
        result = pd.DataFrame(records, columns=["plant", "n_constituents", "n_close", "expected", "p_value"])
        result = result.sort_values("p_value")
        ranked = result["p_value"].to_numpy()
        q_values = ranked * len(result) / np.arange(1, len(result) + 1)
        result["q_value"] = np.minimum.accumulate(q_values[::-1])[::-1]
        result["anti_target"] = name
        rows.append(result)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(PAPER / "plant_enrichment_potent.csv", index=False)
    write_supplement(similarity, table)
    print(table.loc[table.q_value < 0.05].to_string(index=False), flush=True)


def write_supplement(similarity, enrichment):
    catalog = pd.read_csv(
        IMPPAT / "imppat.csv",
        usecols=["IMPPAT Phytochemical identifier", "Indian medicinal plant", "Plant part"],
    ).rename(columns={
        "IMPPAT Phytochemical identifier": "imppat_id",
        "Indian medicinal plant": "plant",
        "Plant part": "part",
    })
    # Plants named in the results. Each must also have q < 0.05 and at least
    # two potent constituents. Other plants can pass that test; they stay in
    # plant_enrichment_potent.csv and are not flagged in the supplement.
    named = {
        "hERG": {
            "Rauvolfia serpentina", "Rauvolfia tetraphylla", "Sinopodophyllum hexandrum",
            "Dysosma aurantiocaulis", "Cinchona calisaya", "Alangium salviifolium",
            "Carapichea ipecacuanha", "Croton tiglium", "Camellia sinensis",
        },
        "CYP": {"Sinopodophyllum hexandrum", "Dysosma aurantiocaulis"},
        "PXR": {"Catharanthus roseus"},
    }
    for name, plants in named.items():
        passed = set(enrichment.loc[
            (enrichment.anti_target == name) & (enrichment.q_value < 0.05) & (enrichment.n_close >= 2),
            "plant",
        ])
        named[name] = plants & passed
    rows = []
    for name in TARGETS:
        subset = similarity.loc[
            (similarity[f"consensus_{name}"] >= CUTOFF) & (similarity[f"potency_{name}"] >= POTENT)
        ].copy()
        for _, compound in subset.iterrows():
            links = catalog.loc[
                (catalog.imppat_id == compound.imppat_id) & (catalog.plant.isin(named[name]))
            ]
            plants = sorted(links.plant.dropna().unique())
            part_groups = []
            for plant in plants:
                parts = sorted({
                    str(part).strip()
                    for part in links.loc[links.plant == plant, "part"].dropna()
                    if str(part).strip()
                })
                part_groups.append(f"{plant}: {'; '.join(parts)}")
            rows.append({
                "anti_target": name,
                "phytochemical_name": compound.phytochemical_name,
                "imppat_id": compound.imppat_id,
                "n_plants_reporting_compound": int(compound.n_plants),
                "retained_for_plant_test": bool(compound.n_plants <= UBIQUITOUS),
                "in_table_plant": "; ".join(plants),
                "plant_parts": " | ".join(part_groups),
                "consensus_similarity": round(float(compound[f"consensus_{name}"]), 3),
                "neighbor_id": compound[f"neighbor_{name}"],
                "neighbor_potency": round(float(compound[f"potency_{name}"]), 2),
                "neighbor_potency_detail": compound[f"neighbor_potency_{name}"],
                "neighbor_ci": compound[f"neighbor_ci_{name}"],
            })
    out = pd.DataFrame(rows).sort_values(["anti_target", "phytochemical_name"])
    out.to_csv(ROOT / "supplementary_table_S3_compounds.csv", index=False)
    print(f"supplement rows {len(out)}", flush=True)


if __name__ == "__main__":
    main()
