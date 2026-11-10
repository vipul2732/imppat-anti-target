# IMPPAT anti-target chemical-space coverage

Code, input tables, and result tables for the analysis of how Indian medicinal-plant phytochemicals in IMPPAT overlap the measured chemical space of hERG, CYP enzymes, and PXR.

## Run

Open `reproduce.ipynb` from this folder and run it from the top. It calls the scripts in `src/` in the order of the methods and writes the tables and analysis figures to `results/paper/`.

Requirements: Python 3.11 with RDKit, pandas, numpy, scikit-learn, and matplotlib. The similarity search is the slow step.

Inputs, already in `input/`:

- `imppat.csv`, the public IMPPAT plant–phytochemical associations
- `sdf_files/`, structures retrieved from PubChem by phytochemical name
- `failed_pubchem_downloads.csv`, names PubChem did not return
- `hERG_Dataset.csv`, the ToxTree hERG set
- `cyp-challenge-TRAIN_inhibition.csv` and `pxr-challenge_TRAIN.csv`, the OpenADMET challenge sets
- the unblinded PXR test tables

## Figure S1

The hERG prediction-error curve is produced by the scripts in `herg_cliffs/src/` from the files in `herg_cliffs/`. The plotted values are in `herg_cliffs/results/applicability/error_by_similarity.csv`. `reproduce.ipynb` does not redraw that figure.

## Sources

IMPPAT associations are from the public IMPPAT release. The hERG table is the ToxTree set of Arab et al. The CYP and PXR tables are the public OpenADMET challenge sets.
