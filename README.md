# Pre-ictal delta/gamma ratio and focal-to-bilateral tonic-clonic seizures (FBTCS)

Code for the analysis of whether the scalp-EEG delta/gamma ratio in the 60 s before focal seizure onset
predicts focal-to-bilateral tonic-clonic seizures (FBTCS), using the EPILEPSIAE database.

Maya Sasson · CPL lab, Ben-Gurion University of the Negev
(PI: Prof. Oren Shriki; co-supervisor: Dr. Miriam Guendelman)

**No data are included.** EPILEPSIAE data are available from the database owners under their own terms.

## Pipeline

### 1. Feature extraction (run on the machine that holds the raw EEG)

| Script | What it does |
|---|---|
| `processing/data_processing.py` | For every seizure: reads the scalp EEG, cuts the 60 s before the annotated focal onset, keeps the 19 standard 10–20 electrodes, band-pass filters 0.5–40 Hz, re-references to the common average, computes Welch PSD and relative band power (delta 0.5–4, sigma 12–16, beta 16–30, gamma 30–40 Hz) and band ratios per channel. Output: `final_output.csv` (one row per seizure). |

### 2. Analysis (run on `final_output.csv`)

Everything below runs in the right order with:

```
python run_all.py                      # whole pipeline
python run_all.py topography figures   # only some steps
```

Paths to the data tables are set at the top of each script. CSV results are written to `results/`,
figures to `figures/` and a log of every run to `run_logs/` (all three are not tracked by git).

## Folder layout

```
run_all.py          runs the analysis pipeline in order
processing/         lab-computer step: raw EEG -> final_output.csv (+ onset/topography check)
analysis/           analysis scripts run by run_all.py
figures_code/       figure scripts run by run_all.py (Figures 1-8)
extra_checks/       one-off checks used during the analysis
archive/            earlier exploration scripts, not part of the final pipeline
results/  figures/  run_logs/   outputs (not in git)
```

| Step (`run_all.py` group) | Script(s) | What it does |
|---|---|---|
| `tables` | `analysis/statistical_anaylsis_updated.py` | Outlier exclusion (top 5% of each of the 8 ratio features), log10(x+1) transform; writes the analysis table (1,036 seizures; 1,029 after removing "unclear" vigilance). |
| `poster_figs` | `figures_code/poster_figures_updated.py` | Figures 1–5 (+ optional 1b, 4b, 5b). |
| `decomposition` | `delta_gamma_decomposition.py` | Ratio vs. delta alone vs. gamma alone vs. delta + gamma (Bayesian mixed logistic regression, random intercept per patient). |
| `lopo` | `leave_one_patient_out.py` (run twice) | Leave-one-patient-out validation (AUC). |
| `normalization` | `prepare_patient_normalized.py`, `patient_normalized_analysis.py` | Per-patient and per-patient-and-stage normalization of the ratio. |
| `topography` | `prepare_topography_data.py`, `topography_analysis.py` | Seizure's own lobe vs. rest of scalp. |
| `ratios` | `prepare_more_ratios.py`, `compare_ratios.py` | Delta/beta and delta/total power vs. delta/gamma. |
| `figures` | `figures_code/followup_figures.py` | Figures 6–8. |
| `vigilance` | `vigilance_covariate_models.py` | Vigilance state as covariate and as interaction; Figure 9. |

All other analysis scripts in the table are in `analysis/`.

## Requirements

Python 3 with `numpy`, `pandas`, `scipy`, `statsmodels`, `patsy`, `scikit-learn`, `matplotlib`, `seaborn`; `mne` for `data_processing.py`.
