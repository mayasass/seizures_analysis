"""
patient_normalized_analysis.py

Compares three versions of the delta/gamma ratio for predicting FBTCS:
  1. population_z: the existing approach (compare each seizure to
     everyone).
  2. patient_z: compare each seizure to that PATIENT's own other
     seizures (removes patient-level scale differences).
  3. patient_stage_z: compare each seizure to that patient's other
     seizures in the SAME sleep stage (removes patient-level AND
     stage-level scale differences).

Sample sizes will differ across the three, since (2) and (3) can only
use seizures where a valid leave-one-out baseline exists (i.e., the
patient -- or patient+stage combo -- has more than one seizure). This
script reports each model on its own valid subset, and separately
reports (1) restricted to the SAME subset as (2)/(3), so the comparison
isn't confounded by different samples.

NOTE: since (2) and (3) already remove between-patient variation by
construction, we don't expect the patient random effect to add much on
top -- but it's kept in the model for consistency with the rest of the
project's modeling approach.

Usage:
    python patient_normalized_analysis.py
"""

import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy import stats
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

DATA_PATH = "/Users/maya/Documents/lab_project/final_tables/patient_normalized.csv"


def fit_and_summarize(formula, df, name):
    endog, exog = dmatrices(formula, data=df, return_type='dataframe')
    df_used = df.loc[exog.index]
    exog_re = pd.get_dummies(df_used['pat_num'], drop_first=False)
    n_dropped = len(df) - len(df_used)
    if n_dropped > 0:
        print(f"  NOTE: {n_dropped} row(s) dropped due to NaN/inf in the formula's variables.")
    ident = np.ones(exog_re.shape[1], dtype=int)
    model = BinomialBayesMixedGLM(endog, exog, exog_re, ident)
    result = model.fit_vb(verbose=False)
    coef_df = pd.DataFrame({
        'name': result.model.exog_names,
        'beta': result.fe_mean,
        'SD': result.fe_sd,
        'OR': np.exp(result.fe_mean),
        'z': result.fe_mean / result.fe_sd,
        'p': 2 * (1 - stats.norm.cdf(np.abs(result.fe_mean / result.fe_sd)))
    }).round(4)
    print(f"\n=== {name} (N={len(df_used)}) ===")
    print(f"formula: {formula}")
    print(coef_df.to_string(index=False))
    return coef_df


def zscore(series):
    return (series - series.mean()) / series.std()


def main():
    df = pd.read_csv(DATA_PATH)

    print(f"Total seizures: {len(df)}")
    print(f"Valid patient-normalized baseline: {df['patient_centered_z'].notna().sum()}")
    print(f"Valid patient+stage-normalized baseline: {df['patient_stage_centered_z'].notna().sum()}")

    # --- Model 1: population z-score, on the FULL sample (reference point) ---
    df['population_z'] = zscore(df['log_PZ_delta_gamma_ratio'])
    fit_and_summarize('FBTCS_code ~ population_z', df, 'population_full_sample')

    # --- Model 2: patient-normalized, on its own valid subset ---
    df_patient = df[df['patient_centered_z'].notna()].copy()
    fit_and_summarize('FBTCS_code ~ patient_centered_z', df_patient, 'patient_normalized')

    # --- Model 1b: population z-score, restricted to the SAME subset as Model 2 ---
    # (fair comparison -- otherwise any difference could just be sample composition)
    df_patient['population_z_matched'] = zscore(df_patient['log_PZ_delta_gamma_ratio'])
    fit_and_summarize('FBTCS_code ~ population_z_matched', df_patient, 'population_matched_to_patient_subset')

    # --- Model 3: patient+stage-normalized, on its own valid subset ---
    df_stage = df[df['patient_stage_centered_z'].notna()].copy()
    fit_and_summarize('FBTCS_code ~ patient_stage_centered_z', df_stage, 'patient_stage_normalized')

    # --- Model 1c: population z-score, restricted to the SAME subset as Model 3 ---
    df_stage['population_z_matched'] = zscore(df_stage['log_PZ_delta_gamma_ratio'])
    fit_and_summarize('FBTCS_code ~ population_z_matched', df_stage, 'population_matched_to_stage_subset')

    print(
        "\nHow to read this:\n"
        "- Always compare a normalized model to the POPULATION model on the "
        "SAME subset (e.g. 'patient_normalized' vs "
        "'population_matched_to_patient_subset'), not to the full-sample "
        "population model -- otherwise you can't tell whether a difference "
        "is due to the normalization or just a smaller/different sample.\n"
        "- If patient/stage normalization has a STRONGER effect (bigger OR, "
        "smaller p) than population z-scoring on the same subset, that "
        "supports the idea that within-patient variation carries the real "
        "signal, and cross-patient scale differences were adding noise.\n"
        "- If normalization doesn't improve things (or the effect gets "
        "weaker/disappears), the population z-score combined with the "
        "model's existing patient random effect was already doing an "
        "adequate job, and this extra step isn't needed."
    )


if __name__ == "__main__":
    main()
