"""
topography_analysis.py

Tests: is the pre-ictal delta/gamma effect FOCAL (concentrated over the
patient's known seizure lobe and side) or GLOBAL (similar everywhere on
the scalp)?

Since final_output.csv only has patient-level 'lobe'/'side' (from
etiology), not each seizure's specific origin electrodes, this is a
lobe-level test, coarser than the per-seizure onset-topomap check we
did earlier -- but still directly answers the focal-vs-global question
the PI raised.

Method, per seizure:
  1. Only keep seizures with a known, specific lobe (frontal, temporal,
     parietal, occipital, or central) and known side (left/right) --
     drops 'other'/'unknown'/'hemisphere' cases where we can't define a
     matching electrode set.
  2. Split the 19 electrodes into "matching" (over the seizure's lobe,
     correct side, midline electrodes always included for lobes that
     have one) vs "non-matching" (everywhere else).
  3. Average each seizure's z-scored log delta/gamma ratio separately
     across matching vs non-matching electrodes.
  4. Compare matching vs non-matching:
       a. Paired comparison WITHIN FBTCS seizures only (is the ratio
          higher specifically at the seizure's own lobe?)
       b. Same paired comparison WITHIN non-FBTCS seizures, as a control
          (should show little/no such asymmetry if the effect is
          FBTCS-specific).
       c. Two simple mixed logistic models: FBTCS ~ matching_z and
          FBTCS ~ nonmatching_z, to compare which one predicts FBTCS
          better.

Usage:
    python topography_analysis.py
"""

import numpy as np
import pandas as pd
from scipy import stats as sstats
from patsy import dmatrices
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

DATA_PATH = "/Users/maya/Documents/lab_project/final_tables/topography_all_electrodes_log10.csv"

LOBE_ELECTRODES = {
    'frontal': {'FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'FZ'},
    'temporal': {'T7', 'T8', 'P7', 'P8'},
    'central': {'C3', 'C4', 'CZ'},
    'parietal': {'P3', 'P4', 'PZ'},
    'occipital': {'O1', 'O2'},
}
LEFT_ELECTRODES = {'FP1', 'F3', 'F7', 'C3', 'P3', 'P7', 'T7', 'O1'}
RIGHT_ELECTRODES = {'FP2', 'F4', 'F8', 'C4', 'P4', 'P8', 'T8', 'O2'}
MIDLINE_ELECTRODES = {'FZ', 'CZ', 'PZ'}
ALL_ELECTRODES = LEFT_ELECTRODES | RIGHT_ELECTRODES | MIDLINE_ELECTRODES


def matching_electrodes(lobe, side):
    lobe = str(lobe).strip().lower()
    side = str(side).strip().lower()
    if lobe not in LOBE_ELECTRODES or side not in ('left', 'right'):
        return None
    side_set = LEFT_ELECTRODES if side == 'left' else RIGHT_ELECTRODES
    return {e for e in LOBE_ELECTRODES[lobe] if e in MIDLINE_ELECTRODES or e in side_set}


def zcol(electrode):
    return f"log_{electrode}_delta_gamma_ratio_z"


def compute_matching_nonmatching(df):
    matching_vals, nonmatching_vals, keep_mask = [], [], []

    for _, row in df.iterrows():
        matches = matching_electrodes(row['lobe'], row['side'])
        if not matches:
            keep_mask.append(False)
            matching_vals.append(np.nan)
            nonmatching_vals.append(np.nan)
            continue

        nonmatches = ALL_ELECTRODES - matches
        m_vals = [row[zcol(e)] for e in matches if zcol(e) in df.columns]
        nm_vals = [row[zcol(e)] for e in nonmatches if zcol(e) in df.columns]

        if not m_vals or not nm_vals:
            keep_mask.append(False)
            matching_vals.append(np.nan)
            nonmatching_vals.append(np.nan)
            continue

        matching_vals.append(np.nanmean(m_vals))
        nonmatching_vals.append(np.nanmean(nm_vals))
        keep_mask.append(True)

    df = df.copy()
    df['matching_z'] = matching_vals
    df['nonmatching_z'] = nonmatching_vals
    return df[keep_mask].dropna(subset=['matching_z', 'nonmatching_z'])


def paired_comparison(df, label):
    diff = df['matching_z'] - df['nonmatching_z']
    stat, p = sstats.wilcoxon(df['matching_z'], df['nonmatching_z'])
    print(f"\n{label} (N={len(df)}):")
    print(f"  mean(matching_z)    = {df['matching_z'].mean():.3f}")
    print(f"  mean(nonmatching_z) = {df['nonmatching_z'].mean():.3f}")
    print(f"  mean difference     = {diff.mean():.3f}")
    print(f"  Wilcoxon signed-rank p = {p:.4f}")


def fit_model(formula, df, name):
    endog, exog = dmatrices(formula, data=df, return_type='dataframe')
    # dmatrices silently drops rows with NaN/inf in any formula variable;
    # exog_re must be built from the same surviving rows or the fit
    # crashes with a shape mismatch (see compare_ratios.py / chat).
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
        'p': 2 * (1 - sstats.norm.cdf(np.abs(result.fe_mean / result.fe_sd)))
    }).round(4)
    print(f"\n=== {name} (N={len(df_used)}) ===")
    print(coef_df.to_string(index=False))
    return coef_df


def main():
    df = pd.read_csv(DATA_PATH)
    n0 = len(df)
    df = df[df['vigilance'].astype(str).str.lower() != 'unclear'].copy()
    print(f"Excluded {n0 - len(df)} rows with vigilance == 'unclear' ({len(df)} remain).")
    df = compute_matching_nonmatching(df)
    print(f"Usable seizures (known specific lobe + known side): {len(df)}")

    fbtcs = df[df['FBTCS_code'] == 1]
    non_fbtcs = df[df['FBTCS_code'] == 0]

    print("\n--- Paired comparison: matching (own lobe) vs non-matching (rest of scalp) ---")
    paired_comparison(fbtcs, "FBTCS seizures")
    paired_comparison(non_fbtcs, "non-FBTCS seizures (control)")

    print("\n--- Which predicts FBTCS better: matching-region ratio, or the rest of the scalp? ---")
    fit_model('FBTCS_code ~ matching_z', df, 'matching_only')
    fit_model('FBTCS_code ~ nonmatching_z', df, 'nonmatching_only')
    fit_model('FBTCS_code ~ matching_z + nonmatching_z', df, 'both')

    print(
        "\nHow to read this:\n"
        "- If FBTCS seizures show matching_z clearly > nonmatching_z (and "
        "the Wilcoxon p is small), while non-FBTCS seizures show little/no "
        "such gap, that supports a FOCAL effect: specific to the seizure's "
        "own lobe/side.\n"
        "- If both FBTCS and non-FBTCS show a similar matching-vs-nonmatching "
        "gap, that gap is probably just a general electrode/lobe property, "
        "not something specific to FBTCS.\n"
        "- If 'nonmatching_only' predicts FBTCS about as well as "
        "'matching_only', that supports a GLOBAL effect rather than a "
        "focal one -- the ratio increase isn't specific to the seizure's "
        "own region."
    )


if __name__ == "__main__":
    main()