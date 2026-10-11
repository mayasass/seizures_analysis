"""
delta_only_uc_stratified_check.py

Follow-up to the UC_included vs UC_excluded LOPO comparison: is
delta_only's AUC collapse (0.570 -> 0.524) driven by a subset of
patients who lost MOST of their seizures when UC rows were dropped
(their random-effect / personal signal becomes unreliable with very
little data left), or is it a broad effect across most patients?

Approach:
  1. Using the FULL (UC-included) dataset, compute each patient's
     fraction of seizures that are UC.
  2. Flag "majority-UC" patients: >50% of their seizures were UC (so
     excluding UC removes most of what we know about them).
  3. Run LOPO for delta_only on the UC-EXCLUDED dataset (same as the
     EXCLUDE_UC=True run), but stratify the pooled held-out predictions
     by whether the held-out patient was majority-UC in the original
     data.
  4. Report AUC separately for majority-UC patients vs. the rest.

If delta_only's AUC is much worse specifically among majority-UC
patients, that points to a data-thinning artifact for that subgroup
rather than a genuine reversal of delta's effect. If it's similarly
weak across both groups, the collapse is a broader, more real finding.

Prerequisite: same clean_final_output_log10.csv as the other scripts
(needs log_PZ_delta, FBTCS_code, pat_num, classif., vigilance).

Usage:
    python delta_only_uc_stratified_check.py
"""

import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

DATA_PATH = '/Users/maya/Documents/lab_project/final_tables/clean_final_output_log10.csv'
LIMIT_PATIENTS = None  # set to an int to spot-check quickly first
FORMULA = 'FBTCS_code ~ log_PZ_delta_z'


def zscore(series):
    return (series - series.mean()) / series.std()


def fit_fixed_effects(formula, train_df):
    endog, exog = dmatrices(formula, data=train_df, return_type='dataframe')
    exog_re = pd.get_dummies(train_df['pat_num'], drop_first=False)
    ident = np.ones(exog_re.shape[1], dtype=int)
    model = BinomialBayesMixedGLM(endog, exog, exog_re, ident)
    result = model.fit_vb(verbose=False)
    return result.fe_mean, exog.design_info


def predict_held_out(fe_mean, design_info, test_df):
    X_test = dmatrix(design_info, test_df, return_type='dataframe')
    return expit(X_test.values @ fe_mean)


def main():
    df_full = pd.read_csv(DATA_PATH)
    df_full = df_full[df_full['vigilance'].str.lower() != 'unclear'].copy()

    # --- Step 1-2: compute per-patient UC fraction on the pre-exclusion data ---
    uc_frac = (
        df_full.assign(is_uc=(df_full['classif.'] == 'UC').astype(int))
        .groupby('pat_num')['is_uc']
        .mean()
    )
    majority_uc_patients = set(uc_frac[uc_frac > 0.5].index)
    fully_uc_patients = set(uc_frac[uc_frac == 1.0].index)

    print(f"Patients with >50% UC seizures (majority-UC): {len(majority_uc_patients)}")
    print(f"Patients with 100% UC seizures (fully dropped when excluding UC): {len(fully_uc_patients)}")
    print(f"Total patients (pre-exclusion): {df_full['pat_num'].nunique()}")

    # --- Step 3: build the UC-excluded analysis dataset, same as EXCLUDE_UC=True ---
    df = df_full[df_full['classif.'] != 'UC'].copy()
    df['log_PZ_delta_z'] = zscore(df['log_PZ_delta'])

    n_lost_patients = df_full['pat_num'].nunique() - df['pat_num'].nunique()
    print(f"Patients that disappear entirely from the UC-excluded analysis: {n_lost_patients}")
    print(f"(these are exactly the fully-UC patients, and can't be in this LOPO at all)\n")

    patients = df['pat_num'].unique()
    if LIMIT_PATIENTS:
        patients = patients[:LIMIT_PATIENTS]

    all_true, all_pred, all_pat = [], [], []
    for i, patient in enumerate(patients):
        train_df = df[df['pat_num'] != patient]
        test_df = df[df['pat_num'] == patient]
        if test_df.empty or train_df.empty:
            continue
        try:
            fe_mean, design_info = fit_fixed_effects(FORMULA, train_df)
            probs = predict_held_out(fe_mean, design_info, test_df)
        except Exception as e:
            print(f"  Skipped patient {patient}: {e}")
            continue

        all_true.extend(test_df['FBTCS_code'].tolist())
        all_pred.extend(probs.tolist())
        all_pat.extend([patient] * len(test_df))

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(patients)} patients done...")

    results = pd.DataFrame({'pat_num': all_pat, 'y_true': all_true, 'y_pred': all_pred})
    results['was_majority_uc'] = results['pat_num'].isin(majority_uc_patients)

    print(f"\nPooled held-out N: {len(results)}")
    print(f"  From majority-UC patients: {results['was_majority_uc'].sum()}")
    print(f"  From other patients:       {(~results['was_majority_uc']).sum()}")

    def safe_auc(sub):
        if sub['y_true'].nunique() < 2:
            return None
        return roc_auc_score(sub['y_true'], sub['y_pred'])

    overall_auc = safe_auc(results)
    majority_auc = safe_auc(results[results['was_majority_uc']])
    other_auc = safe_auc(results[~results['was_majority_uc']])

    print(f"\n=== delta_only AUC, stratified by original UC burden ===")
    print(f"Overall (matches the main LOPO run):     {overall_auc}")
    print(f"Majority-UC patients only:                {majority_auc}")
    print(f"Other patients only:                      {other_auc}")

    print(
        "\nHow to read this:\n"
        "- If 'majority-UC patients only' AUC is much worse than 'other "
        "patients only', the collapse is concentrated in patients who lost "
        "most of their data -- likely a data-thinning artifact, not "
        "necessarily evidence delta's effect reverses.\n"
        "- If both subgroups show similarly weak AUC (~0.5), the collapse "
        "is broad and more likely a genuine finding: delta alone doesn't "
        "generalize once ambiguous UC seizures are removed, regardless of "
        "which patients you look at."
    )


if __name__ == "__main__":
    main()
