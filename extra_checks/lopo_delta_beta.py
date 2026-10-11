"""
lopo_delta_beta.py

Quick, focused LOPO check: does delta_beta_ratio generalize to unseen
patients as well as delta_gamma_ratio, given they looked similarly
strong in-sample but are highly correlated (r=0.84) and delta_beta lost
significance once delta_gamma was controlled for?

Reuses the same fixed-effects-only prediction approach as
leave_one_patient_out.py (see that script for the full explanation of
why this is the correct way to evaluate generalization to a new patient).

Usage:
    python lopo_delta_beta.py
"""

import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix
from scipy.special import expit
from sklearn.metrics import roc_auc_score, log_loss
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

DATA_PATH = "/Users/maya/Documents/lab_project/final_tables/more_ratios_log10.csv"
LIMIT_PATIENTS = None  # set to an int to spot-check quickly first

FORMULAS = {
    'delta_gamma': 'FBTCS_code ~ log_PZ_delta_gamma_ratio_z',  # for direct reference
    'delta_beta': 'FBTCS_code ~ log_PZ_delta_beta_ratio_z',
}


def fit_fixed_effects(formula, train_df):
    endog, exog = dmatrices(formula, data=train_df, return_type='dataframe')
    train_df_used = train_df.loc[exog.index]
    exog_re = pd.get_dummies(train_df_used['pat_num'], drop_first=False)
    ident = np.ones(exog_re.shape[1], dtype=int)
    model = BinomialBayesMixedGLM(endog, exog, exog_re, ident)
    result = model.fit_vb(verbose=False)
    return result.fe_mean, exog.design_info


def predict_held_out(fe_mean, design_info, test_df):
    X_test = dmatrix(design_info, test_df, return_type='dataframe')
    logits = X_test.values @ fe_mean
    return expit(logits), X_test.index


def run_lopo(df, formula, model_name):
    patients = df['pat_num'].unique()
    if LIMIT_PATIENTS:
        patients = patients[:LIMIT_PATIENTS]

    all_true, all_pred = [], []
    skipped = []

    for i, patient in enumerate(patients):
        train_df = df[df['pat_num'] != patient]
        test_df = df[df['pat_num'] == patient]
        if test_df.empty or train_df.empty:
            continue
        try:
            fe_mean, design_info = fit_fixed_effects(formula, train_df)
            probs, kept_index = predict_held_out(fe_mean, design_info, test_df)
        except Exception as e:
            skipped.append((patient, str(e)))
            continue

        test_df_used = test_df.loc[kept_index]
        all_true.extend(test_df_used['FBTCS_code'].tolist())
        all_pred.extend(probs.tolist())

        if (i + 1) % 20 == 0:
            print(f"  [{model_name}] {i + 1}/{len(patients)} patients done...")

    if skipped:
        print(f"  Skipped {len(skipped)} patients: {skipped[:5]}")

    return np.array(all_true), np.array(all_pred)


def main():
    df = pd.read_csv(DATA_PATH)
    df = df[df['vigilance'].str.lower() != 'unclear'].copy()
    print(f"N={len(df)} seizures, {df['pat_num'].nunique()} patients")

    summary = []
    for name, formula in FORMULAS.items():
        print(f"\n=== LOPO: {name} ===")
        y_true, y_pred = run_lopo(df, formula, name)
        auc = roc_auc_score(y_true, y_pred)
        ll = log_loss(y_true, y_pred)
        print(f"  Pooled N: {len(y_true)}  AUC: {auc:.3f}  Log-loss: {ll:.3f}")
        summary.append({'model': name, 'N': len(y_true), 'AUC': auc, 'log_loss': ll})

    print("\n=== SUMMARY ===")
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    print(
        "\nIf delta_beta's AUC is close to delta_gamma's, they're "
        "interchangeable in practice (expected, given r=0.84). If it's "
        "meaningfully lower, that's further evidence delta_gamma is the "
        "better-generalizing choice, not just the in-sample-favored one."
    )


if __name__ == "__main__":
    main()
