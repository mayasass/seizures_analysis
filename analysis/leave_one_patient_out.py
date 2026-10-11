"""
leave_one_patient_out.py

Leave-one-patient-out (LOPO) cross-validation for the FBTCS prediction
models (ratio_only, delta_and_gamma, delta_only, gamma_only).

WHY this matters beyond the p-values we already have:
A model can look significant in-sample and still generalize poorly to a
genuinely new patient -- especially with a random intercept per patient,
which can let the model "memorize" patient-specific baselines rather than
learning a real population-level relationship. LOPO tests exactly that:
for each patient, refit the model on everyone else, then predict that
patient's seizures using ONLY the fixed-effect (population-level)
coefficients -- since a truly new patient has no estimated random effect
yet. This mirrors what you'd need at deployment/generalization time.

Metric: pooled AUC (and log-loss) across all held-out predictions from
all folds combined, plus per-model comparison.

Runtime note: this fits one Bayesian mixed GLM per patient per model
formula (~150+ patients x 4 formulas = 600+ fits). Expect this to take
a while -- start with LIMIT_PATIENTS set to a small number to sanity
check it runs before doing the full leave-one-out loop.

Prerequisite: same clean_final_output_log10.csv produced by the updated
statistical_anaylsis.py (needs log_PZ_delta, log_PZ_gamma,
log_PZ_delta_gamma_ratio, FBTCS_code, pat_num, classif., vigilance).

Usage:
    python leave_one_patient_out.py
"""

import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix
from scipy.special import expit
from sklearn.metrics import roc_auc_score, log_loss
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

DATA_PATH = '/Users/maya/Documents/lab_project/final_tables/clean_final_output_log10.csv'

# Set to an integer (e.g. 10) to spot-check quickly on a subset of
# patients before running the full leave-one-out loop. None = all patients.
LIMIT_PATIENTS = None

# Toggle to match whichever UC handling you're currently comparing
# (see chat discussion -- run this script once per setting if you want
# both, same as the decomposition script's UC_included / UC_excluded).
EXCLUDE_UC = False
# run_all.py sets EXCLUDE_UC=1 in the environment for the second run, so you no
# longer need to edit the line above by hand (editing it still works too).
import os
if os.environ.get('EXCLUDE_UC') == '1':
    EXCLUDE_UC = True

FORMULAS = {
    'ratio_only': 'FBTCS_code ~ log_PZ_delta_gamma_ratio_z',
    'delta_and_gamma': 'FBTCS_code ~ log_PZ_delta_z + log_PZ_gamma_z',
    'delta_only': 'FBTCS_code ~ log_PZ_delta_z',
    'gamma_only': 'FBTCS_code ~ log_PZ_gamma_z',
}


def zscore(series):
    return (series - series.mean()) / series.std()


def prepare_data():
    df = pd.read_csv(DATA_PATH)

    required = ['log_PZ_delta', 'log_PZ_gamma', 'log_PZ_delta_gamma_ratio',
                'FBTCS_code', 'pat_num', 'classif.', 'vigilance']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df['log_PZ_delta_z'] = zscore(df['log_PZ_delta'])
    df['log_PZ_gamma_z'] = zscore(df['log_PZ_gamma'])
    df['log_PZ_delta_gamma_ratio_z'] = zscore(df['log_PZ_delta_gamma_ratio'])

    df = df[df['vigilance'].str.lower() != 'unclear'].copy()
    if EXCLUDE_UC:
        df = df[df['classif.'] != 'UC'].copy()

    return df


def fit_fixed_effects(formula, train_df):
    """Fit the mixed model on the training fold and return the fixed-effect
    coefficients plus the training design info (needed to build a matching
    design matrix for the held-out patient's data)."""
    endog, exog = dmatrices(formula, data=train_df, return_type='dataframe')
    # dmatrices silently drops rows with NaN/inf in any formula variable;
    # exog_re must be built from the same surviving rows or the fit
    # crashes with a shape mismatch (see compare_ratios.py / chat).
    train_df_used = train_df.loc[exog.index]
    exog_re = pd.get_dummies(train_df_used['pat_num'], drop_first=False)
    ident = np.ones(exog_re.shape[1], dtype=int)

    model = BinomialBayesMixedGLM(endog, exog, exog_re, ident)
    result = model.fit_vb(verbose=False)

    return result.fe_mean, exog.design_info


def predict_held_out(fe_mean, design_info, test_df):
    """Predict probabilities for the held-out patient using ONLY the
    fixed effects -- a new patient has no estimated random intercept, so
    this is the correct way to generalize the model to unseen patients.
    Rows with NaN/inf in the predictor(s) are dropped here too, for the
    same reason as in fit_fixed_effects."""
    X_test = dmatrix(design_info, test_df, return_type='dataframe')
    logits = X_test.values @ fe_mean
    # Slope-only score: the same linear predictor WITHOUT the intercept. Within one
    # held-out patient the ranking is identical, but pooled across folds it is not
    # contaminated by tiny fold-to-fold intercept shifts (see main() comment).
    non_int = [i for i, n in enumerate(design_info.column_names) if n != 'Intercept']
    score = X_test.values[:, non_int] @ np.asarray(fe_mean)[non_int]
    return expit(logits), X_test.index, score


def run_lopo(df, formula, model_name):
    patients = df['pat_num'].unique()
    if LIMIT_PATIENTS:
        patients = patients[:LIMIT_PATIENTS]

    all_true, all_pred, all_pat, all_score = [], [], [], []
    fold_coefs = []
    skipped = []

    for i, patient in enumerate(patients):
        train_df = df[df['pat_num'] != patient]
        test_df = df[df['pat_num'] == patient]

        if test_df.empty or train_df.empty:
            skipped.append((patient, "empty train/test split"))
            continue

        try:
            fe_mean, design_info = fit_fixed_effects(formula, train_df)
            probs, kept_index, score = predict_held_out(fe_mean, design_info, test_df)
            fold_coefs.append(np.asarray(fe_mean, dtype=float))
        except Exception as e:
            skipped.append((patient, str(e)))
            continue

        test_df_used = test_df.loc[kept_index]
        all_true.extend(test_df_used['FBTCS_code'].tolist())
        all_pred.extend(probs.tolist())
        all_score.extend(np.asarray(score).tolist())
        all_pat.extend([patient] * len(test_df_used))

        if (i + 1) % 20 == 0:
            print(f"  [{model_name}] {i + 1}/{len(patients)} patients done...")

    if skipped:
        print(f"  Skipped {len(skipped)} patients (errors or empty folds):")
        for pat, reason in skipped[:5]:
            print(f"    {pat}: {reason}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

    return np.array(all_true), np.array(all_pred), all_pat, np.array(all_score), np.array(fold_coefs)


def main():
    df = prepare_data()
    print(f"N={len(df)} seizures, {df['pat_num'].nunique()} patients "
          f"(EXCLUDE_UC={EXCLUDE_UC})")
    base_rate = df['FBTCS_code'].mean()
    base_ll = log_loss(df['FBTCS_code'], np.full(len(df), base_rate))
    print(f"FBTCS base rate {base_rate:.3f}; log-loss of ALWAYS predicting the base rate = {base_ll:.3f} "
          f"(a model must beat THIS number to be useful on log-loss)")

    summary = []
    for model_name, formula in FORMULAS.items():
        print(f"\n=== LOPO: {model_name} ===")
        print(f"formula: {formula}")

        y_true, y_pred, _, y_score, coefs = run_lopo(df, formula, model_name)

        if len(np.unique(y_true)) < 2:
            print("  Can't compute AUC: only one class present in pooled held-out labels.")
            continue

        auc = roc_auc_score(y_true, y_pred)
        auc_slope = roc_auc_score(y_true, y_score)
        ll = log_loss(y_true, y_pred)
        n_used = len(y_true)

        # Fold diagnostics: did any fit diverge?
        bad = (~np.isfinite(coefs)).any(axis=1) | (np.abs(coefs) > 5).any(axis=1)
        print(f"  Pooled held-out N: {n_used}")
        print(f"  AUC (pooled predictions):          {auc:.3f}")
        print(f"  AUC (slope-only score, no intercept): {auc_slope:.3f}")
        print(f"  Log-loss: {ll:.3f}  (base-rate baseline {base_ll:.3f})")
        print(f"  Fold fits: {len(coefs)}; suspicious (non-finite or |coef|>5): {int(bad.sum())}")
        print(f"  Intercept across folds: min {np.nanmin(coefs[:, 0]):.3f}, max {np.nanmax(coefs[:, 0]):.3f}; "
              f"slope(s) median {np.round(np.nanmedian(coefs[:, 1:], axis=0), 3)}, "
              f"range {np.round(np.nanmin(coefs[:, 1:], axis=0), 3)} to {np.round(np.nanmax(coefs[:, 1:], axis=0), 3)}")

        summary.append({'model': model_name, 'N': n_used, 'AUC': auc, 'AUC_slope_only': auc_slope,
                        'log_loss': ll, 'baseline_log_loss': base_ll, 'bad_folds': int(bad.sum())})

    summary_df = pd.DataFrame(summary).round(3)
    print("\n\n=== SUMMARY: which model generalizes best to unseen patients? ===")
    print(summary_df.to_string(index=False))
    out_name = f"lopo_summary_{'UC_excluded' if EXCLUDE_UC else 'UC_included'}.csv"
    summary_df.to_csv(out_name, index=False)
    print(f"(saved {out_name})")

    print(
        "\nHow to read this:\n"
        "- Higher AUC (max 1.0, 0.5 = chance) = better at ranking FBTCS vs "
        "non-FBTCS seizures in patients the model has never seen.\n"
        "- Lower log-loss = better-calibrated probabilities, but compare it with the base-rate baseline printed above.\n"
        "- If 'AUC (pooled predictions)' is far below 'AUC (slope-only)', the gap comes from tiny fold-to-fold intercept "
        "shifts (a held-out patient with many FBTCS lowers the training intercept) -- a known leave-one-out artifact for "
        "weak predictors, not a real reversal. Trust the slope-only AUC for near-null models.\n"
        "- If 'ratio_only' has the best AUC here too (not just the best "
        "p-value in-sample), that's stronger evidence it's a genuinely "
        "useful summary, not just a good in-sample fit.\n"
        "- If a simpler model (e.g. delta_only) matches or beats the ratio "
        "here, that favors reporting the simpler model as the headline "
        "result."
    )


if __name__ == "__main__":
    main()