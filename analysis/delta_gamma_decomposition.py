"""
delta_gamma_decomposition.py

Answers: "is it really the delta/gamma RATIO that predicts FBTCS, or is
one of the two bands doing all the work?"

Fits three Bayesian mixed-effects logistic models (same structure as
stats.py's existing models -- random intercept per patient) on the PZ
electrode:
    1. Ratio only:      FBTCS_code ~ log_PZ_delta_gamma_ratio_z
    2. Delta+gamma:      FBTCS_code ~ log_PZ_delta_z + log_PZ_gamma_z
    3. Delta only / gamma only (for reference)

Also reports the correlation between log_PZ_delta_z and log_PZ_gamma_z,
since if they're highly correlated, model 2's individual coefficients
become unstable and harder to trust in isolation (see chat discussion).

Prerequisite: run the UPDATED statistical_anaylsis.py first (the one that
also keeps and log-transforms raw PZ_delta / PZ_gamma, not just the
ratio columns), so clean_final_output_log10.csv has:
    log_PZ_delta, log_PZ_gamma, log_PZ_delta_gamma_ratio

Usage:
    python delta_gamma_decomposition.py
"""

import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy import stats
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

DATA_PATH = '/Users/maya/Documents/lab_project/final_tables/clean_final_output_log10.csv'


def zscore(series):
    return (series - series.mean()) / series.std()


def fit_and_summarize(formula, df, name):
    """Same pattern as stats.py's model_single_feature/summary_table,
    pulled out standalone so this script doesn't depend on importing the
    rest of stats.py."""
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
        'p': 2 * (1 - stats.norm.cdf(np.abs(result.fe_mean / result.fe_sd)))
    }).round(4)

    print(f"\n=== {name} (N={len(df_used)}) ===")
    print(f"formula: {formula}")
    print(coef_df.to_string(index=False))
    coef_df.to_csv(f"decomposition_{name}.csv", index=False)
    return result, coef_df


def run_all_models(df, tag):
    """Runs the same 4 models (ratio_only, delta_and_gamma, delta_only,
    gamma_only) on whatever sample df has already been filtered to, and
    tags the output files/prints so results from different sample
    definitions (e.g. UC included vs excluded) don't overwrite each other."""
    corr = df[['log_PZ_delta_z', 'log_PZ_gamma_z']].corr().iloc[0, 1]
    print(f"\n--- {tag}: N={len(df)}, "
          f"FBTCS count={int(df['FBTCS_code'].sum())} ---")
    print(f"Correlation between log_PZ_delta_z and log_PZ_gamma_z: {corr:.3f}")

    results = {}
    for formula, name in [
        ('FBTCS_code ~ log_PZ_delta_gamma_ratio_z', 'ratio_only'),
        ('FBTCS_code ~ log_PZ_delta_z + log_PZ_gamma_z', 'delta_and_gamma'),
        ('FBTCS_code ~ log_PZ_delta_z', 'delta_only'),
        ('FBTCS_code ~ log_PZ_gamma_z', 'gamma_only'),
    ]:
        _, coef_df = fit_and_summarize(formula, df, f"{tag}_{name}")
        results[name] = coef_df
    return results


def main():
    df = pd.read_csv(DATA_PATH)

    required = ['log_PZ_delta', 'log_PZ_gamma', 'log_PZ_delta_gamma_ratio',
                'FBTCS_code', 'pat_num', 'classif.', 'vigilance']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Did you regenerate "
            f"clean_final_output_log10.csv with the updated "
            f"statistical_anaylsis.py that keeps raw PZ_delta/PZ_gamma "
            f"and the classif./vigilance metadata columns?"
        )

    df['log_PZ_delta_z'] = zscore(df['log_PZ_delta'])
    df['log_PZ_gamma_z'] = zscore(df['log_PZ_gamma'])
    df['log_PZ_delta_gamma_ratio_z'] = zscore(df['log_PZ_delta_gamma_ratio'])

    # 'unclear' vigilance is always excluded -- there's no valid category
    # to assign it to, unlike UC (see chat discussion). This matches the
    # filtering already used elsewhere (e.g. stats.py).
    n_before = len(df)
    df = df[df['vigilance'].str.lower() != 'unclear'].copy()
    print(f"Excluded {n_before - len(df)} rows with vigilance == 'unclear' "
          f"({len(df)} remain).")

    # --- Model A: UC included (current default -- UC counts as FBTCS_code=0) ---
    df_uc_included = df.copy()
    results_a = run_all_models(df_uc_included, "UC_included")

    # --- Model B: UC excluded entirely (only SG vs CP/SP) ---
    n_before_uc = len(df)
    df_uc_excluded = df[df['classif.'] != 'UC'].copy()
    print(f"\nExcluded {n_before_uc - len(df_uc_excluded)} UC seizures for "
          f"the UC-excluded comparison ({len(df_uc_excluded)} remain).")
    results_b = run_all_models(df_uc_excluded, "UC_excluded")

    # --- Side-by-side comparison: does delta's effect hold up either way? ---
    print("\n\n=== SIDE-BY-SIDE: does the delta effect depend on UC handling? ===")
    for name in ['ratio_only', 'delta_and_gamma', 'delta_only', 'gamma_only']:
        row_a = results_a[name].iloc[-1] if name != 'delta_and_gamma' else \
            results_a[name][results_a[name]['name'] == 'log_PZ_delta_z'].iloc[0]
        row_b = results_b[name].iloc[-1] if name != 'delta_and_gamma' else \
            results_b[name][results_b[name]['name'] == 'log_PZ_delta_z'].iloc[0]
        print(f"\n{name}:")
        print(f"  UC included:  OR={row_a['OR']:.3f}  p={row_a['p']:.4f}")
        print(f"  UC excluded:  OR={row_b['OR']:.3f}  p={row_b['p']:.4f}")

    print(
        "\nHow to read the side-by-side:\n"
        "- If OR and p stay close between 'included' and 'excluded' for a "
        "given model, the result is robust to how UC is handled -- safe to "
        "report either way.\n"
        "- If they diverge a lot (OR shifts substantially, or p crosses "
        "0.05 in one but not the other), that's worth reporting explicitly "
        "as a sensitivity finding rather than picking one silently."
    )


if __name__ == "__main__":
    main()
