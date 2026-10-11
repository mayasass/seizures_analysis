"""
compare_ratios.py

Compares three candidate ratio definitions on PZ (your primary
electrode): delta/gamma (current), delta/beta, and delta/total-power.
Same pattern as delta_gamma_decomposition.py.

Usage:
    python compare_ratios.py
"""

import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy import stats
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

DATA_PATH = "/Users/maya/Documents/lab_project/final_tables/more_ratios_log10.csv"

RATIO_COLS = {
    'delta_gamma': 'log_PZ_delta_gamma_ratio_z',
    'delta_beta': 'log_PZ_delta_beta_ratio_z',
    'delta_totalpower': 'log_PZ_delta_totalpower_ratio_z',
}


def fit_and_summarize(formula, df, name):
    endog, exog = dmatrices(formula, data=df, return_type='dataframe')
    # IMPORTANT: dmatrices silently drops rows with NaN/inf in any formula
    # variable. exog_re must be built from the SAME surviving rows, or its
    # row count won't match exog/endog and the fit crashes with a shape
    # mismatch (seen in practice: one seizure had a NaN ratio value, e.g.
    # from a zero-valued denominator, and got silently dropped by
    # dmatrices while exog_re still had the original row count).
    df_used = df.loc[exog.index]
    exog_re = pd.get_dummies(df_used['pat_num'], drop_first=False)
    n_dropped = len(df) - len(df_used)
    if n_dropped > 0:
        print(f"  NOTE: {n_dropped} row(s) dropped from this fit due to "
              f"NaN/inf in the formula's variables (see df_used vs df).")
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


def main():
    df = pd.read_csv(DATA_PATH)
    df = df[df['vigilance'].str.lower() != 'unclear'].copy()

    print("Correlation between the three ratios:")
    print(df[list(RATIO_COLS.values())].corr().round(3).to_string())

    for name, col in RATIO_COLS.items():
        fit_and_summarize(f'FBTCS_code ~ {col}', df, name)

    combined_formula = f"FBTCS_code ~ {' + '.join(RATIO_COLS.values())}"
    fit_and_summarize(combined_formula, df, 'all_three_combined')

    print(
        "\nHow to read this:\n"
        "- Compare OR/p across the three individual models -- whichever "
        "has the strongest, most significant effect on its own is the "
        "best-performing single ratio.\n"
        "- High correlation between two ratios means they're capturing "
        "similar information (unsurprising if beta and gamma covary, or "
        "if total_power is dominated by delta itself).\n"
        "- In 'all_three_combined', whichever ratio keeps a significant, "
        "stable coefficient once the others are accounted for is the one "
        "carrying genuinely independent information -- same logic as the "
        "delta/gamma decomposition."
    )


if __name__ == "__main__":
    main()
