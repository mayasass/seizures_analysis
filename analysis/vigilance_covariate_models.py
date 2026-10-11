"""
vigilance_covariate_models.py  (11 Oct 2026, Miriam's suggestion)

Main cohort (clean_final_output_log10.csv, 'unclear' vigilance removed -> 1,029 seizures,
UC seizures included as non-FBTCS). All five vigilance states, awake = reference.

Prints:
  0. Seizures and FBTCS per vigilance state (counted from the data).
  1. Unadjusted model         FBTCS ~ ratio                      (same as the main result, for comparison)
  2. Covariate model          FBTCS ~ ratio + vigilance state
       -> does the ratio still predict FBTCS among seizures in the SAME state?
  3. Interaction model        FBTCS ~ ratio * vigilance state
       -> the ratio's OR WITHIN each state (same model type as Figure 5).
All models: Bayesian mixed logistic regression with a random intercept per patient,
exactly as in the other scripts. ratio = z-scored log10 Pz delta/gamma ratio.

NOTE: N3 has 2 FBTCS and REM 3, so their estimates will have very wide CIs and can be unstable.
"""

import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

DATA_PATH = "/Users/maya/Documents/lab_project/final_tables/clean_final_output_log10.csv"

STAGE_ORDER = ['awake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM']
CANON = {
    'awake': 'awake', 'rem': 'REM',
    'sleep stage i': 'NREM 1', 'sleep stage 1': 'NREM 1', 'nrem 1': 'NREM 1',
    'sleep stage ii': 'NREM 2', 'sleep stage 2': 'NREM 2', 'nrem 2': 'NREM 2',
    'sleep stage iii': 'NREM 3', 'sleep stage 3': 'NREM 3', 'nrem 3': 'NREM 3',
    'sleep stage iv': 'NREM 3', 'sleep stage 4': 'NREM 3',
}
VIG = 'C(stage, Treatment(reference="awake"))'


def load():
    df = pd.read_csv(DATA_PATH)
    # z-score BEFORE removing 'unclear', exactly as in delta_gamma_decomposition.py
    r = df['log_PZ_delta_gamma_ratio']
    df['ratio_z'] = (r - r.mean()) / r.std()
    n0 = len(df)
    df = df[df['vigilance'].astype(str).str.lower() != 'unclear'].copy()
    df['stage'] = df['vigilance'].astype(str).str.strip().str.lower().map(CANON)
    bad = df.loc[df['stage'].isna(), 'vigilance'].unique()
    if len(bad):
        raise ValueError(f"Unrecognized vigilance labels (would be dropped): {bad}")
    print(f"Excluded {n0 - len(df)} 'unclear' rows -> N = {len(df)} seizures, "
          f"{int(df['FBTCS_code'].sum())} FBTCS, {df['pat_num'].nunique()} patients")
    return df


def fit(formula, df):
    endog, exog = dmatrices(formula, data=df, return_type='dataframe')
    if len(exog) != len(df):
        print(f"  NOTE: {len(df) - len(exog)} row(s) dropped (NaN/inf in the formula variables)")
    used = df.loc[exog.index]
    exog_re = pd.get_dummies(used['pat_num'], drop_first=False)
    model = BinomialBayesMixedGLM(endog, exog, exog_re, np.ones(exog_re.shape[1], dtype=int))
    res = model.fit_vb(verbose=False)
    return pd.Series(res.fe_mean, index=exog.columns), pd.Series(res.fe_sd, index=exog.columns)


def or_text(b, sd):
    return f"{np.exp(b):.2f} ({np.exp(b - 1.96 * sd):.2f}–{np.exp(b + 1.96 * sd):.2f})"


def coef_table(mean, sd, title):
    print(f"\n=== {title} ===")
    rows = []
    for name in mean.index:
        if name == 'Intercept':
            continue
        b, s = mean[name], sd[name]
        z = b / s
        rows.append({'term': name.replace(VIG, 'stage'), 'OR (95% CI)': or_text(b, s),
                     'beta': round(b, 4), 'SD': round(s, 4), 'p': round(2 * norm.sf(abs(z)), 4)})
    print(pd.DataFrame(rows).to_string(index=False))


def plot_forest(m1, s1, m2, s2, per_state, n_fbtcs):
    """Forest plot of the ratio OR (95% CI): unadjusted, adjusted for vigilance, and within each state.
    Every number comes from the models fitted above."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, b, sd, color = [], [], [], []
    labels += [f'All seizures, unadjusted ({n_fbtcs} FBTCS)', f'All seizures, adjusted for vigilance ({n_fbtcs} FBTCS)']
    b += [m1['ratio_z'], m2['ratio_z']]
    sd += [s1['ratio_z'], s2['ratio_z']]
    color += ['#3B6EA8', '#3B6EA8']
    for _, r in per_state.iterrows():
        labels.append(f"Within {r['state']} ({r['FBTCS']} FBTCS)")
        b.append(r['_b']); sd.append(r['_sd']); color.append('#D9822B')
    b, sd = np.array(b), np.array(sd)
    orv, lo, hi = np.exp(b), np.exp(b - 1.96 * sd), np.exp(b + 1.96 * sd)

    y = np.arange(len(labels))[::-1].astype(float)
    y[2:] -= 0.6                                   # small gap between the two groups
    fig, ax = plt.subplots(figsize=(8, 5))
    for yi, o, l, h, c in zip(y, orv, lo, hi, color):
        ax.errorbar(o, yi, xerr=[[o - l], [h - o]], fmt='o', color=c, capsize=3, ms=6, lw=1.5)
    ax.axvline(1, color='gray', ls='--', lw=1)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    xmin, xmax = min(lo.min(), 0.9) * 0.8, max(hi.max(), 1.1) * 1.25
    ax.set_xlim(xmin, xmax)
    ax.set_xticks([t for t in [0.03, 0.06, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64] if xmin <= t <= xmax])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Odds ratio of FBTCS per 1 SD of the Pz delta/gamma ratio (95% CI), log scale')
    ax.set_title('Delta/gamma ratio and FBTCS, accounting for vigilance state', pad=12)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    here = os.environ.get('FIGURES_DIR', os.path.dirname(os.path.abspath(__file__)))   # set by run_all.py
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(here, f'fig9_ratio_by_vigilance_models.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("\nSaved fig9_ratio_by_vigilance_models.png / .svg")


def main():
    df = load()

    print("\n=== Seizures and FBTCS per vigilance state ===")
    t = df.groupby('stage')['FBTCS_code'].agg(seizures='count', FBTCS='sum').reindex(STAGE_ORDER)
    t.loc['Total'] = t.sum()
    t = t.astype(int)
    t['% FBTCS'] = (100 * t['FBTCS'] / t['seizures']).round(1)
    print(t.to_string())

    m1, s1 = fit('FBTCS_code ~ ratio_z', df)
    coef_table(m1, s1, '1. Unadjusted: FBTCS ~ ratio')

    m2, s2 = fit(f'FBTCS_code ~ ratio_z + {VIG}', df)
    coef_table(m2, s2, '2. Covariate model: FBTCS ~ ratio + vigilance state (reference = awake)')

    m3, s3 = fit(f'FBTCS_code ~ ratio_z * {VIG}', df)
    coef_table(m3, s3, '3. Interaction model: FBTCS ~ ratio * vigilance state (raw coefficients)')

    # ratio OR within each state = main ratio slope + that state's interaction term
    # (fit_vb treats coefficients as independent, so SD = sqrt(sd1^2 + sd2^2))
    print("\n=== 3b. Ratio OR per 1 SD WITHIN each vigilance state (from model 3) ===")
    rows = []
    for st in STAGE_ORDER:
        b, var = m3['ratio_z'], s3['ratio_z'] ** 2
        inter = f'ratio_z:{VIG}[T.{st}]'
        if st != 'awake':
            if inter not in m3.index:
                print(f"  {st}: interaction term not found, skipped")
                continue
            b += m3[inter]
            var += s3[inter] ** 2
        n_f = int(t.loc[st, 'FBTCS'])
        rows.append({'state': st, 'FBTCS': n_f, 'ratio OR (95% CI)': or_text(b, np.sqrt(var)),
                     '_b': b, '_sd': np.sqrt(var)})
    out = pd.DataFrame(rows)
    print(out.drop(columns=['_b', '_sd']).to_string(index=False))

    plot_forest(m1, s1, m2, s2, out, int(t.loc['Total', 'FBTCS']))

    print("\nHow to read this:")
    print("- Compare the ratio OR in model 2 with model 1. If it stays similar, the ratio's effect is NOT just")
    print("  a by-product of which vigilance state the seizure happened in.")
    print("- In 3b, states with only 2-5 FBTCS will have very wide CIs; treat them as exploratory.")


if __name__ == "__main__":
    main()