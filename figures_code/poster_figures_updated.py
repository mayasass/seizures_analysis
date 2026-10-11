"""
poster_figures_updated.py

Recreates the poster's core result figures on the current, corrected
dataset (clean_final_output_log10.csv), matching stats.py's original
analyses but cleaned up and saved as publication-ready files (PNG + SVG,
300 DPI) rather than just shown on screen.

Figures produced:
  fig1_sleep_stage_distribution.png/svg
      Pie chart: distribution of seizures across sleep stages.
  fig2_fbtcs_likelihood_by_stage.png/svg
      Bar chart: % FBTCS per sleep stage, with n/N counts annotated.
  fig3_delta_gamma_by_stage_and_type.png/svg
      Bar chart: mean log10 PZ delta/gamma ratio by sleep stage and
      seizure type (FBTCS vs. not), with error bars.
  fig4_predicted_probability_ratio.png/svg
      Predicted probability of FBTCS vs. PZ delta/gamma ratio (single
      curve, population-level model).
  fig5_predicted_probability_by_vigilance.png/svg
      Same, with one curve per sleep stage (interaction model).

Usage:
    python poster_figures_updated.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import dmatrices
from scipy.special import expit
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

DATA_PATH = "/Users/maya/Documents/lab_project/final_tables/clean_final_output_log10.csv"
import os
OUTPUT_DIR = os.environ.get("FIGURES_DIR", ".")   # set by run_all.py

VIG_MAP = {
    'sleep stage i': 'NREM 1', 'sleep stage 1': 'NREM 1',
    'sleep stage ii': 'NREM 2', 'sleep stage 2': 'NREM 2',
    'sleep stage iii': 'NREM 3', 'sleep stage 3': 'NREM 3',
    'sleep stage iv': 'NREM 3', 'sleep stage 4': 'NREM 3',
}
STAGE_ORDER = ['awake', 'NREM 1', 'NREM 2', 'NREM 3', 'REM']


CANON = dict(VIG_MAP)
CANON.update({'awake': 'awake', 'rem': 'REM',
              'nrem 1': 'NREM 1', 'nrem 2': 'NREM 2', 'nrem 3': 'NREM 3'})


def recode_vigilance(v):
    # case-insensitive lookup, but return the CANONICAL label (e.g. 'REM', not 'rem')
    raw = str(v).strip()
    return CANON.get(raw.lower(), raw)


def savefig(fig, name):
    fig.savefig(f"{OUTPUT_DIR}/{name}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{OUTPUT_DIR}/{name}.svg", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {name}.png / .svg")


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df['vigilance'].str.lower() != 'unclear'].copy()
    df['vigilance_recode'] = df['vigilance'].apply(recode_vigilance)
    bad = sorted(set(df['vigilance_recode']) - set(STAGE_ORDER))
    if bad:
        raise ValueError(f"Unrecognized vigilance labels (would be silently dropped): {bad}")
    print("Seizures per sleep stage:\n", df['vigilance_recode'].value_counts().reindex(STAGE_ORDER).to_string(), sep="")
    df['vigilance_recode'] = pd.Categorical(df['vigilance_recode'], categories=STAGE_ORDER, ordered=True)
    df['seizure_type'] = df['FBTCS_code'].map({0: 'not FBTCS', 1: 'FBTCS'})
    df['log_PZ_delta_gamma_ratio_z'] = (
        (df['log_PZ_delta_gamma_ratio'] - df['log_PZ_delta_gamma_ratio'].mean())
        / df['log_PZ_delta_gamma_ratio'].std()
    )
    return df


def fig1_sleep_stage_distribution(df):
    counts = df['vigilance_recode'].value_counts().reindex(STAGE_ORDER)
    assert counts.sum() == len(df), f"pie counts {counts.sum()} != N {len(df)}"
    fig, ax = plt.subplots(figsize=(6, 6))
    pct = 100 * counts / counts.sum()
    SMALL = 5  # slices below 5% get their label outside with a short line, so texts do not overlap
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, startangle=90, colors=plt.cm.Paired.colors,
        autopct=lambda p: f'{p:.1f}%' if p >= SMALL else '')
    # move the labels of the small slices to the upper right, one under the other
    small = [i for i, p in enumerate(pct) if p < SMALL]
    # slice nearest the top gets the top label, so the lines do not cross
    small.sort(key=lambda i: -(wedges[i].theta1 + wedges[i].theta2) / 2)
    for k, i in enumerate(small):
        w = wedges[i]
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        texts[i].set_visible(False)
        ax.annotate(f"{counts.index[i]} {pct.iloc[i]:.1f}%",
                    xy=(np.cos(ang), np.sin(ang)), xytext=(0.75, 1.12 - 0.14 * k),
                    ha='left', va='center',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))
    ax.set_title('Distribution of Seizures Across Sleep Stages', pad=30)
    ax.axis('equal')
    savefig(fig, "fig1_sleep_stage_distribution")


def fig2_fbtcs_likelihood_by_stage(df):
    grouped = df.groupby('vigilance_recode', observed=True).agg(
        total=('FBTCS_code', 'count'),
        fbtcs=('FBTCS_code', 'sum'),
    )
    grouped['pct'] = grouped['fbtcs'] / grouped['total'] * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(grouped.index.astype(str), grouped['pct'], color='skyblue', edgecolor='black')
    for bar, (idx, row) in zip(bars, grouped.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{int(row['fbtcs'])}/{int(row['total'])}", ha='center', fontsize=9, weight='bold')
    ax.set_ylabel('% FBTCS')
    ax.set_title('FBTCS Likelihood by Sleep Stage')
    ax.set_ylim(0, grouped['pct'].max() * 1.4 + 2)
    plt.xticks(rotation=30)
    fig.tight_layout()
    savefig(fig, "fig2_fbtcs_likelihood_by_stage")


def fig3_delta_gamma_by_stage_and_type(df):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=df, x='vigilance_recode', y='log_PZ_delta_gamma_ratio',
                hue='seizure_type', palette='Set2', errorbar='sd', ax=ax)
    # number of seizures behind each bar, counted from the data, written above the error bar
    stats = df.groupby(['vigilance_recode', 'seizure_type'], observed=True)['log_PZ_delta_gamma_ratio'] \
              .agg(['mean', 'std', 'count'])
    hue_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    for container, hue in zip(ax.containers, hue_labels):
        for bar, stage in zip(container, STAGE_ORDER):
            if (stage, hue) not in stats.index:
                continue
            m, sd, n = stats.loc[(stage, hue)]
            top = m + (0 if pd.isna(sd) else sd)
            ax.text(bar.get_x() + bar.get_width() / 2, top + 0.03, f"{int(n)}",
                    ha='center', va='bottom', fontsize=8)
    ax.set_title('log10 PZ Delta/Gamma Ratio by Sleep Stage and Seizure Type')
    ax.set_ylabel('Mean log10 delta/gamma ratio')
    ax.set_xlabel('Sleep Stage')
    plt.xticks(rotation=30)
    fig.tight_layout()
    savefig(fig, "fig3_delta_gamma_by_stage_and_type")


# Literature values, as shown on the poster (Fig 1.2, from Ng & Pavlova 2013,
# Epilepsy Res Treat 932790). VERIFY these against the paper before publishing.
LITERATURE_SEIZURE_PCT = {'awake': 28, 'NREM 1': 16, 'NREM 2': 44, 'NREM 3': 11, 'REM': 1}


def fig1b_distribution_vs_literature(df):
    """Our seizure distribution across vigilance states next to the published one."""
    counts = df['vigilance_recode'].value_counts().reindex(STAGE_ORDER)
    ours = 100 * counts / counts.sum()
    lit = pd.Series(LITERATURE_SEIZURE_PCT).reindex(STAGE_ORDER)
    x = np.arange(len(STAGE_ORDER))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - width / 2, lit, width, color='#B0B0B0', label='Literature (Ng & Pavlova, 2013)')
    b2 = ax.bar(x + width / 2, ours, width, color='#3B6EA8', label=f'EPILEPSIAE, this study (N = {counts.sum()})')
    for bar, v in zip(b1, lit):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.0f}%", ha='center', fontsize=9)
    for bar, v, n in zip(b2, ours, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%\n(n={n})", ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_ylabel('% of focal seizures')
    ax.set_ylim(0, max(ours.max(), lit.max()) + 12)
    ax.set_title('Distribution of Focal Seizures Across Vigilance States', pad=12)
    ax.legend(frameon=False, loc='upper right')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig1b_distribution_vs_literature")


def print_uc_summary(df):
    """How many UC (unclassified seizure type) seizures, overall and per stage."""
    uc = df['classif.'].astype(str).str.strip().str.upper() == 'UC'
    print(f"\nUC (unclassified seizure type): {int(uc.sum())} of {len(df)} seizures "
          f"({100 * uc.mean():.1f}%)")
    print(pd.crosstab(df['vigilance_recode'], uc.map({True: 'UC', False: 'classified'}),
                      margins=True).to_string())


def fit_ratio_model(df):
    formula = 'FBTCS_code ~ log_PZ_delta_gamma_ratio_z'
    endog, exog = dmatrices(formula, data=df, return_type='dataframe')
    df_used = df.loc[exog.index]
    exog_re = pd.get_dummies(df_used['pat_num'], drop_first=False)
    ident = np.ones(exog_re.shape[1], dtype=int)
    model = BinomialBayesMixedGLM(endog, exog, exog_re, ident)
    return model.fit_vb(verbose=False)


def fit_interaction_model(df):
    formula = ('FBTCS_code ~ log_PZ_delta_gamma_ratio_z * '
               'C(vigilance_recode, Treatment(reference="NREM 3"))')
    endog, exog = dmatrices(formula, data=df, return_type='dataframe')
    df_used = df.loc[exog.index]
    exog_re = pd.get_dummies(df_used['pat_num'], drop_first=False)
    ident = np.ones(exog_re.shape[1], dtype=int)
    model = BinomialBayesMixedGLM(endog, exog, exog_re, ident)
    return model.fit_vb(verbose=False), exog.design_info


def fig4_predicted_probability_ratio(df):
    result = fit_ratio_model(df)
    x_grid = np.linspace(-2.5, 2.5, 100)
    intercept = result.fe_mean[0]
    slope = result.fe_mean[1]
    probs = expit(intercept + slope * x_grid)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_grid, probs, color='C0', linewidth=2)
    ax.set_xlabel('log10 PZ delta/gamma ratio (z-scored)')
    ax.set_ylabel('P(FBTCS)')
    ax.set_title('Predicted Probability of FBTCS by Delta/Gamma Ratio')
    fig.tight_layout()
    savefig(fig, "fig4_predicted_probability_ratio")


def fig5_predicted_probability_by_vigilance(df):
    result, design_info = fit_interaction_model(df)
    names = design_info.column_names
    x_grid = np.linspace(-2.5, 2.5, 100)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for stage in STAGE_ORDER:
        row = np.zeros_like(result.fe_mean)
        row[names.index('Intercept')] = 1
        stage_col = f'C(vigilance_recode, Treatment(reference="NREM 3"))[T.{stage}]'
        inter_col = f'log_PZ_delta_gamma_ratio_z:{stage_col}'

        if stage != 'NREM 3':
            if stage_col not in names:
                print(f"  Skipping {stage}: not found in model (check category exists in data)")
                continue
            row[names.index(stage_col)] = 1

        probs = []
        for x in x_grid:
            r = row.copy()
            r[names.index('log_PZ_delta_gamma_ratio_z')] = x
            if stage != 'NREM 3' and inter_col in names:
                r[names.index(inter_col)] = x
            probs.append(expit((r * result.fe_mean).sum()))

        ax.plot(x_grid, probs, label=stage, linewidth=2)

    ax.set_xlabel('log10 PZ delta/gamma ratio (z-scored)')
    ax.set_ylabel('P(FBTCS)')
    ax.set_title('Predicted Probability of FBTCS by Delta/Gamma Ratio and Sleep Stage')
    ax.legend(title='Sleep Stage')
    fig.tight_layout()
    savefig(fig, "fig5_predicted_probability_by_vigilance")


# ---------------------------------------------------------------------------
# EXTRA versions with 95% bands (added 8 Oct 2026). Figs 4 and 5 above are
# unchanged; these are saved as separate files (fig4b, fig5b) for Oren to choose.
# Band = 2.5th-97.5th percentile of predicted probability over 4,000 draws of the
# fixed effects from the model's own posterior (mean fe_mean, SD fe_sd; fit_vb
# treats the coefficients as independent, so the draws are independent too).
# ---------------------------------------------------------------------------
N_DRAWS = 4000


def _band(X, result, seed=0):
    rng = np.random.default_rng(seed)
    draws = rng.normal(result.fe_mean, result.fe_sd, size=(N_DRAWS, len(result.fe_mean)))
    p = expit(X @ draws.T)                       # (n_x, N_DRAWS)
    return expit(X @ result.fe_mean), np.percentile(p, 2.5, axis=1), np.percentile(p, 97.5, axis=1)


def fig4b_predicted_probability_ratio_with_band(df):
    result = fit_ratio_model(df)
    x_grid = np.linspace(-2.5, 2.5, 100)
    X = np.column_stack([np.ones_like(x_grid), x_grid])
    mean, lo, hi = _band(X, result)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(x_grid, lo, hi, color='C0', alpha=0.2, linewidth=0, label='95% band')
    ax.plot(x_grid, mean, color='C0', linewidth=2, label='Model estimate')
    ax.set_xlabel('log10 PZ delta/gamma ratio (z-scored)')
    ax.set_ylabel('P(FBTCS)')
    ax.set_ylim(bottom=0)
    ax.set_title('Predicted Probability of FBTCS by Delta/Gamma Ratio', pad=12)
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    savefig(fig, "fig4b_predicted_probability_ratio_with_band")


def fig5b_predicted_probability_by_vigilance_with_band(df):
    result, design_info = fit_interaction_model(df)
    names = design_info.column_names
    x_grid = np.linspace(-2.5, 2.5, 100)
    counts = df.groupby('vigilance_recode', observed=True)['FBTCS_code'].agg(['sum', 'count'])

    # one small panel per stage (5 overlapping bands on one axis would be unreadable)
    fig, axes = plt.subplots(1, len(STAGE_ORDER), figsize=(15, 3.8), sharey=True)
    for ax, stage in zip(axes, STAGE_ORDER):
        stage_col = f'C(vigilance_recode, Treatment(reference="NREM 3"))[T.{stage}]'
        inter_col = f'log_PZ_delta_gamma_ratio_z:{stage_col}'
        X = np.zeros((len(x_grid), len(names)))
        X[:, names.index('Intercept')] = 1
        X[:, names.index('log_PZ_delta_gamma_ratio_z')] = x_grid
        if stage != 'NREM 3':
            if stage_col not in names:
                print(f"  Skipping {stage}: not found in model")
                continue
            X[:, names.index(stage_col)] = 1
            X[:, names.index(inter_col)] = x_grid
        mean, lo, hi = _band(X, result)
        ax.fill_between(x_grid, lo, hi, color='C0', alpha=0.2, linewidth=0)
        ax.plot(x_grid, mean, color='C0', linewidth=2)
        n_f, n_all = int(counts.loc[stage, 'sum']), int(counts.loc[stage, 'count'])
        ax.set_title(f"{stage}\n{n_f} FBTCS / {n_all} seizures", fontsize=10)
        ax.set_xlabel('Ratio (z-scored)')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel('P(FBTCS)')
    axes[0].set_ylim(0, 1)
    fig.suptitle('Predicted Probability of FBTCS by Delta/Gamma Ratio and Sleep Stage (95% bands)', y=1.04)
    fig.tight_layout()
    savefig(fig, "fig5b_predicted_probability_by_vigilance_with_band")


def main():
    df = load_data()
    print(f"N={len(df)} seizures after excluding 'unclear' vigilance")

    fig1_sleep_stage_distribution(df)
    fig1b_distribution_vs_literature(df)
    print_uc_summary(df)
    fig2_fbtcs_likelihood_by_stage(df)
    fig3_delta_gamma_by_stage_and_type(df)
    fig4_predicted_probability_ratio(df)
    fig5_predicted_probability_by_vigilance(df)
    fig4b_predicted_probability_ratio_with_band(df)          # extra, for Oren to choose
    fig5b_predicted_probability_by_vigilance_with_band(df)   # extra, for Oren to choose

    print("\nAll poster figures regenerated on the current dataset.")


if __name__ == "__main__":
    main()