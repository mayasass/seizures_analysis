"""
followup_figures.py  (rewritten 8 Oct 2026)

Figures 6-8 for the paper. Figs 6 and 7 no longer refit any model: they read the
result tables that the analysis scripts already saved, so every number on the
figure is exactly the number reported in the Results.

  fig6_decomposition_or_comparison   odds ratios (95% CI) for ratio / delta only / gamma only,
                                     UC included vs. UC excluded
        reads: decomposition_UC_included_*.csv, decomposition_UC_excluded_*.csv
               (written by delta_gamma_decomposition.py into this folder)
  fig7_lopo_auc_comparison           leave-one-patient-out AUC (slope-only) with 95% CI
        reads: lopo_summary_UC_included.csv, lopo_summary_UC_excluded.csv
               (written by leave_one_patient_out.py into this folder)
  fig8_topography_comparison         own lobe vs. rest of scalp, FBTCS vs. non-FBTCS
        reads: topography_all_electrodes_log10.csv

Run from the project folder (run_all.py does this):
    /usr/local/bin/python3 run_all.py figures
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TOPO_DATA_PATH = "/Users/maya/Documents/lab_project/final_tables/topography_all_electrodes_log10.csv"
RESULTS = os.environ.get('RESULTS_DIR', HERE)      # set by run_all.py
OUTPUT_DIR = os.environ.get('FIGURES_DIR', HERE)   # set by run_all.py

UC_COLORS = {'UC included': '#3B6EA8', 'UC excluded': '#D9822B'}
UC_FILES = {'UC included': 'UC_included', 'UC excluded': 'UC_excluded'}
N_FBTCS = 58                                   # same 58 FBTCS in both UC handlings
N_TOTAL = {'UC included': 1029, 'UC excluded': 745}
# Which UC handling to SHOW in Figs 6-7. Main analysis = UC included (Maya, 8 Oct).
# The UC-excluded numbers stay in the Results text as the sensitivity check.
# To show both again: UC_SHOW = ['UC included', 'UC excluded']
UC_SHOW = ['UC included']

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


def savefig(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.svg"), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {name}.png / .svg")


def need(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run the analysis step that writes it first "
                                f"(run_all.py decomposition / lopo / topography).")
    return path


# ---------------------------------------------------------------- Fig 6
# (results file, coefficient name or None = the model's only predictor, row label)
# Same model order as Figure 7; the delta + gamma model has two predictors, so two rows.
FIG6_MODELS = [('ratio_only', None, 'Delta/gamma ratio'),
               ('delta_only', None, 'Delta only'),
               ('delta_and_gamma', 'log_PZ_delta_z', 'Delta + gamma model: delta'),
               ('delta_and_gamma', 'log_PZ_gamma_z', 'Delta + gamma model: gamma'),
               ('gamma_only', None, 'Gamma only')]


def fig6_decomposition_or_comparison():
    rows = []
    for uc_label in UC_SHOW:
        tag = UC_FILES[uc_label]
        for key, term, label in FIG6_MODELS:
            t = pd.read_csv(need(os.path.join(RESULTS, f"decomposition_{tag}_{key}.csv")))
            r = t[t['name'] != 'Intercept'].iloc[-1] if term is None else t[t['name'] == term].iloc[0]
            b, sd = float(r['beta']), float(r['SD'])
            rows.append(dict(uc=uc_label, model=label, OR=np.exp(b),
                             lo=np.exp(b - 1.96 * sd), hi=np.exp(b + 1.96 * sd), p=float(r['p'])))
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    labels = [lab for _, _, lab in FIG6_MODELS]
    for i, uc_label in enumerate(UC_SHOW):
        sub = res[res['uc'] == uc_label].set_index('model').reindex(labels)
        y = np.arange(len(labels))[::-1] + (0 if len(UC_SHOW) == 1 else (0.15 if i == 0 else -0.15))
        ax.errorbar(sub['OR'], y, xerr=[sub['OR'] - sub['lo'], sub['hi'] - sub['OR']],
                    fmt='o', color=UC_COLORS[uc_label], capsize=3, ms=6, lw=1.5,
                    label=f"{uc_label} (N = {N_TOTAL[uc_label]})")
        for yy, (_, r) in zip(y, sub.iterrows()):
            ax.text(2.6, yy, f"{r['OR']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})",
                    va='center', fontsize=9, color=UC_COLORS[uc_label])
    ax.axvline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 2.5])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(np.arange(len(labels))[::-1])
    ax.set_yticklabels(labels)
    ax.set_xlabel('Odds ratio per 1 SD (95% CI), log scale')
    if len(UC_SHOW) == 1:   # one analysis only: no legend, N in the title
        ax.set_title(f'Pz delta/gamma ratio vs. delta and gamma (N = {N_TOTAL[UC_SHOW[0]]:,})', pad=12)
    else:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
        ax.set_title('Pz delta/gamma ratio vs. delta and gamma', pad=28)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    savefig(fig, "fig6_decomposition_or_comparison")


# ---------------------------------------------------------------- Fig 7
FIG7_MODELS = [('ratio_only', 'Delta/gamma ratio'), ('delta_only', 'Delta only'),
               ('delta_and_gamma', 'Delta + gamma'), ('gamma_only', 'Gamma only')]


def hanley_mcneil_se(auc, n_pos, n_neg):
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    return np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
                    + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))


def fig7_lopo_auc_comparison():
    rows = []
    for uc_label in UC_SHOW:
        tag = UC_FILES[uc_label]
        t = pd.read_csv(need(os.path.join(RESULTS, f"lopo_summary_{tag}.csv"))).set_index('model')
        for key, label in FIG7_MODELS:
            auc = float(t.loc[key, 'AUC_slope_only'])
            n = int(t.loc[key, 'N'])
            se = hanley_mcneil_se(auc, N_FBTCS, n - N_FBTCS)
            rows.append(dict(uc=uc_label, model=label, AUC=auc, lo=auc - 1.96 * se, hi=auc + 1.96 * se))
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))

    # horizontal layout, same as Figure 6: models on the y-axis, AUC along the x-axis
    fig, ax = plt.subplots(figsize=(7.5, 4))
    labels = [lab for _, lab in FIG7_MODELS]
    for i, uc_label in enumerate(UC_SHOW):
        sub = res[res['uc'] == uc_label].set_index('model').reindex(labels)
        y = np.arange(len(labels))[::-1] + (0 if len(UC_SHOW) == 1 else (0.15 if i == 0 else -0.15))
        ax.errorbar(sub['AUC'], y, xerr=[sub['AUC'] - sub['lo'], sub['hi'] - sub['AUC']],
                    fmt='o', color=UC_COLORS[uc_label], capsize=3, ms=6, lw=1.5,
                    label=f"{uc_label} (N = {N_TOTAL[uc_label]})")
        for yy, (_, r) in zip(y, sub.iterrows()):
            ax.text(0.765, yy, f"{r['AUC']:.2f} ({r['lo']:.2f}\u2013{r['hi']:.2f})",
                    va='center', fontsize=9, color=UC_COLORS[uc_label])
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_xlim(0.35, 0.75)
    ax.set_yticks(np.arange(len(labels))[::-1])
    ax.set_yticklabels(labels)
    ax.set_xlabel('AUC in left-out patients (95% CI); dashed line = chance')
    if len(UC_SHOW) == 1:   # one analysis only: no legend, N in the title
        ax.set_title(f'Leave-one-patient-out validation (N = {N_TOTAL[UC_SHOW[0]]:,})', pad=12)
    else:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
        ax.set_title('Leave-one-patient-out validation', pad=28)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    savefig(fig, "fig7_lopo_auc_comparison")


# ---------------------------------------------------------------- Fig 8
def matching_electrodes(lobe, side):
    lobe = str(lobe).strip().lower()
    side = str(side).strip().lower()
    if lobe not in LOBE_ELECTRODES or side not in ('left', 'right'):
        return None
    side_set = LEFT_ELECTRODES if side == 'left' else RIGHT_ELECTRODES
    return {e for e in LOBE_ELECTRODES[lobe] if e in MIDLINE_ELECTRODES or e in side_set}


def zcol(electrode):
    return f"log_{electrode}_delta_gamma_ratio_z"


def fig8_topography_comparison():
    df = pd.read_csv(need(TOPO_DATA_PATH))
    df = df[df['vigilance'].astype(str).str.lower() != 'unclear'].copy()

    m_list, nm_list = [], []
    for _, row in df.iterrows():
        matches = matching_electrodes(row['lobe'], row['side'])
        if not matches:
            m_list.append(np.nan); nm_list.append(np.nan)
            continue
        m = [row[zcol(e)] for e in matches if zcol(e) in df.columns]
        nm = [row[zcol(e)] for e in ALL_ELECTRODES - matches if zcol(e) in df.columns]
        m_list.append(np.nanmean(m) if m else np.nan)
        nm_list.append(np.nanmean(nm) if nm else np.nan)
    df['matching_z'] = m_list
    df['nonmatching_z'] = nm_list
    df = df.dropna(subset=['matching_z', 'nonmatching_z'])
    df['group'] = df['FBTCS_code'].map({0: 'Non-FBTCS', 1: 'FBTCS'})

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    groups = ['FBTCS', 'Non-FBTCS']
    regions = [('matching_z', 'Own lobe', '#C44E52'), ('nonmatching_z', 'Rest of scalp', '#3B6EA8')]
    x = np.arange(len(groups))
    for j, (col, lab, color) in enumerate(regions):
        means = [df.loc[df['group'] == g, col].mean() for g in groups]
        ci = [1.96 * df.loc[df['group'] == g, col].sem() for g in groups]
        ax.errorbar(x + (-0.12 if j == 0 else 0.12), means, yerr=ci, fmt='o', color=color,
                    capsize=3, ms=6, lw=1.5, label=lab)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n = {(df['group'] == g).sum()})" for g in groups])
    ax.set_ylabel('Delta/gamma ratio, z-score (mean, 95% CI)')
    ax.set_title("Seizure's own lobe vs. rest of scalp", pad=12)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    print(f"Fig 8: {len(df)} seizures with known lobe and side "
          f"({(df['group'] == 'FBTCS').sum()} FBTCS)")
    savefig(fig, "fig8_topography_comparison")


def main():
    fig6_decomposition_or_comparison()
    fig7_lopo_auc_comparison()
    fig8_topography_comparison()
    print("\nAll follow-up figures generated.")


if __name__ == "__main__":
    main()