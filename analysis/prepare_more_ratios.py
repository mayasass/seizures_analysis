"""
prepare_more_ratios.py

Adds delta/beta and delta/total-power ratios (PI's request) on top of the
existing delta/gamma ratio, for the primary electrode (PZ) plus T7/T8/FZ
for consistency with the original ratio set. All three ratio types are
computed directly from columns final_output.csv already has (delta,
beta, gamma, total_power per electrode) -- no rerun of the raw EEG
pipeline needed.

Uses the corrected single-pass union outlier exclusion (see
prepare_topography_data.py / chat discussion): compute all thresholds on
the full data first, drop a seizure once if it exceeds ANY of them,
rather than compounding sequential filters.

Usage:
    python prepare_more_ratios.py
"""

import pandas as pd
import numpy as np

INPUT_PATH = "/Users/maya/Documents/lab_project/final_tables/final_output.csv"
OUTPUT_PATH = "/Users/maya/Documents/lab_project/final_tables/more_ratios_log10.csv"

ELECTRODES = ['T7', 'T8', 'PZ', 'FZ']
METADATA_COLS = ["pat_num", "seizure_num", "classif.", "FBTCS_code",
                  "vigilance", "vigilance_code", "lobe", "temporal_code", "side"]


# --- Cohort choice -------------------------------------------------------------
# True  = use the SAME 1,029-seizure POSTER cohort as the main analysis (top-5% rule on the
#         8 ratio columns: delta/gamma and sigma/beta at T7, T8, PZ, FZ; no extra exclusion).
# False = the older behaviour: this script's own, stricter union exclusion over ITS columns.
POSTER_COHORT = True
POSTER_RATIO_COLS = [f"{e}_{r}" for e in ['T7', 'T8', 'PZ', 'FZ']
                     for r in ['delta_gamma_ratio', 'sigma_beta_ratio']]


def poster_cohort_keep_mask(df_full):
    th = df_full[POSTER_RATIO_COLS].quantile(0.95)
    keep = pd.Series(True, index=df_full.index)
    for c in POSTER_RATIO_COLS:
        keep &= df_full[c] < th[c]
    return keep


def main():
    df = pd.read_csv(INPUT_PATH)

    # Compute the two new ratios directly from existing columns.
    ratio_cols = []
    for e in ELECTRODES:
        df[f"{e}_delta_beta_ratio"] = df[f"{e}_delta"] / df[f"{e}_beta"]
        df[f"{e}_delta_totalpower_ratio"] = df[f"{e}_delta"] / df[f"{e}_total_power"]
        ratio_cols += [f"{e}_delta_gamma_ratio", f"{e}_delta_beta_ratio", f"{e}_delta_totalpower_ratio"]

    poster_keep = poster_cohort_keep_mask(df) if POSTER_COHORT else None
    df = df[METADATA_COLS + ratio_cols].copy()
    n_start = len(df)

    # Single-pass union exclusion (not sequential/compounding).
    thresholds = df[ratio_cols].quantile(0.95)
    outlier_masks = pd.DataFrame({col: (df[col] > thresholds[col]) for col in ratio_cols})
    union_outlier_mask = outlier_masks.any(axis=1)
    if POSTER_COHORT:
        union_outlier_mask = ~poster_keep.loc[df.index]
        print(f"[POSTER_COHORT=True] Using the poster rule (8 ratio columns) instead of the 12-column rule: "
              f"{int(union_outlier_mask.sum())} / {n_start} seizures excluded.")

    n_dropped = union_outlier_mask.sum()
    print(f"Seizures exceeding at least one ratio's 95th percentile: "
          f"{n_dropped} / {n_start} ({100 * n_dropped / n_start:.1f}%)")

    df = df[~union_outlier_mask].copy()
    print(f"Kept {len(df)} / {n_start} seizures.")

    for col in ratio_cols:
        df[f"log_{col}"] = np.log10(df[col] + 1)
        z_col = f"log_{col}_z"
        df[z_col] = (df[f"log_{col}"] - df[f"log_{col}"].mean()) / df[f"log_{col}"].std()

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()