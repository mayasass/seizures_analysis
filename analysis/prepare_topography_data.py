"""
prepare_topography_data.py

statistical_anaylsis.py / statistical_anaylsis_updated.py only keep 4
electrodes (T7, T8, PZ, FZ). The topography check needs ALL 19 to compare
"over the seizure's known lobe" vs "everywhere else on the scalp" -- so
this is a separate cleaning pass over final_output.csv, all electrodes.

ON THE EXCLUSION METHOD (this matters, see chat discussion):
The original statistical_anaylsis.py drops a row the moment ANY one
feature exceeds its own 95th percentile, applied one feature at a time,
where each subsequent filter is applied to the ALREADY-reduced dataframe
from the previous one. Doing that naively across 19 electrodes (instead
of the original 4) risks heavy compounding if each electrode's outliers
were an independent ~5% of the data.

This version instead does a SINGLE-PASS union exclusion: compute all 19
electrodes' 95th-percentile thresholds on the full, unfiltered data
first, then drop a seizure once if it exceeds ANY of them. This
preserves the original intent (fully remove seizures with a likely
artifact) without compounding, and the printed overlap stats below tell
you empirically how much these artifact-seizures overlap across
electrodes -- rather than assuming it either way.

Usage:
    python prepare_topography_data.py
"""

import pandas as pd
import numpy as np

INPUT_PATH = "/Users/maya/Documents/lab_project/final_tables/final_output.csv"
OUTPUT_PATH = "/Users/maya/Documents/lab_project/final_tables/topography_all_electrodes_log10.csv"

ELECTRODES = ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'C3', 'C4',
              'P3', 'P4', 'P7', 'P8', 'O1', 'O2', 'FZ', 'CZ', 'PZ', 'T7', 'T8']

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

    ratio_cols = [f"{e}_delta_gamma_ratio" for e in ELECTRODES]
    missing = [c for c in ratio_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in final_output.csv: {missing}")

    poster_keep = poster_cohort_keep_mask(df) if POSTER_COHORT else None
    df = df[METADATA_COLS + ratio_cols].copy()
    n_start = len(df)

    # Compute all thresholds on the FULL, unfiltered data first (not
    # sequentially on a shrinking dataframe), then figure out, per
    # electrode, which specific seizures exceed it.
    thresholds = df[ratio_cols].quantile(0.95)
    outlier_masks = {col: (df[col] > thresholds[col]) for col in ratio_cols}

    # How much do these outlier sets actually overlap across electrodes?
    # This is the empirical check for whether "any one electrode extreme"
    # tends to mean "this whole seizure is contaminated" (high overlap,
    # union close to ~5%) vs. "different electrodes go extreme for
    # unrelated reasons" (low overlap, union could approach 19x5%).
    per_seizure_outlier_count = pd.DataFrame(outlier_masks).sum(axis=1)
    print("How many electrodes (of 19) each outlier seizure exceeded:")
    print(per_seizure_outlier_count[per_seizure_outlier_count > 0].value_counts().sort_index().to_string())

    union_outlier_mask = per_seizure_outlier_count > 0
    if POSTER_COHORT:
        union_outlier_mask = ~poster_keep.loc[df.index]
        print(f"\n[POSTER_COHORT=True] Using the poster rule (8 ratio columns) instead of the 19-electrode rule above: "
              f"{int(union_outlier_mask.sum())} / {n_start} seizures excluded.")
    n_dropped = union_outlier_mask.sum()
    print(f"\nTotal seizures exceeding at least one electrode's threshold: "
          f"{n_dropped} / {n_start} ({100 * n_dropped / n_start:.1f}%)")
    print(f"(if fully independent across 19 electrodes, this could have been "
          f"as high as ~{100 * (1 - 0.95**19):.0f}%; if fully overlapping, "
          f"as low as ~5%)")

    df = df[~union_outlier_mask].copy()
    print(f"\nKept {len(df)} / {n_start} seizures after single-pass union exclusion.")

    for col in ratio_cols:
        df[f"log_{col}"] = np.log10(df[col] + 1)

    for e in ELECTRODES:
        col = f"log_{e}_delta_gamma_ratio"
        df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std()

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} seizures x {len(ELECTRODES)} electrodes to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()