"""
prepare_patient_normalized.py

Computes two normalized versions of log_PZ_delta_gamma_ratio, in
addition to the existing population-level z-score:

  1. patient_centered: each seizure's ratio minus that PATIENT's own
     average ratio (computed from their OTHER seizures only -- leave-
     one-out, so a seizure never counts toward its own baseline).
  2. patient_stage_centered: same idea, but the baseline is specific to
     that patient's OTHER seizures in the SAME sleep stage.

IMPORTANT LIMITATION: there's no separate resting/baseline EEG in this
dataset -- only pre-ictal windows from seizures themselves. So "a
patient's baseline" here necessarily means "this patient's other
seizures," not a true seizure-free reference. Patients with only one
seizure (or only one seizure in a given sleep stage, for the
stage-specific version) have NO valid leave-one-out baseline and are
marked accordingly (NaN) rather than silently given a degenerate zero.

Usage:
    python prepare_patient_normalized.py
"""

import pandas as pd
import numpy as np

DATA_PATH = "/Users/maya/Documents/lab_project/final_tables/clean_final_output_log10.csv"
OUTPUT_PATH = "/Users/maya/Documents/lab_project/final_tables/patient_normalized.csv"

# Same recode stats.py uses, so sleep-stage grouping is consistent with
# the rest of the project.
VIG_MAP = {
    'sleep stage i': 'NREM 1', 'sleep stage 1': 'NREM 1',
    'sleep stage ii': 'NREM 2', 'sleep stage 2': 'NREM 2',
    'sleep stage iii': 'NREM 3', 'sleep stage 3': 'NREM 3',
    'sleep stage iv': 'NREM 3', 'sleep stage 4': 'NREM 3',
}


def recode_vigilance(v):
    v = str(v).strip().lower()
    return VIG_MAP.get(v, str(v).strip())


def leave_one_out_mean(group):
    """For each row, the mean of all OTHER rows in its group.
    NaN if the group has fewer than 2 members (no valid baseline)."""
    n = len(group)
    if n < 2:
        return pd.Series([np.nan] * n, index=group.index)
    total = group.sum()
    return (total - group) / (n - 1)


def main():
    df = pd.read_csv(DATA_PATH)
    df = df[df['vigilance'].str.lower() != 'unclear'].copy()
    df['vigilance_recode'] = df['vigilance'].apply(recode_vigilance)

    col = 'log_PZ_delta_gamma_ratio'

    # --- Per-patient leave-one-out baseline ---
    df['patient_loo_baseline'] = df.groupby('pat_num')[col].transform(leave_one_out_mean)
    df['patient_centered'] = df[col] - df['patient_loo_baseline']

    n_no_patient_baseline = df['patient_loo_baseline'].isna().sum()
    print(f"Seizures with NO valid per-patient baseline (patient has only "
          f"1 seizure total): {n_no_patient_baseline} / {len(df)}")

    # --- Per-patient x sleep-stage leave-one-out baseline ---
    df['patient_stage_loo_baseline'] = (
        df.groupby(['pat_num', 'vigilance_recode'])[col].transform(leave_one_out_mean)
    )
    df['patient_stage_centered'] = df[col] - df['patient_stage_loo_baseline']

    n_no_stage_baseline = df['patient_stage_loo_baseline'].isna().sum()
    print(f"Seizures with NO valid per-patient-per-stage baseline (only "
          f"1 seizure for that patient in that stage): {n_no_stage_baseline} / {len(df)}")

    # z-score the two centered versions (only over rows where they exist)
    for centered_col in ['patient_centered', 'patient_stage_centered']:
        valid = df[centered_col].notna()
        df.loc[valid, f'{centered_col}_z'] = (
            (df.loc[valid, centered_col] - df.loc[valid, centered_col].mean())
            / df.loc[valid, centered_col].std()
        )

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
