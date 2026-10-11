import pandas as pd
import numpy as np


df = pd.read_csv("/Users/maya/Documents/lab_project/final_tables/final_output.csv")
print(f"[shape] final_output.csv (input):  {df.shape[0]} rows x {df.shape[1]} columns")
metadata_cols = ["pat_num",	"classif.",	"FBTCS_code",	"vigilance", "vigilance_code", "lobe",	"temporal_code", "side"]

feature_cols = ['T7_delta_gamma_ratio', 'T7_sigma_beta_ratio',
                'T8_delta_gamma_ratio', 'T8_sigma_beta_ratio',
                'PZ_delta_gamma_ratio', 'PZ_sigma_beta_ratio',
                'FZ_delta_gamma_ratio', 'FZ_sigma_beta_ratio']

# Raw (non-ratio) band power for the delta/gamma decomposition analysis --
# lets us test whether delta and gamma each carry an independent effect,
# rather than only ever looking at their ratio.
raw_band_cols = ['T7_delta', 'T7_gamma', 'T8_delta', 'T8_gamma',
                  'PZ_delta', 'PZ_gamma', 'FZ_delta', 'FZ_gamma']

ratio_cols = list(feature_cols)                 # the 8 ORIGINAL poster features
all_cols = ratio_cols + raw_band_cols
df = df[metadata_cols + all_cols]  # removing unnecessary columns
print(f"[shape] after keeping needed columns: {df.shape[0]} rows x {df.shape[1]} columns")
OUT_DIR = "/Users/maya/Documents/lab_project/final_tables/"


def build(df, filter_cols, out_name):
    """Drop the top 5% (thresholds from the FULL data, single pass) on filter_cols only;
    log10(x+1) is added for ALL columns."""
    th = df[filter_cols].quantile(0.95)
    keep = pd.Series(True, index=df.index)
    for c in filter_cols:
        keep &= df[c] < th[c]
    out = df[keep].copy()
    for c in all_cols:
        out[f"log_{c}"] = np.log10(out[c] + 1)
    out.to_csv(OUT_DIR + out_name)
    print(f"[shape] {out_name}: {out.shape[0]} rows x {out.shape[1]} columns "
          f"({len(out)} of {len(df)} rows kept; filter on {len(filter_cols)} columns; "
          f"removed {len(df) - len(out)})")
    if "vigilance" in out.columns:
        n_unclear = (out["vigilance"].astype(str).str.lower() == "unclear").sum()
        print(f"         of which vigilance=='unclear': {n_unclear} "
              f"-> rows used in the poster figures: {len(out) - n_unclear}")


# 1) POSTER COHORT: exclusion on the 8 original ratio columns only (should be ~1029 after
#    the vigilance=='unclear' filter). Raw band columns are logged but do NOT remove rows.
build(df, ratio_cols, "clean_final_output_log10.csv")

# 2) STRICT cohort: exclusion on all 16 columns (what the earlier decomposition/LOPO/topography
#    runs used by accident). Kept separately for a sensitivity check.
build(df, all_cols, "clean_final_output_log10_strict16.csv")