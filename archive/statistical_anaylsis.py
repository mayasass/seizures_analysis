import pandas as pd
import numpy as np


df = pd.read_csv("/Users/maya/Documents/lab_project/final_tables/final_output.csv")
metadata_cols = ["pat_num",	"classif.",	"FBTCS_code",	"vigilance", "vigilance_code", "lobe",	"temporal_code", "side"]

feature_cols = ['T7_delta_gamma_ratio', 'T7_sigma_beta_ratio',
                'T8_delta_gamma_ratio', 'T8_sigma_beta_ratio',
                'PZ_delta_gamma_ratio', 'PZ_sigma_beta_ratio',
                'FZ_delta_gamma_ratio', 'FZ_sigma_beta_ratio']

df = df[metadata_cols + feature_cols]  # removing unnecessary columns
th = df[feature_cols].quantile(0.95)

for feature in feature_cols:

    max_value = th[feature]
    df = df[df[feature] < max_value]
    df[f"log_{feature}"] = np.log10(df[feature] + 1)

df.to_csv("/Users/maya/Documents/lab_project/final_tables/clean_final_output_log10.csv")



