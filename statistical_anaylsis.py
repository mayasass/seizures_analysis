import pandas as pd
import numpy as np


df = pd.read_csv("/Users/maya/Documents/lab_project/data/final_output.csv_tests")
metadata_cols = ["pat_num",	"classif.",	"FBTCS_code",	"vigilance", "vigilance_code", "lobe",	"temporal_code", "side"]

feature_cols = ['T7_delta_gamma_ratio', 'T7_sigma_beta_ratio',
                'T8_delta_gamma_ratio', 'T8_sigma_beta_ratio',
                'PZ_delta_gamma_ratio', 'PZ_sigma_beta_ratio',
                'FZ_delta_gamma_ratio', 'FZ_sigma_beta_ratio']

df = df[metadata_cols + feature_cols]  # removing unnecessary columns

for feature in feature_cols:
    df[f"log_{feature}"] = np.log(df[feature] + 1)  # normalizing by log

df.to_csv("/Users/maya/Documents/lab_project/data/clean_final_output.csv_tests")


