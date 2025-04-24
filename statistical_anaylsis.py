import pandas as pd
import numpy as np


df = pd.read_csv("D:/seizures_analysis/output/data_for_testing_23_04.csv")
metadata_cols = ['pat_num', 'is_FBTCS', 'vigilance', 'vigilance_code', 'classif.']

feature_cols = ['T7_delta_gamma_ratio', 'T7_sigma_beta_ratio',
                'T8_delta_gamma_ratio', 'T8_sigma_beta_ratio',
                'PZ_delta_gamma_ratio', 'PZ_sigma_beta_ratio',
                'FZ_delta_gamma_ratio', 'FZ_sigma_beta_ratio']

df = df[metadata_cols + feature_cols]

th = df[feature_cols].quantile(0.95)

for feature in feature_cols:

    max_value = th[feature]
    df = df[df[feature] < max_value ]
    print(df.groupby('is_FBTCS').count()['FZ_sigma_beta_ratio'])
    print(df[feature].min())
    df[f"log_{feature}"] = np.log(df[feature] + 1)

df.to_csv("D:/seizures_analysis/output/data_for_testing_clean.csv")