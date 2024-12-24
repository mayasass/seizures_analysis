import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

preictal_bands = pd.read_csv('D:/seizures_analysis/output/all_spectrum_seizures_analysis_v2.csv')

figure,ax = plt.subplots(1,1, figsize=(15, 5))

sns.violinplot(preictal_bands, x="classif.", y="low_delta_power",hue="vigilance", ax=ax)

figure.savefig('D:/seizures_analysis/output/low_delta_power_vio.svg')