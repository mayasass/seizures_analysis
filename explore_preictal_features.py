import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import numpy as np

preictal_bands = pd.read_csv('D:/seizures_analysis/output/all_spectrum_seizures_analysis_v2.csv_tests')

figure,ax = plt.subplots(1,1, figsize=(15, 5))

sns.violinplot(preictal_bands, x="classif.", y="low_delta_power",hue="vigilance", ax=ax)

figure.savefig('D:/seizures_analysis/output/low_delta_power_vio.svg')

temporal_delta_gamma = pd.read_csv('D:\seizures_analysis\output\lobe_analysis\spectrum_analysis_temporal_20241226_133945.csv_tests')
temporal_delta_gamma = temporal_delta_gamma[temporal_delta_gamma['vigilance'] != 'unclear']

fig,axes = plt.subplots(3,1, figsize=(15, 15))
fig.tight_layout(pad=71.0)

sns.set_style("whitegrid")

sns.violinplot(data=temporal_delta_gamma,
               x="classif.",
               y="low_delta_power",
               hue="vigilance",
               ax=axes[0])
axes[0].set_title('Low Delta Power', pad=20, fontsize=14)
axes[0].set_xlabel('Classification', fontsize=12)
axes[0].set_ylabel('Power', fontsize=12)

sns.violinplot(data=temporal_delta_gamma,
               x="classif.",
               y="low_gamma_power",
               hue="vigilance",
               ax=axes[1])
axes[1].set_title('Low Gamma Power', pad=20, fontsize=14)
axes[1].set_xlabel('Classification', fontsize=12)
axes[1].set_ylabel('Power', fontsize=12)

sns.violinplot(data=temporal_delta_gamma,
               x="classif.",
               y="high_gamma_power",
               hue="vigilance",
               ax=axes[2])
axes[2].set_title('High Gamma Power', pad=20, fontsize=14)
axes[2].set_xlabel('Classification', fontsize=12)
axes[2].set_ylabel('Power', fontsize=12)

# Save the figure
plt.savefig('D:/seizures_analysis/output/gamma_delta_power_vio.svg',
            bbox_inches='tight',
            dpi=300)
plt.close()


