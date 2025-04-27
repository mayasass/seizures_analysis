"""
GLMM Analysis of Seizure Generalization (FBTCS_code)
Predictors: log_PZ_delta_gamma_ratio_z, vigilance state, and lobe
Random effects: patient (pat_num)
"""

# ==============================
# Import required packages
# ==============================
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from patsy import dmatrices

# ==============================
# Load dataset
# ==============================
file_path = 'D:/seizures_analysis/output/clean_final_table.csv'
df = pd.read_csv(file_path)

# ==============================
# Prepare data
# ==============================

# Step 1: Recode vigilance
# Merge "sleep stage III" and "sleep stage IV" into "deep sleep"
df['vigilance_recode'] = df['vigilance'].replace({
    'sleep stage III': 'deep sleep',
    'sleep stage IV': 'deep sleep',
    'sleep stage II': 'light sleep',
    'sleep stage I': 'light sleep'

})

# Step 2: Standardize log_PZ_delta_gamma_ratio
df['log_PZ_delta_gamma_ratio_z'] = (
    df['log_PZ_delta_gamma_ratio'] - df['log_PZ_delta_gamma_ratio'].mean()
) / df['log_PZ_delta_gamma_ratio'].std()

# ==============================
# Prepare model matrices
# ==============================

# Create fixed effects design matrix (exog) and outcome (endog)
endog, exog = dmatrices(
    'FBTCS_code ~ log_PZ_delta_gamma_ratio_z',  # C(lobe) + C(vigilance_recode)',
    data=df,
    return_type='dataframe'
)

# Create random effects matrix (random intercepts for each patient)
exog_re = pd.get_dummies(df['pat_num'], drop_first=False)

# Create the identification array: all random effects belong to same variance component
ident = np.ones(exog_re.shape[1], dtype=int)

# ==============================
# Fit the model
# ==============================

# Initialize the Binomial Bayes Mixed GLM
model = BinomialBayesMixedGLM(endog, exog, exog_re, ident)

# Fit the model using Variational Bayes
result = model.fit_vb()

# ==============================
# Summarize the result
# ==============================
summary_text = result.summary()

# Display results
print(summary_text)
