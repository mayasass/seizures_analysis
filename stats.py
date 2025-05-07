import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from patsy import dmatrices, Treatment
from scipy.special import expit
from scipy import stats
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

# --- Load and preprocess data ---
df = pd.read_csv('D:/seizures_analysis/output/clean_final_table.csv')

# Recode vigilance stages
vig_map = {
    'sleep stage I': 'light NREM',
    'sleep stage II': 'light NREM',
    'sleep stage III': 'deep NREM',
    'sleep stage IV': 'deep NREM'
}
df['vigilance_recode'] = df['vigilance'].replace(vig_map)
df = df[df['vigilance_recode'] != 'unclear'].copy()

# Recode lobes
lobe_map = {
    'frontal': 'frontal',
    'temporal': 'temporal',
    'hemisphere': 'other',
    'none': 'other',
    'unknown': 'other',
    'occipital': 'other',
    'parietal': 'other'
}
df['lobe_recode'] = df['lobe'].replace(lobe_map)

# Standardize log_PZ_delta_gamma_ratio
df['log_PZ_delta_gamma_ratio_z'] = (
    df['log_PZ_delta_gamma_ratio'] - df['log_PZ_delta_gamma_ratio'].mean()
) / df['log_PZ_delta_gamma_ratio'].std()

# --- Model formula (with interaction) ---
formula = ('FBTCS_code ~ log_PZ_delta_gamma_ratio_z * '
           'C(vigilance_recode, Treatment(reference="awake")) '
           '+ C(lobe_recode, Treatment(reference="other"))')  # C(...) for “categorical”

# dependent variable, independent variables
endog, exog = dmatrices(formula, data=df, return_type='dataframe')

# random effects design matrix
exog_re = pd.get_dummies(df['pat_num'], drop_first=False)

# identifier for grouping random effects
ident = np.ones(exog_re.shape[1], dtype=int)

# --- Fit Bayesian GLMM ---
model = BinomialBayesMixedGLM(endog, exog, exog_re, ident)
result = model.fit_vb(verbose=False)

# --- Summary Table ---
coef_df = pd.DataFrame({
    'β': result.fe_mean,
    'SD': result.fe_sd,
    'OR': np.exp(result.fe_mean),
    'z': result.fe_mean / result.fe_sd,
    'p': 2 * (1 - stats.norm.cdf(np.abs(result.fe_mean / result.fe_sd)))
}).round(4)
print(coef_df)

# --- Forest Plot ---
beta = result.fe_mean
se = result.fe_sd
names = result.model.exog_names
mask = [i for i, n in enumerate(names) if n != 'Intercept']
beta = beta[mask]
se = se[mask]
names = [names[i] for i in mask]

OR = np.exp(beta)
OR_l = np.exp(beta - 1.96 * se)
OR_u = np.exp(beta + 1.96 * se)
y = np.arange(len(names))

plt.figure(figsize=(10, 6))
plt.hlines(y, OR_l, OR_u)
plt.plot(OR, y, 'o')
plt.axvline(1, linestyle='--')
plt.yticks(y, names)
plt.xlabel('Odds ratio (95% CI)')
plt.title('Forest Plot of Fixed Effects')
plt.tight_layout(pad=1.5)
plt.show()

# --- Predicted Probabilities vs Δ/γ by Vigilance ---
x_grid = np.linspace(-2.5, 2.5, 100)
vig_levels = ['awake', 'light NREM', 'deep NREM']
plt.figure()

for v in vig_levels:
    row_base = np.zeros_like(result.fe_mean)
    row_base[names.index('Intercept') if 'Intercept' in names else 0] = 1
    if v != 'awake':
        col_name = f'C(vigilance_recode, Treatment(reference="awake"))[T.{v}]'
        row_base[result.model.exog_names.index(col_name)] = 1

    col_log = result.model.exog_names.index('log_PZ_delta_gamma_ratio_z')
    inter_col = None
    if v != 'awake':
        inter_col = result.model.exog_names.index(
            f'log_PZ_delta_gamma_ratio_z:C(vigilance_recode, Treatment(reference="awake"))[T.{v}]')

    probs = []
    for x in x_grid:
        row = row_base.copy()
        row[col_log] = x
        if inter_col is not None:
            row[inter_col] = x
        logit = (row * result.fe_mean).sum()
        probs.append(expit(logit))

    plt.plot(x_grid, probs, label=v)

plt.axhline(0.5, linestyle=':')
plt.xlabel('log_PZ_delta_gamma_ratio_z')
plt.ylabel('P(Generalization)')
plt.title('Predicted Probability by Vigilance State')
plt.legend()
plt.tight_layout()
plt.show()

#--------------- bug fix required
import seaborn as sns
import matplotlib.pyplot as plt

# Get fixed effect names
variable_names = result.model.exog_names
posterior_means = result.fe_mean
posterior_sds = result.fe_sd

print(f"Lengths - names: {len(variable_names)}, means: {len(posterior_means)}, sds: {len(posterior_sds)}")

# Create DataFrame
coef_df = pd.DataFrame({
    'Variable': variable_names,
    'β': posterior_means,
    'SD': posterior_sds
})
coef_df['Lower 95% CI'] = coef_df['β'] - 1.96 * coef_df['SD']
coef_df['Upper 95% CI'] = coef_df['β'] + 1.96 * coef_df['SD']

# Sort variables for cleaner display
coef_df = coef_df.sort_values(by='β')

# Increase figure size to fix tight layout issue
plt.figure(figsize=(10, len(coef_df) * 0.75))
plt.errorbar(coef_df['β'], coef_df['Variable'],
             xerr=1.96 * coef_df['SD'], fmt='o', color='black', ecolor='gray', capsize=4)
plt.axvline(x=0, linestyle='--', color='red')
plt.xlabel("Coefficient (β)")
plt.title("Coefficient Plot with 95% Confidence Intervals")
plt.tight_layout(pad=2.5)  # Increase padding
plt.show()