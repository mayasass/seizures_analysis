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

plt.figure()
plt.hlines(y, OR_l, OR_u)
plt.plot(OR, y, 'o')
plt.axvline(1, linestyle='--')
plt.yticks(y, names)
plt.xlabel('Odds ratio (95% CI)')
plt.title('Forest Plot of Fixed Effects')
plt.tight_layout()
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

coef_df = result.summary().tables[0].data[1:]  # skip header
coef_df = pd.DataFrame(coef_df, columns=result.summary().tables[0].data[0])
coef_df["Coef."] = coef_df["Coef."].astype(float)
coef_df["[0.025"] = coef_df["[0.025"].astype(float)
coef_df["0.975]"] = coef_df["0.975]"].astype(float)

plt.figure(figsize=(8, 6))
sns.pointplot(
    y=coef_df[""],
    x=coef_df["Coef."],
    join=False,
    color="black",
    errwidth=1,
    capsize=0.1
)
for i, row in coef_df.iterrows():
    plt.plot([row["[0.025"], row["0.975]"]], [i, i], color='gray', lw=2)

plt.axvline(0, color='red', linestyle='--')
plt.title("Fixed Effects Estimates with 95% CI")
plt.xlabel("Coefficient (log-odds)")
plt.ylabel("Predictor")
plt.tight_layout()
plt.show()
