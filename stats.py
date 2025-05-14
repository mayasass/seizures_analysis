import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from patsy import dmatrices, Treatment
from scipy.special import expit
from scipy import stats
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Load and preprocess data ---
# df = pd.read_csv('D:/seizures_analysis/output/clean_final_table.csv_tests')
df = pd.read_csv('/Users/maya/Documents/lab_project/final_tables/clean_final_output.csv')
#df = pd.read_csv('/Users/maya/Documents/lab_project/final_tables/clean_final_table.csv_tests')

# -------recoding-----------------------------------------
# Recode vigilance stages
vig_map = {
    'sleep stage I': 'NREM 1',
    'sleep stage II': 'NREM 2',
    'sleep stage III': 'NREM 3',
    'sleep stage IV': 'NREM 3'
}
df['vigilance_recode'] = df['vigilance'].replace(vig_map)
df = df[df['vigilance_recode'] != 'unclear'].copy()
#df = df[df['vigilance_recode'] != 'NREM 3'].copy()

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

# --------------functions------------------------------------------------------------------------------
def model_single_feature(feature, reference=None):
    if reference is None:
        formula = f'FBTCS_code ~ {feature}_z'
    else:
        # Use vigilance_recode instead of vigilance
        var = 'vigilance_recode' if feature == 'vigilance' else feature
        formula = f'FBTCS_code ~ C({var}, Treatment(reference="{reference}"))'

    endog, exog = dmatrices(formula, data=df, return_type='dataframe')
    exog_re = pd.get_dummies(df['pat_num'], drop_first=False)
    ident = np.ones(exog_re.shape[1], dtype=int)

    model = BinomialBayesMixedGLM(endog, exog, exog_re, ident)
    print(f"{feature}_results")
    summary_table(model, feature)
    return model

def summary_table(model, name):
    result = model.fit_vb(verbose=False)
    coef_df = pd.DataFrame({
        'name': result.model.exog_names,
        'β': result.fe_mean,
        'SD': result.fe_sd,
        'OR': np.exp(result.fe_mean),
        'z': result.fe_mean / result.fe_sd,
        'p': 2 * (1 - stats.norm.cdf(np.abs(result.fe_mean / result.fe_sd)))
    }).round(4)
    coef_df.to_csv(f"test_{name}.csv_tests")

def forest_plot(result,name):
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
    plt.title(f'Forest Plot of {name}')
    plt.tight_layout(pad=1.5)
    plt.savefig(f'forest_plot_{name}_with_quanty')
    plt.show()
    return names


def predicted_probabilities_by_vig(result, name='log_PZ_delta_gamma_ratio_z'):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import expit
    vig_levels = ['awake', 'NREM 1', 'NREM 2', 'NREM 3']
    #vig_levels = ['awake', 'NREM 1', 'NREM 2']
    probs = []

    # Base row: zeros with intercept = 1
    row_base = np.zeros_like(result.fe_mean)
    intercept_name = 'Intercept' if 'Intercept' in result.model.exog_names else result.model.exog_names[0]
    row_base[result.model.exog_names.index(intercept_name)] = 1

    for v in vig_levels:
        row = row_base.copy()

        # For non-reference levels, set dummy variable
        if v != 'awake':
            col_name = f'C(vigilance_recode, Treatment(reference="awake"))[T.{v}]'
            idx = result.model.exog_names.index(col_name)
            row[idx] = 1

        logit = (row * result.fe_mean).sum()
        p = expit(logit)
        probs.append(p)

    # Plotting
    plt.figure()
    plt.bar(vig_levels, probs, color='C1')
    plt.ylim(0, 1)
    plt.ylabel('P(Generalization)')
    plt.title('Predicted Probability by Vigilance State')
    plt.tight_layout()
    plt.savefig(f'vigilance_only_prediction_{timestamp}_with_quanty')
    plt.show()


def predicted_probabilities(result):
    # Generate a grid of z-scored log_PZ_delta_gamma_ratio values
    x_grid = np.linspace(-2.5, 2.5, 100)

    # Prepare row template with correct length and Intercept set
    row_base = np.zeros_like(result.fe_mean)
    intercept_name = 'Intercept' if 'Intercept' in result.model.exog_names else result.model.exog_names[0]
    row_base[result.model.exog_names.index(intercept_name)] = 1

    # Get index for the feature column
    col_log = result.model.exog_names.index('log_PZ_delta_gamma_ratio_z')

    # Compute predicted probabilities over the x_grid
    probs = []
    for x in x_grid:
        row = row_base.copy()
        row[col_log] = x
        logit = (row * result.fe_mean).sum()
        probs.append(expit(logit))

    # Plotting
    plt.figure()
    plt.plot(x_grid, probs, label='PZ ratio effect', color='C0')
    plt.xlabel('log_PZ_delta_gamma_ratio_z')
    plt.ylabel('P(Generalization)')
    plt.title('Predicted Probability by PZ delta/gamma ratio')
    plt.tight_layout()
    plt.savefig(f'curve_PZ_ratio_{timestamp}_with_quanty')
    plt.show()

# ----modeling by features--------------------------------------------------------------------------------------

features = {'log_PZ_delta_gamma_ratio': None,
            'vigilance': 'awake'
            }
for feature, ref in features.items():
    model = model_single_feature(feature, ref)
    result = model.fit_vb(verbose=False)
    forest_plot(result, feature)
    predicted_probabilities_by_vig(result) if feature == "vigilance" else predicted_probabilities(result)


# ----Model all relevant features (with interaction) ------------------------------------------------------------
formula = ('FBTCS_code ~ log_PZ_delta_gamma_ratio_z * '
           'C(vigilance_recode, Treatment(reference="awake")) '
           '+ C(lobe_recode, Treatment(reference="temporal"))')  # C(...) for “categorical”

endog, exog = dmatrices(formula, data=df, return_type='dataframe')  # dependent variable, independent variables
exog_re = pd.get_dummies(df['pat_num'], drop_first=False)  # random effects design matrix
ident = np.ones(exog_re.shape[1], dtype=int)  # identifier for grouping random effects
model = BinomialBayesMixedGLM(endog, exog, exog_re, ident)  # --- Fit Bayesian GLMM ---
result = model.fit_vb(verbose=False)

summary_table(model, "all_features")
names = forest_plot(result, "log_PZ_delta_gamma_ratio_z")

# --- Predicted Probabilities vs Δ/γ by Vigilance --------------------------------------------------------------
x_grid = np.linspace(-2.5, 2.5, 100)
vig_levels = ['awake', 'NREM 1', 'NREM 2', 'NREM 3']
#vig_levels = ['awake', 'NREM 1', 'NREM 2']
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

# plt.axhline(0.5, linestyle=':')
plt.xlabel('log_PZ_delta_gamma_ratio_z')
plt.ylabel('P(Generalization)')
plt.title(f'Predicted Probability by vigilance')
plt.legend()
plt.tight_layout()
plt.savefig(f'probability_predictin_ curve_{timestamp}_with_quanty')
plt.show()

#--------------- coeffcient plot-----------------------------------------------------------
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
plt.savefig(f'coef_{timestamp}_allFixed')
plt.show()

