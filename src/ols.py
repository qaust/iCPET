import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import pickle
from tqdm import tqdm

# Custom / Lcoal
from config import model_config
from regression import reg

# Stats
from scipy.stats import shapiro
from sklearn.linear_model import (
	LinearRegression, LassoCV
)
from sklearn.model_selection import (
	cross_val_score, RepeatedKFold
)
import statsmodels.api as sm

# Global vars
SEED = 123
TEST_SIZE = 0.25
HEATMAP_COLORS = sns.diverging_palette(h_neg=250, h_pos=359, as_cmap=True)
SIGNIFICANCE_CUTOFF = 0.05
CV_FOLDS = 5
CUSTOM_CV = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=10, random_state=SEED)


# Cov type
# Options:
#  - robust
#  - clustered
COV_TYPE = 'robust'

def get_params(model, X, y):
    """Returns pd.Series of coefs for comparison with statsmodels params."""
    model.fit(X, y)
    coef = pd.Series(model.coef_, index=model.feature_names_in_)
    coef['const'] = model.intercept_
    return coef.sort_values()

def model_residual_correlation(model):
    """Returns measure of correlation."""
    return np.corrcoef(np.arange(len(model.resid)), model.resid)[1, 0]

def fit_model(X, y):
    """Fit statsmodels OLS model with robust SEs and sklearn OLS model."""
    # Fit statsmodels model for pvalues and coef
    model_sm = sm.OLS(y.copy(), X.copy()).fit(cov_type='HC3')
    # if COV_TYPE == 'robust':
    #     model_sm = sm.OLS(y, X).fit(cov_type='HC3')
    # elif COV_TYPE == 'clustered':
    #     model_sm = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': pe_numbers})
    # Define sklearn model for CV evaluation
    model_sk = LinearRegression(fit_intercept=True, n_jobs=-1)
    # Check that model params match
    sk_model_params = get_params(model_sk, X.copy(), y.copy())
    sm_model_params = model_sm.params.sort_values()
    params_match = np.isclose(sk_model_params, sm_model_params, atol=1e-5)
    if not np.all(params_match):
        print(f"Regressions on {y.name} did not match for sklearn and statsmodels. CV scores may differ.")
    return model_sm, model_sk

def store_model_results(model_sm, model_sk, X, y):
    """
    Params:
        - model_sm: statsmodel model for coefs, pvalues, and residuals.
        - model_sk: sklearn model for cross validation
        - X: X data.
        - y: y data.
    """
    # Calculate CV scores
    cv_scores = cross_val_score(
        model_sk, X, y, 
        scoring='r2', 
        cv=CUSTOM_CV, n_jobs=-1
    )
    # Store model results
    model_results = pd.DataFrame(
        {
            'y': y.name,
            'model_dfn': [tuple(X.columns.values)],
            'nobs': model_sm.nobs,
            'shapiro_resid_pvalue': shapiro(model_sm.resid).pvalue,
            'metric_train': model_sk.score(X, y),
            'metric_cv_mean': np.mean(np.maximum(cv_scores, np.zeros_like(cv_scores))),
            'metric_cv_std': np.std(np.maximum(cv_scores, np.zeros_like(cv_scores))),
            'fpvalue': model_sm.f_pvalue
        }
    )
    # Set model index
    model_results = model_results.set_index(['y', 'model_dfn'])
    return model_results

def store_coef_results(model_sm, y):
    """
    Params:
        - model_sm: statsmodel model for coefs, pvalues, and residuals.
        - y: y data.
    """
    results = pd.DataFrame(
        {
            'model_dfn': [tuple(model_sm.params.index) for _ in range(len(model_sm.params))],
            'coef': model_sm.params, 
            'pval': model_sm.pvalues,
        },
    )
    results['signif'] = results['pval'].apply(reg.add_significance)
    results = results.reset_index(names='x')
    results['y'] = y.name
    results = results.pivot(index=['y', 'model_dfn'], columns=['x'], values=['coef', 'pval', 'signif'])
    results.columns = ['_'.join(idx) for idx in results.columns]
    return results

def combine_model_results(model_sm, model_sk, X, y):
    model_results = store_model_results(model_sm, model_sk, X, y)
    coef_results = store_coef_results(model_sm, y)
    assert model_results.shape[0] == coef_results.shape[0] 
    combined_results = pd.concat([model_results, coef_results], axis=1)
    return combined_results

def backward_stepwise_selection(X, y, cutoff):
    # Make copies of X, y
    X_temp = sm.add_constant(X.copy())
    y_temp = y.copy()
    
    # Fit initial model
    if COV_TYPE == 'robust':
        model_sm = sm.OLS(y_temp, X_temp).fit(cov_type='HC3')
    elif COV_TYPE == 'clustered':
        model_sm = sm.OLS(y_temp, X_temp).fit(cov_type='cluster', cov_kwds={'groups': pe_numbers})
    coefs = model_sm.params[1:]
    pvals = model_sm.pvalues[1:]
    df_temp = pd.DataFrame({
        'coefs': coefs,
        'pvals': pvals
    })
    current_varlist = list(coefs.index.values)

    # Store progression in a list of lists
    progression = list()
    progression.append(dict(zip(coefs.index.values, zip(coefs.values, pvals.values))))
    
    # Iterate until all are stat signif
    while not np.all(df_temp['pvals'] < cutoff):
        
        # Drop the variable with the highest pvalue
        new_vars = df_temp.drop(index=df_temp['pvals'].idxmax()).index.values
        
        # If remaining varlist is empty, break and return the last regression results
        if len(new_vars) == 0:
            break

        # Subset X to new list of variables
        X_temp = sm.add_constant(X_temp.loc[:, new_vars])
        
        # Re-fit model
        model_sm = sm.OLS(y_temp, X_temp).fit(cov_type='HC3')
        coefs = model_sm.params[1:]
        pvals = model_sm.pvalues[1:]
        df_temp = pd.DataFrame({
            'coefs': coefs,
            'pvals': pvals
        })
        progression.append(dict(zip(coefs.index.values, zip(coefs.values, pvals.values))))
        current_varlist = [var for var in model_sm.params.index.values if var != 'const']
    
    return current_varlist, progression

def main(cov_type='robust'):

    with open(Path('../data/prediction_data.pkl'), 'rb') as f:
        data = pickle.load(f)

    X = data.get('X')
    y = data.get('y')
    body_features = data.get('body_features')
    cardio_features = data.get('cardio_features')
    control_features = data.get('controls')
    all_features = body_features + cardio_features + control_features

    print(X.shape)
    print(y.shape)
    print(body_features)
    print(cardio_features)
    print(control_features)

    univariable_results = pd.DataFrame()

    # UNIVARIATE REGRESSIONS
    for target in tqdm(model_config.num_targets):
        for feature in all_features:
            # Fit model
            X_temp = sm.add_constant(X[feature])
            y_temp = y.loc[:, target]
            model_sm, model_sk = fit_model(X_temp, y_temp)

            # Store results
            univariable_results = pd.concat(
                [univariable_results, combine_model_results(model_sm, model_sk, X_temp, y_temp)],
                axis=0
            )
        univariable_results = univariable_results.reset_index()
        univariable_results['selection_method'] = 'All'
        univariable_results['model_dfn'] = univariable_results['model_dfn'].apply(lambda x: x[1])
        univariable_results['category'] = 'univariable_' + univariable_results['model_dfn']
        univariable_results['controls'] = 'None'
        univariable_results.index = univariable_results[['category', 'selection_method', 'y', 'controls']].apply('%'.join, axis=1)
        univariable_results.index.name = 'Lookup'

    # UNIVARIATE REGRESSIONS (CONTROL FOR GENDER)
    univariable_gender_results = pd.DataFrame()

    for target in tqdm(model_config.num_targets):
        for feature in all_features:

            if feature in model_config.controls_encoded:
                continue
                
            # Fit model
            features = [feature, 'gender_cl_Male']
            X_temp = sm.add_constant(X[features])
            y_temp = y.loc[:, target]
            model_sm, model_sk = fit_model(X_temp, y_temp)

            # Store results
            univariable_gender_results = pd.concat(
                [univariable_gender_results, combine_model_results(model_sm, model_sk, X_temp, y_temp)],
                axis=0
            )
    univariable_gender_results = univariable_gender_results.reset_index()
    univariable_gender_results['selection_method'] = 'All'
    univariable_gender_results['model_dfn'] = univariable_gender_results['model_dfn'].apply(lambda x: x[1])
    univariable_gender_results['category'] = 'univariable_' + univariable_gender_results['model_dfn']
    univariable_gender_results['controls'] = 'gender'
    univariable_gender_results.index = univariable_gender_results[['category', 'selection_method', 'y', 'controls']].apply('%'.join, axis=1)
    univariable_gender_results.index.name = 'Lookup'
    
    # UNIVARIATE REGRESSIONS (CONTROL FOR AGE)
    univariable_age_results = pd.DataFrame()

    for target in tqdm(model_config.num_targets):
        for feature in all_features:

            if feature in model_config.controls_encoded:
                continue
                
            # Fit model
            features = [feature, 'age']
            X_temp = sm.add_constant(X[features])
            y_temp = y.loc[:, target]
            model_sm, model_sk = fit_model(X_temp, y_temp)

            # Store results
            univariable_age_results = pd.concat(
                [univariable_age_results, combine_model_results(model_sm, model_sk, X_temp, y_temp)],
                axis=0
            )
    univariable_age_results = univariable_age_results.reset_index()
    univariable_age_results['selection_method'] = 'All'
    univariable_age_results['model_dfn'] = univariable_age_results['model_dfn'].apply(lambda x: x[1])
    univariable_age_results['category'] = 'univariable_' + univariable_age_results['model_dfn']
    univariable_age_results['controls'] = 'age'
    univariable_age_results.index = univariable_age_results[['category', 'selection_method', 'y', 'controls']].apply('%'.join, axis=1)
    univariable_age_results.index.name = 'Lookup'

    # MULTIVARIABLE FEATURE SELECTION AND REGRESSION
    MAX_NUM_REGRESSORS = y.shape[0] // 10
    print(f"MAX_NUM_REGRESSORS: {MAX_NUM_REGRESSORS}")

    multivariable_results = pd.DataFrame()

    for target in tqdm(model_config.num_targets):
        
        low_alpha = -3
        high_alpha = 4
        
        lassoCV = LassoCV(
            alphas=np.logspace(low_alpha, high_alpha, 100),
            cv=CUSTOM_CV,
            fit_intercept=True,
            max_iter=100_000,
            tol=0.001,
            n_jobs=-1
        )

        exceeds_max_num_regressors = True
        while exceeds_max_num_regressors:
            lassoCV.fit(X, y[target])
            coefs = pd.DataFrame(
                {'coef': lassoCV.coef_},
                index=lassoCV.feature_names_in_
            )
            remaining_features_lasso = coefs.loc[~np.isclose(coefs['coef'], 0.0), :].index.values

            if len(remaining_features_lasso) > MAX_NUM_REGRESSORS:
                low_alpha += 0.2
                lassoCV.set_params(**{'alphas': np.logspace(low_alpha, high_alpha, 50)})
            else:
                exceeds_max_num_regressors = False
                
        print(f"{target:<20s} alpha={lassoCV.alpha_:.3f}, feats: {remaining_features_lasso}")
        
        # Fit models
        X_temp_lasso = sm.add_constant(X[remaining_features_lasso])
        y_temp = y[target]
        model_sm_lasso, model_sk_lasso = fit_model(X_temp_lasso, y_temp)

        # Collect model/coef information and store
        model_eval = store_model_results(model_sm_lasso, model_sk_lasso, X_temp_lasso, y_temp)
        model_coefs = store_coef_results(model_sm_lasso, y_temp)
        model_results = pd.concat([model_eval, model_coefs], axis=1)
        multivariable_results = pd.concat([multivariable_results, model_results], axis=0)

        multivariable_results = multivariable_results.reset_index()
        multivariable_results['selection_method'] = 'LassoCV'
        multivariable_results['category'] = 'composite'
        multivariable_results['controls'] = 'None'
        multivariable_results.index = multivariable_results[['category', 'selection_method', 'y', 'controls']].apply('%'.join, axis=1)
        multivariable_results.index.name = 'Lookup'

    ols_results = pd.concat(
        [
            univariable_results, 
            univariable_gender_results,
            univariable_age_results, 
            multivariable_results
        ], axis=0
    )
    if cov_type == 'robust':
        ols_results.to_csv('../output/regressions/ols_results_robust.csv')
    elif cov_type == 'clustered':
        ols_results.to_csv('../output/regressions/ols_results_clustered.csv')


if __name__=='__main__':
    main()