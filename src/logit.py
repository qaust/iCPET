import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
import os
import pickle
from tqdm.notebook import trange, tqdm
from config import model_config

from scipy.stats import shapiro
from sklearn.ensemble import (
	RandomForestClassifier
)
from sklearn.feature_selection import(
	RFECV, SequentialFeatureSelector
)
from sklearn.linear_model import (
	LinearRegression, LogisticRegression,
    LogisticRegressionCV
)
from sklearn.metrics import (
	confusion_matrix, classification_report, f1_score,
	roc_curve, roc_auc_score, auc, RocCurveDisplay
)
from sklearn.model_selection import (
	train_test_split, RandomizedSearchCV, GridSearchCV, 
	cross_val_score, cross_val_predict, KFold, StratifiedKFold,
    RepeatedStratifiedKFold
)
from sklearn.pipeline import (
	Pipeline
)
from sklearn.preprocessing import (
	LabelEncoder, OneHotEncoder, StandardScaler,
	RobustScaler, QuantileTransformer,
)
import statsmodels.api as sm
from xgboost import XGBClassifier

from regression import reg


SEED = 123
TEST_SIZE = 0.25
CV_FOLDS = 5
CUSTOM_CV = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=10, random_state=SEED)

HEATMAP_COLORS = sns.diverging_palette(h_neg=359, h_pos=250, as_cmap=True)
plt.style.use('ggplot')

USE_INITIAL = True
USE_CLUSTERED_SE = False

def get_sorted_params(fitted_model):
    """Returns pd.Series of coefs for comparison with statsmodels params."""
    coef = pd.Series(
        np.array(fitted_model.coef_).flatten(), 
        index=np.array(fitted_model.feature_names_in_).flatten()
    )
    # print(fitted_model.get_params().get('fit_intercept'))
    if fitted_model.get_params().get('fit_intercept'):
        coef['const'] = fitted_model.intercept_[0]
        
    return coef.sort_index()

def check_params_equal(model_sm, model_sk):
    """Checks whether the coefficients from an sklearn and statsmodel regression are the same"""
    sorted_params = get_sorted_params(model_sk)
    coefs_are_equal = np.all(np.isclose(sorted_params, model_sm.params.sort_index(), atol=1e-04))
    return coefs_are_equal

def model_residual_correlation(model):
    """Returns measure of correlation."""
    return np.corrcoef(np.arange(len(model.resid)), model.resid)[1, 0]

def fit_model(X, y):
    """Fit statsmodels OLS model with robust SEs and sklearn OLS model."""
    
    # Fit statsmodels
    model_sm = sm.GLM(y.copy(), sm.add_constant(X.copy()), family=sm.families.Binomial())
    if USE_CLUSTERED_SE:
        model_sm = model_sm.fit(cov_type='cluster', cov_kwds={'groups': pe_numbers})
    else: 
        model_sm = model_sm.fit(cov_type='HC3')

    # Fit sklearn 
    model_sk = LogisticRegression(
        random_state=SEED,
        fit_intercept=True,
        max_iter=5_000, 
        penalty=None, 
        solver='lbfgs',
    )
    model_sk.fit(X.copy(), y.copy())

    # Check coefs equal
    params_are_equal = check_params_equal(model_sm, model_sk)
    if not params_are_equal:
        print("\nModels did not have same coefs")
        print(get_sorted_params(model_sk))
        print(model_sm.params.sort_index())
        print("---------------------------------")
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
        scoring='roc_auc', 
        cv=CUSTOM_CV, n_jobs=-1
    )
    # Store model results
    model_results = pd.DataFrame(
        {
            'y': y.name,
            'model_dfn': [('const',) + tuple(X.columns.values)],
            'nobs': model_sm.nobs,
            'shapiro_resid_pvalue': np.nan,
            'metric_train': model_sk.score(X, y),
            'metric_cv_mean': np.mean(cv_scores),
            'metric_cv_std': np.std(cv_scores),
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

def main(use_initial=True, use_clustered_se=False):
    if use_initial:
        with open(Path('../data/classification_data_initial.pkl'), 'rb') as f:
            data = pickle.load(f)
    else:
        with open(Path('../data/classification_data_all.pkl'), 'rb') as f:
            data = pickle.load(f)
        
    X = data.get('X')
    y = data.get('y').squeeze()
    body_features = data.get('body_features')
    cardio_features = data.get('cardio_features')
    control_features = data.get('controls')
    clot_features = data.get('clot_features')
    all_features = body_features + cardio_features + control_features + clot_features

    print(X.shape)
    print(y.shape)
    print(body_features)
    print(cardio_features)
    print(control_features)
    print(clot_features)

    # UNIVARIABLE REGRESSIONS
    univariable_results = pd.DataFrame()

    for feature in tqdm(all_features):

        X_temp = X[[feature]]
        y_temp = y.copy()
        model_sm, model_sk = fit_model(X_temp, y_temp)

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

    # UNIVARIABLE REGRESSIONS (CONTROL FOR AGE)
    univariable_age_results = pd.DataFrame()

    for feature in tqdm(all_features):
        
        if feature in model_config.controls_encoded:
            continue
            
        X_temp = X[[feature, 'age']]
        y_temp = y.copy()
        model_sm, model_sk = fit_model(X_temp, y_temp)

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

    # UNIVARIABLE REGRESSIONS (CONTROL FOR GENDER)
    univariable_gender_results = pd.DataFrame()

    for feature in tqdm(all_features):
        
        if feature in model_config.controls_encoded:
            continue
            
        X_temp = X[[feature, 'gender_cl_Male']]
        y_temp = y.copy()
        model_sm, model_sk = fit_model(X_temp, y_temp)

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

    # MULTIVARIABLE FEATURE SELECTION AND REGRESSION
    model_dfns = [
        control_features, 
        body_features, 
        cardio_features, 
        clot_features, 
        all_features
    ]
    model_dfn_names = [
        'demo', 
        'body', 
        'cardio', 
        'clot', 
        'all'
    ]
    model_dfns_remaining = dict()

    MAX_NUM_REGRESSORS = len(y) // 10
    print(f"MAX_NUM_REGRESSORS: {MAX_NUM_REGRESSORS}")

    y_temp = y.copy()

    multivariable_results = pd.DataFrame()
    for i, feats in enumerate(model_dfns):

        low_Cs = -2
        high_Cs = 2

        logitCV = LogisticRegressionCV(
            Cs=np.logspace(low_Cs, high_Cs, 50), 
            cv=CUSTOM_CV, 
            penalty='l1', 
            solver='liblinear', 
            max_iter=5_000, 
            scoring='roc_auc',
            fit_intercept=True,
            random_state=SEED,
            n_jobs=-1
        )
            
        # Select features
        X_init = X.loc[:, feats]

        more_or_less_than_needed = True
        while more_or_less_than_needed:
            logitCV.fit(X_init, y_temp)
            coefs = pd.DataFrame(
                {'coef': np.squeeze(logitCV.coef_)},
                index=logitCV.feature_names_in_
            )
            remaining_features = list(coefs[coefs['coef'] != 0].index.values)
            if len(remaining_features) > MAX_NUM_REGRESSORS:
                high_Cs -= 0.05 / (MAX_NUM_REGRESSORS / (len(remaining_features) - MAX_NUM_REGRESSORS))
            elif len(remaining_features) == 0:
                low_Cs += 0.2
            else:
                more_or_less_than_needed = False
            logitCV.set_params(**{'Cs': np.logspace(low_Cs, high_Cs, 50)})
            print(f"C_range=({10**low_Cs:.3f}, {10**high_Cs:.3f}), C={logitCV.C_.item():.3f}, {remaining_features}")

        print(f"{i}, {logitCV.C_.item():.3f}, {remaining_features}")
        print("---------------------------------------------------------")
        model_dfns_remaining[model_dfn_names[i]] = remaining_features
        
        # Fit model with selected features
        X_selected = X.loc[:, remaining_features]
        model_sm, model_sk = fit_model(X_selected, y_temp)

        # Store results
        temp_results = combine_model_results(model_sm, model_sk, X_selected, y_temp)
        temp_results['category'] = 'composite_'+ model_dfn_names[i]
        multivariable_results = pd.concat(
            [multivariable_results, temp_results], 
            axis=0
        )

    # MULTIVARIABLE CUSTOM MODEL
    y_temp = y.copy()

    feats = ['a_diameter', 'heart_volume', 'airway_ratio', 'superior_left']
    model_dfns_remaining['composite'] = feats

    # Fit model with selected features
    X_selected = X.loc[:, feats]

    model_sm, model_sk = fit_model(X_selected, y_temp)

    # Store results
    temp_results = combine_model_results(model_sm, model_sk, X_selected, y_temp)
    temp_results['category'] = 'composite_custom'

    multivariable_results = pd.concat(
            [multivariable_results, temp_results], 
            axis=0
        )
    
    # COMBINE MULTIVARIABLE REGRESSION RESULTS
    multivariable_results = multivariable_results.reset_index()
    multivariable_results['selection_method'] = 'LassoCV'
    multivariable_results['controls'] = 'None'
    multivariable_results.index = multivariable_results[['category', 'selection_method', 'y', 'controls']].apply('%'.join, axis=1)
    multivariable_results.index.name = 'Lookup'
    print(multivariable_results.shape)
    multivariable_results.tail()

    # COMBINE ALL REGRESSION RESULTS
    logit_results = pd.concat(
        [
            univariable_results,
            univariable_age_results,
            univariable_gender_results,
            multivariable_results
        ], axis=0
    )

    fname = 'logit_results'
    if use_initial:
        fname += '_initial'
    else: 
        fname += '_all'
    if use_clustered_se:
        fname += '_clustered'
    else: 
        fname += '_robust'
        
    logit_results.to_csv(f'../output/regressions/{fname}.csv')

    print("Done exporting regression results.")

    # ROC CURVES
    model_names = [
        'Demographic Features', 
        'Body Composition Features', 
        'Cardiopulmonary Features', 
        'Clot Features', 
        'Composite'
    ]
    model_dfn_names = [
        'demo', 
        'body', 
        'cardio', 
        'clot', 
        'composite',
    ]


    fig, axs = plt.subplots(figsize=(7, 7))

    for i, model_name in enumerate(model_dfn_names):
        print(model_name)
        
        feat = model_dfns_remaining[model_name]
        X_temp = X[feat].reset_index(drop=True)
        y_temp = y.copy().reset_index(drop=True)
        
        # cv = StratifiedKFold(n_splits=CV_FOLDS)
        cv = CUSTOM_CV
        classifier = LogisticRegression(
                random_state=SEED,
                fit_intercept=True,
                max_iter=1_000, 
                penalty=None, 
                solver='newton-cg',
        )
        
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 500)
        
        for fold, (train, test) in enumerate(cv.split(X_temp, y_temp)):
            try:
                classifier.fit(X_temp.loc[train, :], y[train])
                ffpr, ftpr, fthresh = roc_curve(y_temp[test].squeeze(), classifier.predict_proba(X_temp.loc[test])[:, 1])
                fauc = roc_auc_score(y_temp[test], classifier.predict(X_temp.loc[test, :]))
                interp_tpr = np.interp(mean_fpr, ffpr, ftpr)
                tprs.append(interp_tpr)
                aucs.append(fauc)
            except:
                print(f"Could not fit fold {fold}")
                continue
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_error_auc = np.std(aucs) / np.sqrt(len(aucs))

        n_std_error = 1.96
        
        axs.plot(
            mean_fpr,
            mean_tpr,
            label=f"{model_names[i]} (AUC = {mean_auc:.2f} $\pm$ {n_std_error * std_error_auc:.2f})",
            lw=2,
            alpha=0.8,
        )
        
        std_error_tpr = np.std(tprs, axis=0) / np.sqrt(len(tprs))
        
        tprs_upper = np.minimum(mean_tpr + n_std_error * std_error_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - n_std_error * std_error_tpr, 0)
        axs.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            alpha=0.15,
            # label=f"$\pm$ {n_std_error:.2f} SE",
        )
        
    axs.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        # title='ROC AUC Curves for Composite Models',
    )

    axs.plot([0, 1], [0, 1], "k--", label="Chance Level (AUC = 0.5)")

    axs.axis("square")
    axs.legend(loc="lower right", fontsize=10)

    plt.tight_layout()

    plt.savefig('../figures/roc_curves_2.png')

    print("Done plotting ROC curve.")