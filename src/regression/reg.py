import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import (
	LinearRegression, LogisticRegression,
)
from sklearn.metrics import r2_score



class RegressionOutput:
    @staticmethod
    def add_significance(value):
        if value <= 0.001:
            return '***'
        elif value <= 0.01:
            return '**'
        elif value <= 0.05:
            return '*'
        else:
            return ' '
        
    @staticmethod
    def r2_adjusted(r2, n, k):
        return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

def regression_one_model(df, Xindvars, Yvar, kind='ols', summary=True):
    """Performs regression model
    Args: 
        - df (pd.DataFrame): Dataframe
        - Xindvars (str or list): X variable names
        - Yvar (str): Y variable
        - summary (boolean): Whether you want the summary
    Returns:
        - Fitted regression model
    """
    if(type(Yvar)==str):
        Yvar=[Yvar]
    if(len(Yvar)!=1):
        print("Error: please enter a single y variable")
        return np.nan
    else:
        xf = df.dropna(subset=Yvar+Xindvars)[Xindvars+Yvar]
        Xexog = xf[Xindvars]
        if kind == 'ols':
            model = sm.OLS(xf[Yvar],Xexog)
        elif kind == 'logit':
            model = sm.Logit(xf[Yvar],Xexog)
        reg = model.fit(disp=0)
    if(summary):
        return reg.summary2()
    else:
        return reg

def add_significance(value):
    if value <= 0.001:
        return '***'
    elif value <= 0.01:
        return '**'
    elif value <= 0.05:
        return '*'
    else:
        return ' '

def run_regression_group(df, indVarGroups, Y, kind='ols'):
    """
    :param df:
    :param indVarGroups:
    :param Y:
    :param kind:
    :return results:
    """
    # Get model names
    model_names = list(indVarGroups.keys())

    # Initialize empty dataframe to store results
    results = pd.DataFrame()

    # Loop through models and perform regressions
    for model in model_names:
        temp_model = regression_one_model(
            df=df, 
            Xindvars=indVarGroups[model], 
            Yvar=Y, 
            summary=False,
            kind=kind
        )

        # Get results from coefficient table
        results_temp = temp_model.summary2().tables[1]

        # Add additional metrics
        results_temp['Model Name'] = model
        results_temp['Kind'] = temp_model.summary2().tables[0].iloc[0, 1]
        results_temp['yVar'] = temp_model.summary2().tables[0].iloc[1, 1]
        results_temp['xVar'] = results_temp.index
        results_temp['nobs'] = temp_model.summary2().tables[0].iloc[3, 1]
        results_temp['r2'] = temp_model.summary2().tables[0].iloc[6, 1]
        results_temp['r2_adj'] = temp_model.summary2().tables[0].iloc[0, 3]

        # Add to dataframe
        results = pd.concat([results, results_temp], axis=0)

    # Add significance    
    if 'P>|t|' in results.columns:
        results.rename(columns={'P>|t|': 'pval', 't': 'test_statistic'}, inplace=True)
    elif 'P>|z|' in results.columns:
        results.rename(columns={'P>|z|': 'pval', 'z': 'test_statistic'}, inplace=True)

    results['significance'] = results['pval'].apply(add_significance)

    # Add lookup column/index for excel
    results.index = [f"{k}_{m}_{y}_{x}" for k, m, y, x in zip(results['Kind'], results['Model Name'], results['yVar'], results['xVar'])]

    # Clean up and return
    del temp_model
    del results_temp
    return results

def numeric_ttests(df, groupby):

    # Separate groups
    groups = df.groupby(groupby)
    n_groups = df[groupby].value_counts().shape[0]

    # Initialize lists to store items
    cols = list()
    t_stats = list()
    p_values = list()
    group_aggregation = list()

    # Loop through columns to perform tests
    for col in df.select_dtypes(['int', 'float']).columns:
        cols.append(col)

        # Get group values
        group_values = [group_data[col].dropna() for _, group_data in groups]

        # Perform t-test if n groups is 2
        if n_groups==2:
            test = stats.ttest_ind(
                *group_values,
                equal_var=False,
                nan_policy='omit'
            )
        # Perform one-way ANOVA test if n groups > 2
        elif n_groups>2:
            test = stats.f_oneway(*group_values)
        # Otherwise, raise an exception
        else:
            raise Exception("Less than two groups")
        t_stats.append(test.statistic)
        p_values.append(test.pvalue)

        # Compute  and store summary stats
        group_sum = groups[col].agg(['mean', 'count', 'std'])
        while group_sum.ndim > 1:
            group_sum = group_sum.unstack()
        group_sum.index = [
            '_'.join(str(item) for item in tup) for tup in group_sum.index
        ]
        group_aggregation.append(group_sum)

    # Compile ttest information into dataframe
    # and add significance levels
    ttests_df = pd.DataFrame(
        {
            't_stat': t_stats, 
            'p_value': p_values},
        index=cols
    )
    ttests_df['significance'] = ttests_df.p_value.apply(add_significance)

    # Compile summary stats into dataframe
    group_aggregation_df = pd.DataFrame(
        group_aggregation,
        index=cols
    )

    # Concatenate together
    ttests_df = pd.concat([group_aggregation_df, ttests_df], axis=1)

    return ttests_df