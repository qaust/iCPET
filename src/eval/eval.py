import pandas as pd
import numpy as np


def avg_feature_importance(feat_importances: dict):
    """
    :param feat_impotances:
    :return values_array:
    :return avg_feat_importance_df:
    """
    # Get the unique keys from all dictionaries
    keys = set().union(*feat_importances)

    # Create a structured array to hold the values for each key
    dtype = [(key, float) for key in keys]
    values_array = np.zeros(len(feat_importances), dtype=dtype)

    # Fill the array with values from dictionaries
    for i, dictionary in enumerate(feat_importances):
        for key, value in dictionary.items():
            values_array[i][key] = value

    # Calculate the average values for each key
    averages = np.mean(list(values_array[key] for key in keys), axis=1)
    sds = np.std(list(values_array[key] for key in keys), axis=1)

    avg_feat_importance_df = (
        pd.DataFrame({
            'feature': list(keys),
            'avg_feature_importance': averages,
            'std_feature_importance': sds,
        })
        .sort_values(by='avg_feature_importance', ascending=False)
    )

    return values_array, avg_feat_importance_df