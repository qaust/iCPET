from pathlib import Path
import pickle
from config import model_config
import pandas as pd

from sklearn.preprocessing import (
	LabelEncoder, OneHotEncoder, StandardScaler,
)

SEED = 123

if __name__=='__main__':

    df = pd.read_pickle(Path('../data/df_clean.pkl'))
    df = df.drop(columns=['study_date_mask_cl', 'study_date_mask_pe',])
    df = df.dropna(subset='total_clot_burden')
    df['resolved_pe'] = df['resolved_pe'].map({'Unresolved': 0, 'Resolved': 1})
    print(df.shape)

    cols = model_config.cat_targets + model_config.num_targets + model_config.body_feat + model_config.cardiopulmonary_feat + model_config.controls + model_config.clot_feat

    df_nonnull = df.loc[:, cols].dropna(subset=cols)
    print(df_nonnull.shape)
    num_columns = list(df_nonnull.select_dtypes(['int', 'float']).columns)
    cat_columns = list(df_nonnull.select_dtypes(['category']).columns.difference(model_config.cat_targets))
    all_columns = num_columns + cat_columns

    encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
    transformer = StandardScaler()
    label_encoder = LabelEncoder()

    y_temp = pd.Series(
        label_encoder.fit_transform(df_nonnull[model_config.cat_targets].squeeze()),
        index=df_nonnull.index,
        name='resolved_pe'
    )
    df_temp_cat = pd.DataFrame(
        encoder.fit_transform(df_nonnull[cat_columns]),
        index = df_nonnull.index,
        columns = encoder.get_feature_names_out()
    )
    df_temp_num = pd.DataFrame(
        transformer.fit_transform(df_nonnull[num_columns]),
        columns=df_nonnull[num_columns].columns,
        index=df_nonnull[num_columns].index
    )
    df_temp_all = pd.concat([df_temp_num, df_temp_cat], axis=1)

    df_pp = pd.concat([y_temp, df_temp_all], axis=1)

    targets = model_config.num_targets + model_config.cat_targets
    # Separate X
    X = df_pp.loc[:, df_pp.columns.difference(targets)]
    # Separate Y
    Y = df_pp.loc[:, targets]

    print(f"X.shape: {X.shape}")
    print(f"Y.shape: {Y.shape}")

    all_needed_columns = (
        model_config.cat_targets + 
        model_config.num_targets + 
        model_config.body_feat + 
        model_config.cardiopulmonary_feat + 
        model_config.controls_encoded + 
        model_config.clot_feat
    )

    prediction_needed_columns = (
        model_config.num_targets + 
        model_config.body_feat + 
        model_config.cardiopulmonary_feat + 
        model_config.controls_encoded
    )

    classification_needed_columns = (
        model_config.cat_targets + 
        model_config.body_feat + 
        model_config.cardiopulmonary_feat + 
        model_config.controls_encoded + 
        model_config.clot_feat
    )

    prediction_features     = model_config.body_feat + model_config.cardiopulmonary_feat + model_config.controls_encoded
    classification_features = model_config.body_feat + model_config.cardiopulmonary_feat + model_config.controls_encoded + model_config.clot_feat

    # Check columns to drop are named correctly
    assert set(all_needed_columns).issubset(set(df_pp.columns))
    assert set(prediction_needed_columns).issubset(set(df_pp.columns))
    assert set(classification_needed_columns).issubset(set(df_pp.columns))

    # Drop columns for ols
    df_prediction = df_pp.loc[:, prediction_needed_columns].dropna()

    X_prediction = df_prediction.loc[:, prediction_features]
    y_prediction = df_prediction.loc[:, model_config.num_targets]

    print(f"X.shape: {X_prediction.shape}")
    print(f"y.shape: {y_prediction.shape}")

    prediction_data = dict(
        X = X_prediction,
        y = y_prediction,
        body_features = model_config.body_feat,
        cardio_features = model_config.cardiopulmonary_feat,
        controls = model_config.controls_encoded
    )

    with open(Path('../data/prediction_data.pkl'), 'wb') as f:
        pickle.dump(prediction_data, f)
    print("Exported prediction data.")

        # Drop columns for ols
    # df_classification = df_pp.loc[:, classification_needed_columns].dropna()
    df_classification = df_pp.loc[df.pe_obs==0, classification_needed_columns].dropna()
    X_classification = df_classification.loc[:, classification_features]
    y_classification = df_classification.loc[:, model_config.cat_targets]

    print(f"X.shape: {X_classification.shape}")
    print(f"y.shape: {y_classification.shape}")

    classification_data = dict(
        X = X_classification,
        y = y_classification,
        body_features = model_config.body_feat,
        cardio_features = model_config.cardiopulmonary_feat,
        controls = model_config.controls_encoded,
        clot_features = model_config.clot_feat
    )

    with open(Path('../data/classification_data_initial.pkl'), 'wb') as f:
        pickle.dump(classification_data, f)
    print("Exported classification data.")
