import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

def initialize_lime_explainer(_X_train_encoded: pd.DataFrame, all_features_list, class_names_list, all_categorical_cols):

    training_values = _X_train_encoded[all_features_list].values

    categorical_feature_indices = [
        _X_train_encoded.columns.get_loc(col)
        for col in all_categorical_cols
        if col in _X_train_encoded.columns
    ]

    # categorical_names is optional; the main app can provide a mapping if desired
    categorical_names_map = None

    lime_explainer = LimeTabularExplainer(
        training_data=training_values,
        feature_names=all_features_list,
        class_names=class_names_list,
        categorical_features=categorical_feature_indices,
        categorical_names=categorical_names_map,
        mode='classification',
        random_state=42
    )

    return lime_explainer

def predict_proba_catboost_for_lime(data, model_obj, all_features_list, continuous_cols, all_categorical_cols):
    # Normalize input to DataFrame with expected columns
    if isinstance(data, np.ndarray):
        arr = data
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        X = pd.DataFrame(arr, columns=all_features_list)
    elif isinstance(data, list):
        arr = np.array(data)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        X = pd.DataFrame(arr, columns=all_features_list)
    else:
        X = pd.DataFrame(data)

    X = X.reindex(columns=all_features_list, fill_value=0)

    # Ensure continuous types
    for col in continuous_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0).astype(float)

    # Ensure categorical/ordinal columns are integers for LIME
    for col in all_categorical_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).round().astype(int)

    probs = model_obj.predict_proba(X[all_features_list])
    return np.array(probs)
