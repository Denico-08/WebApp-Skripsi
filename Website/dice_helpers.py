import numpy as np
import pandas as pd
import dice_ml
from dice_ml import Dice
from typing import Any

# Provide a lightweight DiceCatBoostWrapper and helper functions that don't depend on Streamlit UI.
class DiceCatBoostWrapper:
    """Wrapper for CatBoost model to be used by DiCE."""
    def __init__(self, model, feature_names, continuous_features, categorical_features, class_labels=None):
        self.model = model
        self.feature_names = feature_names
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.class_labels = class_labels
        if class_labels is not None:
            self.classes_ = list(class_labels)

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        X_copy = X.copy()
        for col in self.continuous_features:
            if col in X_copy.columns:
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(X_copy[col].median() or 0)
        for col in self.categorical_features:
            if col in X_copy.columns:
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(0).round().astype(int)
        X_copy = X_copy[self.feature_names]
        probs = self.model.predict_proba(X_copy)
        # reorder if needed
        try:
            model_cls = list(getattr(self.model, 'classes_', []))
            if hasattr(self, 'class_labels') and self.class_labels is not None and model_cls:
                if model_cls != list(self.class_labels):
                    idx = [model_cls.index(c) for c in self.class_labels]
                    probs = np.array(probs)[:, idx]
        except Exception:
            pass
        return probs

def prepare_dice_data(x_train_encoded: pd.DataFrame, model_obj: Any, encoders: dict, all_features_list: list, all_categorical_cols: list, target_name: str):
    """Prepare a DataFrame for DiCE with a correctly-typed target column."""
    df_dice = x_train_encoded.copy()
    for col in all_categorical_cols:
        if col in df_dice.columns:
            df_dice[col] = pd.to_numeric(df_dice[col], errors='coerce').fillna(0).round().astype(int)
    try:
        probs_all = model_obj.predict_proba(df_dice[all_features_list])
        preds_idx = np.argmax(probs_all, axis=1)
        class_labels = np.array(encoders[target_name].classes_)
        derived_labels = class_labels[preds_idx]
        df_dice[target_name] = pd.Categorical(derived_labels, categories=list(encoders[target_name].classes_))
    except Exception:
        try:
            string_predictions = model_obj.predict(df_dice[all_features_list]).ravel()
            df_dice[target_name] = string_predictions
        except Exception:
            df_dice[target_name] = np.full((len(df_dice),), encoders[target_name].classes_[0])
    return df_dice

def get_dice_recommendations(x_train_encoded: pd.DataFrame, model_obj: Any, encoders: dict, input_df_processed: pd.DataFrame, desired_class_index: int, all_features_list: list, continuous_cols: list, all_categorical_cols: list, target_name: str, allow_weight_change: bool=False):
    """Simplified DiCE runner that returns the dice result or None.

    This function avoids Streamlit UI calls and progress bars; the main app is responsible
    for displaying progress and messages.
    """
    df_dice = prepare_dice_data(x_train_encoded, model_obj, encoders, all_features_list, all_categorical_cols, target_name)

    dice_continuous_features = continuous_cols
    data_interface = dice_ml.Data(dataframe=df_dice, continuous_features=dice_continuous_features, outcome_name=target_name)
    wrapped_model = DiceCatBoostWrapper(model_obj, all_features_list, continuous_cols, all_categorical_cols, class_labels=encoders[target_name].classes_)
    model_interface = dice_ml.Model(model=wrapped_model, backend="sklearn", model_type='classifier')

    query_instance = input_df_processed[all_features_list].copy()
    for col in all_categorical_cols:
        if col in query_instance.columns:
            query_instance[col] = query_instance[col].astype(int)

    desired_class_label = encoders[target_name].classes_[desired_class_index]

    strategies = [
        {'name': 'user_preference', 'allow_weight': allow_weight_change, 'total_cfs': 15, 'methods': ['genetic', 'random'], 'proximity_weight': 0.5, 'diversity_weight': 1.0},
        {'name': 'with_weight_relaxed', 'allow_weight': True, 'total_cfs': 20, 'methods': ['genetic', 'random'], 'proximity_weight': 0.2, 'diversity_weight': 2.0},
        {'name': 'very_relaxed', 'allow_weight': True, 'total_cfs': 30, 'methods': ['random', 'genetic'], 'proximity_weight': 0.1, 'diversity_weight': 3.0}
    ]

    for strategy in strategies:
        base_immutables = ['Gender', 'Age', 'Height', 'family_history_with_overweight']
        immutable_features = base_immutables if strategy['allow_weight'] else base_immutables + ['Weight']
        features_to_vary = [col for col in all_features_list if col not in immutable_features]

        current_weight = float(query_instance['Weight'].iloc[0])
        if strategy['name'] == 'very_relaxed':
            weight_min = max(30.0, round(current_weight - 100.0, 1))
        elif strategy['name'] == 'with_weight_relaxed':
            weight_min = max(30.0, round(current_weight - 70.0, 1))
        else:
            weight_min = max(30.0, round(current_weight - 50.0, 1))
        weight_max = current_weight

        permitted_range = {**({'Weight': [weight_min, weight_max]} if strategy['allow_weight'] else {}), 'FCVC': [1, 3], 'NCP': [1, 4], 'CH2O': [1, 3], 'FAF': [0, 3], 'TUE': [0, 2]}

        for method in strategy['methods']:
            try:
                dice_explainer = Dice(data_interface, model_interface, method=method)
                dice_result = dice_explainer.generate_counterfactuals(
                    query_instance,
                    total_CFs=strategy['total_cfs'],
                    desired_class=desired_class_label,
                    features_to_vary=features_to_vary, # type: ignore
                    permitted_range=permitted_range,
                    proximity_weight=strategy['proximity_weight'],
                    diversity_weight=strategy['diversity_weight']
                )

                if dice_result and dice_result.cf_examples_list and dice_result.cf_examples_list[0].final_cfs_df is not None and len(dice_result.cf_examples_list[0].final_cfs_df) > 0:
                    return dice_result

            except Exception:
                # Try numeric-encoded fallback
                try:
                    df_dice_numeric = df_dice.copy()
                    df_dice_numeric[target_name] = encoders[target_name].transform(df_dice_numeric[target_name].astype(str))
                    data_interface_num = dice_ml.Data(dataframe=df_dice_numeric, continuous_features=dice_continuous_features, outcome_name=target_name)
                    numeric_class_labels = list(range(len(encoders[target_name].classes_)))
                    wrapped_model_num = DiceCatBoostWrapper(model_obj, all_features_list, continuous_cols, all_categorical_cols, class_labels=numeric_class_labels)
                    model_interface_num = dice_ml.Model(model=wrapped_model_num, backend="sklearn", model_type='classifier')
                    dice_explainer_num = Dice(data_interface_num, model_interface_num, method=method)
                    desired_index = int(encoders[target_name].transform([str(desired_class_label)])[0])
                    dice_result = dice_explainer_num.generate_counterfactuals(
                        query_instance,
                        total_CFs=strategy['total_cfs'],
                        desired_class=desired_index, # type: ignore
                        features_to_vary=features_to_vary, # type: ignore
                        permitted_range=permitted_range,
                        proximity_weight=strategy['proximity_weight'],
                        diversity_weight=strategy['diversity_weight']
                    )
                    if dice_result and dice_result.cf_examples_list and dice_result.cf_examples_list[0].final_cfs_df is not None and len(dice_result.cf_examples_list[0].final_cfs_df) > 0:
                        return dice_result
                except Exception:
                    continue
    return None

def decode_dice_dataframe(df_dice_output: pd.DataFrame, encoders: dict, all_features_list: list, decode_maps: dict, continuous_cols: list, ordinal_cols: list, target_name: str):
    df_decoded = df_dice_output.copy()
    for col_name, mapping in decode_maps.items():
        if col_name in df_decoded.columns:
            df_decoded[col_name] = df_decoded[col_name].astype(str).map({str(k): v for k, v in mapping.items()})
    for col in continuous_cols:
        if col in df_decoded.columns:
            df_decoded[col] = pd.to_numeric(df_decoded[col], errors='coerce').round(2)
    for col in ordinal_cols:
        if col in df_decoded.columns:
            df_decoded[col] = pd.to_numeric(df_decoded[col], errors='coerce').round().astype(int)
    if target_name in df_decoded.columns:
        try:
            numeric_vals = pd.to_numeric(df_decoded[target_name], errors='coerce')
            if not numeric_vals.isna().all():
                df_decoded[target_name] = encoders[target_name].inverse_transform(numeric_vals.round().astype(int))
            else:
                df_decoded[target_name] = df_decoded[target_name].astype(str)
        except Exception:
            df_decoded[target_name] = df_decoded[target_name].astype(str)
    return df_decoded[all_features_list + [target_name]]
