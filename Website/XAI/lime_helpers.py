from config import ALL_CATEGORICAL_COLS, GENDER_MAP, FAMILY_HISTORY_MAP, OBESITY_HIERARCHY, CONTINUOUS_COLS
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

def initialize_lime_explainer(_X_train_encoded, all_features_list, class_names_list):
    
    training_values = _X_train_encoded[all_features_list].values
    
    # Identifikasi indeks fitur kategorikal
    categorical_feature_indices = [
        _X_train_encoded.columns.get_loc(col) 
        for col in ALL_CATEGORICAL_COLS 
        if col in _X_train_encoded.columns
    ]
    
    # Mapping nama kategorikal
    categorical_names_map = {}
    if 'Gender' in all_features_list:
        categorical_names_map[all_features_list.index('Gender')] = list(GENDER_MAP.keys())
    if 'family_history_with_overweight' in all_features_list:
        categorical_names_map[all_features_list.index('family_history_with_overweight')] = list(FAMILY_HISTORY_MAP.keys())
    
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

def get_next_target_class(current_class, class_names):

    # Cari kelas yang ada di hierarchy
    available_hierarchy = [c for c in OBESITY_HIERARCHY if c in class_names]
    
    # Jika kelas tidak ditemukan di hierarchy, gunakan yang tersedia
    if not available_hierarchy:
        available_hierarchy = class_names
    
    try:
        current_index = available_hierarchy.index(current_class)
    except ValueError:
        # Jika current class tidak ada di hierarchy, asumsikan di posisi terburuk
        current_index = 0
    
    # Cari Normal_Weight di hierarchy
    try:
        normal_index = available_hierarchy.index('Normal_Weight')
    except ValueError:
        # Jika Normal_Weight tidak ada, cari yang mengandung 'normal'
        normal_index = None
        for i, c in enumerate(available_hierarchy):
            if 'normal' in c.lower():
                normal_index = i
                break
        if normal_index is None:
            normal_index = len(available_hierarchy) - 1
    
    # Tentukan next target
    if current_index >= normal_index:
        # Sudah normal atau lebih baik
        return current_class, True, [current_class]
    else:
        # Bergerak 1 step menuju normal
        next_class = available_hierarchy[current_index + 1]
        is_final = (current_index + 1 >= normal_index)
        
        # Buat path lengkap
        all_steps = available_hierarchy[current_index:normal_index + 1]
        
        return next_class, is_final, all_steps

def get_step_description(current_class, next_class, step_number, total_steps):

    return f"**Step {step_number}/{total_steps}**: {current_class.replace('_', ' ')} → {next_class.replace('_', ' ')}"

def predict_proba_catboost_for_lime(data, model_obj, all_features_list):
    
    try:
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
        for col in CONTINUOUS_COLS:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0).astype(float)

        # Ensure categorical/ordinal columns are integers for LIME
        for col in ALL_CATEGORICAL_COLS:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).round().astype(int)

        probs = model_obj.predict_proba(X[all_features_list])
        return np.array(probs)
    except Exception:
        # Re-raise so LIME surface shows the original error
        raise