from config import (
    ALL_CATEGORICAL_COLS, GENDER_MAP, FAMILY_HISTORY_MAP, 
    OBESITY_REDUCTION_HIERARCHY, OBESITY_GAIN_HIERARCHY, CONTINUOUS_COLS
)
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import re

def initialize_lime_explainer(_X_train_encoded, all_features_list, class_names_list):
    
    training_values = _X_train_encoded[all_features_list].values
    
    categorical_feature_indices = [
        _X_train_encoded.columns.get_loc(col) 
        for col in ALL_CATEGORICAL_COLS 
        if col in _X_train_encoded.columns
    ]
    
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
    
    # Tentukan hirarki yang akan digunakan
    if current_class == 'Insufficient_Weight':
        active_hierarchy = [c for c in OBESITY_GAIN_HIERARCHY if c in class_names]
    elif current_class in OBESITY_REDUCTION_HIERARCHY:
        active_hierarchy = [c for c in OBESITY_REDUCTION_HIERARCHY if c in class_names]
    else:
        # Jika kelas saat ini adalah Normal_Weight atau tidak ada di hirarki
        return current_class, True, [current_class]

    # Temukan posisi saat ini di hirarki aktif
    try:
        current_index = active_hierarchy.index(current_class)
    except ValueError:
        return current_class, True, [current_class] # Fallback jika tidak ditemukan

    # Target selalu 'Normal_Weight'
    NORMAL_WEIGHT_CLASS = 'Normal_Weight'
    
    # Periksa apakah sudah mencapai atau melewati target
    if current_class == NORMAL_WEIGHT_CLASS:
        return current_class, True, [current_class]
    
    # Tentukan path dan target berikutnya
    if current_index < len(active_hierarchy) - 1:
        next_class = active_hierarchy[current_index + 1]
        
        # Buat path lengkap dari posisi saat ini ke Normal_Weight
        try:
            normal_index_in_hierarchy = active_hierarchy.index(NORMAL_WEIGHT_CLASS)
            all_steps = active_hierarchy[current_index : normal_index_in_hierarchy + 1]
        except ValueError:
            all_steps = [current_class, next_class]

        is_final = (next_class == NORMAL_WEIGHT_CLASS)
        
        return next_class, is_final, all_steps
    else:
        # Sudah di ujung hirarki (seharusnya Normal_Weight)
        return current_class, True, [current_class]

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

def generate_lime_explanation_text(lime_exp, predicted_class_index, predicted_class_name, user_input_raw):

    FEATURE_TRANSLATIONS = {
        'Age': 'Umur Anda',
        'Gender': 'Jenis kelamin Anda',
        'Height': 'Tinggi badan Anda',
        'Weight': 'Berat badan Anda',
        'family_history_with_overweight': 'Riwayat keluarga obesitas',
        'FAVC': 'Anda mengonsumsi makanan tinggi kalori',
        'NCP': 'Anda makan utama',
        'CAEC': 'Anda mengonsumsi cemilan',
        'SMOKE': 'Anda merokok',
        'CH2O': 'Anda meminum air',
        'SCC': 'Anda memantau kalori harian',
        'FAF': 'Anda beraktivitas fisik',
        'TUE': 'Anda menggunakan gawai',
        'CALC': 'Anda mengonsumsi alkohol',
        'MTRANS': 'Transportasi utama Anda',
        'FCVC': 'Anda mengonsumsi sayuran',
    }

    DECODERS = {
        'CH2O': {'1': '<1L', '2': '1-2L', '3': '>2L'},
        'FCVC': {'1': 'Tidak Pernah', '2': 'Kadang-kadang', '3': 'Selalu'},
        'NCP': {'1': '1x sehari', '2': '2x sehari', '3': '3x sehari', '4': '4x sehari'},
        'FAF': {'0': '< 15 menit', '1': '15 - 30 menit', '2': '30 - 60 menit', '3': '60+ menit'},
        'TUE': {'0': '< 1 jam', '1': '1 - 2 jam', '2': '>2 jam'},
        'CAEC': {'no': 'Tidak Pernah', 'Sometimes': '1-2x/minggu', 'Frequently': '3-5x/minggu', 'Always': 'Setiap Hari'},
    }

    def format_sentences_from_features(features, known_keys):
        points = []
        for feature_string, _ in features:
            sentence = None
            try:
                feature_name = None
                for key in known_keys:
                    if key in feature_string:
                        feature_name = key
                        break
                if not feature_name: continue

                raw_value = user_input_raw.get(feature_name)
                if raw_value is None: continue

                if feature_name == 'family_history_with_overweight':
                    sentence = "Tidak ada riwayat keluarga yang obesitas" if str(raw_value).lower() == 'no' else "Adanya riwayat keluarga yang obesitas"
                elif feature_name == 'Weight':
                    sentence = f"Berat badan Anda **{int(float(raw_value))} kg**"
                elif feature_name == 'Height':
                    sentence = f"Tinggi badan Anda **{int(float(raw_value) * 100)} cm**"
                elif feature_name in DECODERS:
                    translated_feature = FEATURE_TRANSLATIONS.get(feature_name, feature_name)
                    decoded_value = DECODERS[feature_name].get(str(raw_value), str(raw_value))
                    sentence = f"{translated_feature} **{decoded_value}**"
                elif feature_name in ['FAVC', 'SMOKE', 'SCC']:
                    sentence = f"Tidak {FEATURE_TRANSLATIONS.get(feature_name, feature_name).lower()}" if str(raw_value).lower() == 'no' else f"Rutin {FEATURE_TRANSLATIONS.get(feature_name, feature_name)}" #type:ignore
                else:
                    translated_feature = FEATURE_TRANSLATIONS.get(feature_name, feature_name)
                    sentence = f"{translated_feature}: **{str(raw_value).replace('_', ' ')}**"
            except Exception:
                continue
            
            if sentence:
                points.append(f"• {sentence}")
        return points

    try:
        explanation_list = lime_exp.as_list(label=predicted_class_index)
    except IndexError:
        return "Gagal menghasilkan penjelasan."

    supporting_features = sorted([f for f in explanation_list if f[1] > 0], key=lambda x: x[1], reverse=True)
    contradicting_features = sorted([f for f in explanation_list if f[1] < 0], key=lambda x: x[1])

    predicted_class_formatted = predicted_class_name.replace('_', ' ')
    final_text_parts = []
    known_feature_keys = list(FEATURE_TRANSLATIONS.keys())

    # Bagian Faktor Pendukung
    if supporting_features:
        intro_support = f"Faktor-faktor utama yang **mendukung** prediksi **{predicted_class_formatted}** adalah:"
        supporting_sentences = format_sentences_from_features(supporting_features[:3], known_feature_keys)
        if supporting_sentences:
            final_text_parts.append(intro_support + "\n\n" + "\n\n".join(supporting_sentences))

    # Bagian Faktor Bertentangan
    if contradicting_features:
        intro_contradict = f"Di sisi lain, faktor-faktor berikut sebenarnya **bertentangan** dengan prediksi ini (namun pengaruhnya lebih kecil):"
        contradicting_sentences = format_sentences_from_features(contradicting_features[:2], known_feature_keys)
        if contradicting_sentences:
            final_text_parts.append(intro_contradict + "\n\n" + "\n\n".join(contradicting_sentences))

    if not final_text_parts:
        return "Tidak dapat mengidentifikasi faktor-faktor yang signifikan untuk prediksi ini."

    return "\n\n---\n\n".join(final_text_parts)