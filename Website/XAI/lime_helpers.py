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
    """
    Menerjemahkan hasil LIME menjadi kalimat natural bahasa Indonesia
    berdasarkan nilai input asli user.
    """
    
    FEATURE_TRANSLATIONS = {
        'Age': 'Umur Anda',
        'Gender': 'Jenis kelamin Anda',
        'Height': 'Tinggi badan Anda',
        'Weight': 'Berat badan Anda',
        'family_history_with_overweight': 'Riwayat keluarga obesitas',
        'FAVC': 'Anda mengonsumsi makanan tinggi kalori',
        'NCP': 'Jadwal makan utama',
        'CAEC': 'Kebiasaan ngemil',
        'SMOKE': 'Status merokok',
        'CH2O': 'Konsumsi air putih',
        'SCC': 'Pemantauan kalori',
        'FAF': 'Aktivitas fisik',
        'TUE': 'Penggunaan gawai',
        'CALC': 'Konsumsi alkohol',
        'MTRANS': 'Transportasi utama',
        'FCVC': 'Konsumsi sayuran',
    }

    DECODERS = {
        'CH2O': {'1': 'Kurang (<1L)', '2': 'Cukup (1-2L)', '3': 'Banyak (>2L)'},
        'FCVC': {'1': 'Jarang', '2': 'Kadang-kadang', '3': 'Sering/Selalu'},
        'NCP': {'1': '1x sehari', '2': '2x sehari', '3': '3x sehari', '4': 'Lebih dari 3x'},
        'FAF': {'0': 'Tidak ada', '1': 'Ringan (1-2 hari)', '2': 'Sedang (3-4 hari)', '3': 'Rutin/Tinggi'},
        'TUE': {'0': 'Rendah (0-2 jam)', '1': 'Sedang (3-5 jam)', '2': 'Tinggi (>5 jam)'},
        'CAEC': {'no': 'Tidak Pernah', 'Sometimes': 'Kadang-kadang', 'Frequently': 'Sering', 'Always': 'Selalu'},
        'CALC': {'no': 'Tidak Pernah', 'Sometimes': 'Kadang-kadang', 'Frequently': 'Sering', 'Always': 'Selalu'}
    }

    def format_sentences_from_features(features, known_keys):
        points = []
        # Urutkan key berdasarkan panjang string (descending) untuk menghindari salah match
        # Contoh: Agar 'family_history_with_overweight' terdeteksi sebelum 'Weight'
        sorted_keys = sorted(known_keys, key=len, reverse=True)

        for feature_string, weight in features:
            sentence = None
            try:
                feature_name = None
                for key in sorted_keys:
                    if key in feature_string:
                        feature_name = key
                        break
                
                if not feature_name: continue

                # Ambil nilai asli dari input user
                raw_value = user_input_raw.get(feature_name)
                if raw_value is None: continue

                # LOGIKA PENYUSUNAN KALIMAT
                if feature_name == 'family_history_with_overweight':
                    val_str = str(raw_value).lower()
                    sentence = "Tidak memiliki riwayat obesitas di keluarga" if val_str == 'no' else "Memiliki riwayat obesitas di keluarga"
                
                elif feature_name == 'Weight':
                    sentence = f"Berat badan saat ini **{int(float(raw_value))} kg**"
                
                elif feature_name == 'Height':
                    # Asumsi input raw height dalam meter (karena pre-processing handle convert) 
                    # TAPI user_input_raw biasanya menyimpan apa yg diinput di form (cm atau m).
                    # Kita cek range nilainya.
                    h_val = float(raw_value)
                    if h_val < 3.0: h_val *= 100 # Convert m to cm for display if needed
                    sentence = f"Tinggi badan **{int(h_val)} cm**"
                
                elif feature_name in DECODERS:
                    label = FEATURE_TRANSLATIONS.get(feature_name, feature_name)
                    decoded_val = DECODERS[feature_name].get(str(raw_value), str(raw_value))
                    sentence = f"{label} tergolong **{decoded_val}**"
                
                elif feature_name in ['FAVC', 'SMOKE', 'SCC']:
                    label = FEATURE_TRANSLATIONS.get(feature_name, feature_name)
                    is_yes = str(raw_value).lower() == 'yes'
                    
                    if feature_name == 'SMOKE':
                        sentence = "Anda **Merokok**" if is_yes else "Anda **Tidak Merokok**"
                    elif feature_name == 'SCC':
                        sentence = "Anda **Memantau kalori**" if is_yes else "Anda **Tidak memantau kalori**"
                    elif feature_name == 'FAVC':
                        sentence = "Sering konsumsi makanan tinggi kalori" if is_yes else "Jarang konsumsi makanan tinggi kalori"
                
                else:
                    # Fallback
                    label = FEATURE_TRANSLATIONS.get(feature_name, feature_name)
                    sentence = f"{label}: **{str(raw_value).replace('_', ' ')}**"

            except Exception:
                continue
            
            if sentence:
                # Tambahkan icon +/- visual
                points.append(f"• {sentence}")
        
        return points

    try:
        explanation_list = lime_exp.as_list(label=predicted_class_index)
    except IndexError:
        return "Gagal menghasilkan penjelasan."

    # Pisahkan fitur mendukung (+) dan bertentangan (-)
    supporting_features = sorted([f for f in explanation_list if f[1] > 0], key=lambda x: x[1], reverse=True)
    contradicting_features = sorted([f for f in explanation_list if f[1] < 0], key=lambda x: x[1])

    predicted_class_formatted = predicted_class_name.replace('_', ' ')
    final_text_parts = []
    known_feature_keys = list(FEATURE_TRANSLATIONS.keys())

    # Bagian 1: Faktor Pendukung (Top 3)
    if supporting_features:
        intro_support = f"##### Faktor Pendukung\nFaktor berikut **meningkatkan** kecenderungan Anda masuk kategori **{predicted_class_formatted}**:"
        supporting_sentences = format_sentences_from_features(supporting_features[:5], known_feature_keys)
        if supporting_sentences:
            final_text_parts.append(intro_support + "\n" + "\n".join(supporting_sentences))

    # Bagian 2: Faktor Bertentangan (Top 2)
    if contradicting_features:
        intro_contradict = f"##### Faktor Pegenndali\nNamun, faktor berikut membantu **menahan** kenaikan tingkat obesitas Anda (menjaga agar tidak lebih parah):"
        contradicting_sentences = format_sentences_from_features(contradicting_features[:3], known_feature_keys)
        if contradicting_sentences:
            final_text_parts.append(intro_contradict + "\n" + "\n".join(contradicting_sentences))

    if not final_text_parts:
        return "Data tidak cukup signifikan untuk menarik kesimpulan spesifik."

    return "\n\n---\n\n".join(final_text_parts)