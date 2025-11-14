import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os
from matplotlib import pyplot as plt
from catboost import CatBoostClassifier
from lime.lime_tabular import LimeTabularExplainer
import dice_ml
from dice_ml import Dice
from sklearn.preprocessing import LabelEncoder
import traceback
from Login import logout
from Website.Connection.supabase_client import insert_input_to_supabase

# ================================================================================
# KONSTANTA DAN KONFIGURASI
# ================================================================================

TARGET_NAME = 'NObeyesdad'

CONTINUOUS_COLS = ['Age', 'Height', 'Weight']
CATEGORICAL_COLS = [
    'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',
    'family_history_with_overweight', 'CAEC', 'MTRANS'
]
ORDINAL_COLS = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
ALL_CATEGORICAL_COLS = CATEGORICAL_COLS + ORDINAL_COLS

# Hierarchy obesitas dari terburuk ke terbaik
OBESITY_HIERARCHY = [
    'Obesity_Type_III',
    'Obesity_Type_II', 
    'Obesity_Type_I',
    'Overweight',
    'Normal_Weight',
    'Insufficient_Weight'
]

# Mapping untuk encoding
GENDER_MAP = {'Female': 0, 'Male': 1}
FAMILY_HISTORY_MAP = {'no': 0, 'yes': 1}
FAVC_MAP = {'no': 0, 'yes': 1}
SCC_MAP = {'no': 0, 'yes': 1}
SMOKE_MAP = {'no': 0, 'yes': 1}
CAEC_MAP = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
CALC_MAP = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
MTRANS_MAP = {
    'Walking': 0,
    'Public_Transportation': 1,
    'Bike': 2,
    'Motorbike': 3,
    'Automobile': 4
}

# Mapping untuk decode
DECODE_MAPS = {
    'Gender': {0: 'Female', 1: 'Male'},
    'family_history_with_overweight': {0: 'no', 1: 'yes'},
    'FAVC': {0: 'no', 1: 'yes'},
    'SCC': {0: 'no', 1: 'yes'},
    'SMOKE': {0: 'no', 1: 'yes'},
    'CAEC': {0: 'no', 1: 'Sometimes', 2: 'Frequently', 3: 'Always'},
    'CALC': {0: 'no', 1: 'Sometimes', 2: 'Frequently', 3: 'Always'},
    'MTRANS': {0: 'Walking', 1: 'Public_Transportation', 2: 'Bike', 3: 'Motorbike', 4: 'Automobile'}
}

# ================================================================================
# FUNGSI HELPER
# ================================================================================

def get_model_paths():
    """Mendapatkan path ke file model dan aset lainnya."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "Model_Website")
    
    return {
        'model': os.path.join(model_dir, "catboost_obesity_model.cbm"),
        'target_encoder': os.path.join(model_dir, "label_encoder_y_v3.pkl"),
        'feature_names': os.path.join(model_dir, "feature_names_v3.pkl"),
        'x_train': os.path.join(model_dir, "X_train_processed.pkl"),
        'class_names': os.path.join(model_dir, "class_names_y_v3.pkl")
    }

def preprocess_input_data(input_dict, all_features_list):

    df = pd.DataFrame([input_dict])
    
    if df.empty:
        return None
    
    try:
        # Proses fitur kontinyu
        for col in CONTINUOUS_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle missing values
            if df[col].isnull().any() or np.isinf(df[col]).any():
                defaults = {'Age': 25, 'Height': 170, 'Weight': 70}
                df[col] = df[col].fillna(defaults.get(col, 0))
            
            # Konversi tinggi dari meter ke cm jika perlu
            if col == 'Height' and df[col].iloc[0] < 10:
                df[col] = df[col] * 100
            
            # Casting tipe data
            if col == 'Age' or col == 'Height':
                df[col] = df[col].round().astype(int)
            else:
                df[col] = df[col].round(3).astype(float)
        
        # Encode fitur kategorikal
        df['Gender'] = df['Gender'].map(GENDER_MAP).astype(int)
        df['family_history_with_overweight'] = df['family_history_with_overweight'].map(FAMILY_HISTORY_MAP).astype(int)
        df['FAVC'] = df['FAVC'].map(FAVC_MAP).astype(int)
        df['SCC'] = df['SCC'].map(SCC_MAP).astype(int)
        df['SMOKE'] = df['SMOKE'].map(SMOKE_MAP).astype(int)
        df['CAEC'] = df['CAEC'].map(CAEC_MAP).astype(int)
        df['CALC'] = df['CALC'].map(CALC_MAP).astype(int)
        df['MTRANS'] = df['MTRANS'].map(MTRANS_MAP).astype(int)
        
        # Proses fitur ordinal
        ordinal_defaults = {'FCVC': 2, 'NCP': 3, 'CH2O': 2, 'FAF': 1, 'TUE': 1}
        for col in ORDINAL_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(ordinal_defaults[col]).round().astype(int)
        
        # Reorder kolom sesuai feature list
        df = df[all_features_list]
        
        return df
        
    except KeyError as e:
        st.error(f"Error: Kolom yang hilang saat preprocessing: {e}")
        return None
    except Exception as e:
        st.error(f"Error saat preprocessing data input: {e}")
        return None

@st.cache_resource
def load_all_assets():

    paths = get_model_paths()
    
    try:
        # Load model
        model = CatBoostClassifier()
        model.load_model(paths['model'])
        
        # Load encoders dan metadata
        target_encoder = joblib.load(paths['target_encoder'])
        all_features = joblib.load(paths['feature_names'])
        class_names = joblib.load(paths['class_names'])
        encoders = {TARGET_NAME: target_encoder}
        
        # Load training data
        x_train_array = joblib.load(paths['x_train'])
        
        # Konversi ke DataFrame jika perlu
        if isinstance(x_train_array, np.ndarray):
            if all_features:
                x_train_encoded = pd.DataFrame(x_train_array, columns=all_features)
            else:
                st.error("Gagal konversi: 'all_features' tidak valid.")
                return None, None, None, None, None
        elif isinstance(x_train_array, pd.DataFrame):
            x_train_encoded = x_train_array
        else:
            st.error(f"Tipe X_train tidak didukung: {type(x_train_array)}")
            return None, None, None, None, None
        
        # Validasi hasil
        if not isinstance(x_train_encoded, pd.DataFrame):
            st.error("X_train tidak berhasil dikonversi ke DataFrame.")
            return None, None, None, None, None
        
        return model, encoders, all_features, class_names, x_train_encoded
        
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error saat memuat aset: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None, None, None

@st.cache_resource
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
# ================================================================================
# WRAPPER DAN FUNGSI DICE
# ================================================================================

class DiceCatBoostWrapper:
    """Wrapper untuk model CatBoost agar kompatibel dengan DiCE."""
    
    def __init__(self, model, feature_names, continuous_features, categorical_features, class_labels=None):
        self.model = model
        self.feature_names = feature_names
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.class_labels = class_labels
        
        if class_labels is not None:
            self.classes_ = list(class_labels)
    
    def predict_proba(self, X):
        """Prediksi probabilitas dengan preprocessing otomatis."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        X_copy = X.copy()
        
        # Preprocessing continuous features
        for col in self.continuous_features:
            if col in X_copy.columns:
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(X_copy[col].median() or 0)
        
        # Preprocessing categorical features
        for col in self.categorical_features:
            if col in X_copy.columns:
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(0).round().astype(int)
        
        X_copy = X_copy[self.feature_names]
        probs = self.model.predict_proba(X_copy)
        
        # Reorder probabilitas jika perlu
        try:
            model_cls = list(getattr(self.model, 'classes_', []))
            if hasattr(self, 'class_labels') and self.class_labels is not None and model_cls:
                if model_cls != list(self.class_labels):
                    idx = [model_cls.index(c) for c in self.class_labels]
                    probs = np.array(probs)[:, idx]
        except Exception:
            pass
        
        return probs

def prepare_dice_data(x_train_encoded, model_obj, encoders, all_features_list):
    df_dice = x_train_encoded.copy()
    
    # Pastikan tipe data kategorikal benar
    for col in ALL_CATEGORICAL_COLS:
        if col in df_dice.columns:
            df_dice[col] = pd.to_numeric(df_dice[col], errors='coerce').fillna(0).round().astype(int)
    
    # Generate labels
    try:
        probs_all = model_obj.predict_proba(df_dice[all_features_list])
        preds_idx = np.argmax(probs_all, axis=1)
        class_labels = np.array(encoders[TARGET_NAME].classes_)
        derived_labels = class_labels[preds_idx]
        df_dice[TARGET_NAME] = pd.Categorical(derived_labels, categories=list(encoders[TARGET_NAME].classes_))
    except Exception:
        try:
            string_predictions = model_obj.predict(df_dice[all_features_list]).ravel()
            df_dice[TARGET_NAME] = string_predictions
        except Exception:
            df_dice[TARGET_NAME] = np.full((len(df_dice),), encoders[TARGET_NAME].classes_[0])
    
    return df_dice

def get_dice_recommendations(x_train_encoded, model_obj, encoders, input_df_processed, 
                            desired_class_index, all_features_list, allow_weight_change=False):

    progress_bar = st.progress(0, text="Mencari rekomendasi perubahan...")
    
    try:
        # Persiapkan data
        progress_bar.progress(10, text="Mempersiapkan data training...")
        df_dice = prepare_dice_data(x_train_encoded, model_obj, encoders, all_features_list)
        
        # DEBUG: Cek distribusi kelas
        st.write(f"🔍 DEBUG - Distribusi kelas di data training:")
        class_dist = df_dice[TARGET_NAME].value_counts()
        st.write(class_dist)
        
        # Konfigurasi DiCE
        progress_bar.progress(30, text="Mengatur batasan nilai fitur...")
        dice_continuous_features = CONTINUOUS_COLS + ORDINAL_COLS
        
        data_interface = dice_ml.Data(
            dataframe=df_dice,
            continuous_features=dice_continuous_features,
            outcome_name=TARGET_NAME
        )
        
        wrapped_model = DiceCatBoostWrapper(
            model_obj,
            all_features_list,
            CONTINUOUS_COLS,
            ALL_CATEGORICAL_COLS,
            class_labels=encoders[TARGET_NAME].classes_
        )
        
        model_interface = dice_ml.Model(
            model=wrapped_model,
            backend="sklearn",
            model_type='classifier'
        )
        
        progress_bar.progress(50, text="Menyiapkan algoritma pencarian...")
        
        # Persiapkan query instance
        query_instance = input_df_processed[all_features_list].copy()
        for col in ALL_CATEGORICAL_COLS:
            if col in query_instance.columns:
                query_instance[col] = query_instance[col].astype(int)
        
        # DEBUG: Cek prediksi query
        current_pred_proba = model_obj.predict_proba(query_instance)
        current_pred_idx = np.argmax(current_pred_proba[0])        
        # Dapatkan desired class label
        desired_class_label = encoders[TARGET_NAME].classes_[desired_class_index]
        
        # STRATEGI PENCARIAN BERTINGKAT
        strategies = [
            {
                'name': 'user_preference',
                'allow_weight': allow_weight_change,
                'total_cfs': 15,
                'methods': ['genetic', 'random'],
                'proximity_weight': 0.5,
                'diversity_weight': 1.0
            },
            {
                'name': 'with_weight_relaxed',
                'allow_weight': True,
                'total_cfs': 20,
                'methods': ['genetic', 'random'],
                'proximity_weight': 0.2,  # Lebih fleksibel
                'diversity_weight': 2.0
            },
            {
                'name': 'very_relaxed',
                'allow_weight': True,
                'total_cfs': 30,
                'methods': ['random', 'genetic'],  # Random dulu karena lebih cepat
                'proximity_weight': 0.1,  # Sangat fleksibel
                'diversity_weight': 3.0
            }
        ]
        
        for strategy_idx, strategy in enumerate(strategies):
            progress_bar.progress(
                60 + (strategy_idx * 10), 
                text=f"Mencoba strategi {strategy_idx + 1}/{len(strategies)}: {strategy['name']}..."
            )
            
            st.write(f"🔄 Mencoba strategi: **{strategy['name']}** (allow_weight={strategy['allow_weight']}, total_CFs={strategy['total_cfs']})")
            
            # Tentukan fitur yang tidak boleh diubah
            base_immutables = ['Gender', 'Age', 'Height', 'family_history_with_overweight']
            immutable_features = base_immutables if strategy['allow_weight'] else base_immutables + ['Weight']
            features_to_vary = [col for col in all_features_list if col not in immutable_features]
            
            st.write(f"   - Fitur yang boleh diubah: {', '.join(features_to_vary)}")
            
            # Batasan nilai - semakin fleksibel untuk strategi lanjutan
            current_weight = float(query_instance['Weight'].iloc[0])
            if strategy['name'] == 'very_relaxed':
                weight_min = max(30.0, round(current_weight - 100.0, 1))  # Range sangat besar
            elif strategy['name'] == 'with_weight_relaxed':
                weight_min = max(30.0, round(current_weight - 70.0, 1))
            else:
                weight_min = max(30.0, round(current_weight - 50.0, 1))
            weight_max = current_weight
            
            permitted_range = {
                **({'Weight': [weight_min, weight_max]} if strategy['allow_weight'] else {}),
                'FCVC': [1, 3],
                'NCP': [1, 4],
                'CH2O': [1, 3],
                'FAF': [0, 3],
                'TUE': [0, 2]
            }
            
            if strategy['allow_weight']:
                st.write(f"   - Range berat badan: {weight_min:.1f}kg - {weight_max:.1f}kg")
            
            # Coba berbagai metode dalam strategi ini
            for method_idx, method in enumerate(strategy['methods']):
                try:
                    st.write(f"   - Mencoba metode: **{method}**...")
                    
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
                    
                    # Validasi hasil
                    if (dice_result and 
                        dice_result.cf_examples_list and 
                        dice_result.cf_examples_list[0].final_cfs_df is not None and 
                        len(dice_result.cf_examples_list[0].final_cfs_df) > 0):
                        
                        num_cfs = len(dice_result.cf_examples_list[0].final_cfs_df)
                        st.success(f"✅ Berhasil! Ditemukan {num_cfs} counterfactual(s) dengan strategi '{strategy['name']}' metode '{method}'")
                        
                        progress_bar.progress(100, text="Selesai!")
                        
                        # Beri tahu user jika menggunakan strategi fallback
                        if strategy['name'] != 'user_preference':
                            if strategy['allow_weight'] and not allow_weight_change:
                                st.info("ℹ️ Rekomendasi ditemukan dengan mengizinkan perubahan berat badan.")
                            if strategy['name'] == 'very_relaxed':
                                st.info("ℹ️ Rekomendasi ditemukan dengan kriteria yang sangat fleksibel.")
                        
                        return dice_result
                    else:
                        st.write(f"      ❌ Metode {method} tidak menghasilkan counterfactual")
                        
                except Exception as e:
                    error_msg = str(e)
                    st.write(f"      ⚠️ Error pada metode {method}: {error_msg[:100]}")
                    
                    # Jika error terkait target class, coba dengan numeric encoding
                    if 'could not be identified' in error_msg.lower() or 'target' in error_msg.lower():
                        try:
                            st.write(f"      🔄 Mencoba dengan numeric encoding...")
                            
                            # Buat data dengan numeric target
                            df_dice_numeric = df_dice.copy()
                            df_dice_numeric[TARGET_NAME] = encoders[TARGET_NAME].transform(df_dice_numeric[TARGET_NAME].astype(str))
                            
                            data_interface_num = dice_ml.Data(
                                dataframe=df_dice_numeric,
                                continuous_features=dice_continuous_features,
                                outcome_name=TARGET_NAME
                            )
                            
                            numeric_class_labels = list(range(len(encoders[TARGET_NAME].classes_)))
                            wrapped_model_num = DiceCatBoostWrapper(
                                model_obj,
                                all_features_list,
                                CONTINUOUS_COLS,
                                ALL_CATEGORICAL_COLS,
                                class_labels=numeric_class_labels
                            )
                            
                            model_interface_num = dice_ml.Model(
                                model=wrapped_model_num,
                                backend="sklearn",
                                model_type='classifier'
                            )
                            
                            dice_explainer_num = Dice(data_interface_num, model_interface_num, method=method)
                            
                            dice_result = dice_explainer_num.generate_counterfactuals(
                                query_instance,
                                total_CFs=strategy['total_cfs'],
                                desired_class=desired_class_index,
                                features_to_vary=features_to_vary, # type: ignore
                                permitted_range=permitted_range,
                                proximity_weight=strategy['proximity_weight'],
                                diversity_weight=strategy['diversity_weight']
                            )
                            
                            if (dice_result and 
                                dice_result.cf_examples_list and 
                                dice_result.cf_examples_list[0].final_cfs_df is not None and 
                                len(dice_result.cf_examples_list[0].final_cfs_df) > 0):
                                
                                num_cfs = len(dice_result.cf_examples_list[0].final_cfs_df)
                                st.success(f"✅ Berhasil dengan numeric encoding! Ditemukan {num_cfs} counterfactual(s)")
                                progress_bar.progress(100, text="Selesai!")
                                return dice_result
                            
                        except Exception as e2:
                            st.write(f"      ❌ Numeric encoding juga gagal: {str(e2)[:100]}")
                            continue
                    
                    continue
        
        # Jika semua strategi gagal
        progress_bar.progress(100, text="Selesai!")
        st.error("❌ Semua strategi pencarian gagal menemukan counterfactual yang valid.")
        return None
        
    except Exception as e:
        st.error(f"Error saat mencari rekomendasi DiCE: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
        progress_bar.empty()

def decode_dice_dataframe(df_dice_output, encoders, all_features_list):

    df_decoded = df_dice_output.copy()
    
    # Decode fitur kategorikal
    for col_name, mapping in DECODE_MAPS.items():
        if col_name in df_decoded.columns:
            df_decoded[col_name] = df_decoded[col_name].astype(str).map({str(k): v for k, v in mapping.items()})
    
    # Format fitur kontinyu
    for col in CONTINUOUS_COLS:
        if col in df_decoded.columns:
            df_decoded[col] = pd.to_numeric(df_decoded[col], errors='coerce').round(2)
    
    # Format fitur ordinal
    for col in ORDINAL_COLS:
        if col in df_decoded.columns:
            df_decoded[col] = pd.to_numeric(df_decoded[col], errors='coerce').round().astype(int)
    
    # Decode target
    if TARGET_NAME in df_decoded.columns:
        try:
            numeric_vals = pd.to_numeric(df_decoded[TARGET_NAME], errors='coerce')
            if not numeric_vals.isna().all():
                df_decoded[TARGET_NAME] = encoders[TARGET_NAME].inverse_transform(numeric_vals.round().astype(int))
            else:
                df_decoded[TARGET_NAME] = df_decoded[TARGET_NAME].astype(str)
        except Exception:
            df_decoded[TARGET_NAME] = df_decoded[TARGET_NAME].astype(str)
    
    return df_decoded[all_features_list + [TARGET_NAME]]

# ================================================================================
# APLIKASI STREAMLIT UTAMA
# ================================================================================

def run_prediction_app():
    """Fungsi utama aplikasi Streamlit."""
    
    st.set_page_config(page_title="Prediksi Obesitas (XAI)", layout="wide")
    
    # Sidebar - User Info
    with st.sidebar:
        st.title(f"👋 Halo, {st.session_state.get('user_name', 'Pengguna')}!")
        st.write(f"**Email:** {st.session_state.get('user')}")
        st.write(f"**Role:** {st.session_state.get('user_role')}")
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.session_state.page = "login"
            st.rerun()
        
        st.markdown("---")
        st.title("Informasi")
        st.write("Aplikasi ini memprediksi tingkat obesitas berdasarkan kebiasaan sehari-hari.")
        
        st.markdown("---")
        if st.button("Hapus Cache & Muat Ulang"):
            st.cache_resource.clear()
            st.rerun()
    
    # Load semua aset
    loaded_assets = load_all_assets()
    model, encoders, ALL_FEATURES, CLASS_NAMES, x_train_encoded = loaded_assets
    
    if any(asset is None for asset in loaded_assets):
        st.error("Gagal memuat aset penting. Aplikasi berhenti.")
        st.stop()
    
    # Cari kelas Normal Weight
    NORMAL_WEIGHT_CLASS = 'Normal_Weight'
    if CLASS_NAMES:
        for c in CLASS_NAMES:
            if "normal" in c.lower():
                NORMAL_WEIGHT_CLASS = c
                break
    
    # Inisialisasi LIME explainer
    lime_explainer = initialize_lime_explainer(x_train_encoded, ALL_FEATURES, CLASS_NAMES)
    
    # Inisialisasi session state
    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False
        st.session_state.user_input_raw = None
        st.session_state.show_lime = False
        st.session_state.show_dice = False
    
    # Header
    st.title("🏥 Prediksi Tingkat Obesitas & Rekomendasi Perubahan")
    st.markdown("Masukkan data Anda di bawah ini untuk mendapatkan prediksi dan penjelasan (XAI).")
    
    # Form Input
    st.header("📋 Masukkan Data Anda")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Data Pribadi")
        gender = st.selectbox('Gender', list(GENDER_MAP.keys()), key='gender')
        age = st.number_input('Umur', 1, 100, 22, 1, key='age')
        height_cm = st.number_input('Tinggi Badan (cm)', 100, 250, 168, key='height_cm')
        weight = st.number_input('Berat Badan (kg)', 30, 200, 63, key='weight')
        family_history = st.selectbox(
            'Riwayat keluarga obesitas?',
            list(FAMILY_HISTORY_MAP.keys()),
            format_func=lambda x: 'Tidak' if x == 'no' else 'Ya',
            key='family'
        )
    
    with col2:
        st.subheader("Kebiasaan Makan")
        favc = st.selectbox(
            'Makan makanan tinggi kalori?',
            list(FAVC_MAP.keys()),
            index=1,
            format_func=lambda x: 'Tidak' if x == 'no' else 'Ya',
            key='favc'
        )
        fcvc = st.selectbox(
            'Frekuensi makan sayur',
            [1, 2, 3],
            index=2,
            format_func=lambda x: {1: 'Tidak Pernah', 2: 'Kadang-kadang', 3: 'Selalu'}[x],
            key='fcvc'
        )
        ncp = st.number_input('Makan utama per hari', 1, 4, 3, 1, key='ncp')
        caec_choice = st.selectbox('Makan cemilan?', list(CAEC_MAP.keys()), index=1, key='caec')
        calc_choice = st.selectbox('Minum alkohol?', list(CALC_MAP.keys()), index=0, key='calc')
    
    with col3:
        st.subheader("Gaya Hidup")
        smoke = st.selectbox(
            'Merokok?',
            list(SMOKE_MAP.keys()),
            format_func=lambda x: 'Tidak' if x == 'no' else 'Ya',
            key='smoke'
        )
        ch2o = st.selectbox(
            'Minum air per hari',
            [1, 2, 3],
            index=1,
            format_func=lambda x: {1: '<1L', 2: '1-2L', 3: '>2L'}[x],
            key='ch2o'
        )
        scc = st.selectbox(
            'Memantau kalori?',
            list(SCC_MAP.keys()),
            format_func=lambda x: 'Tidak' if x == 'no' else 'Ya',
            key='scc'
        )
        faf = st.selectbox('Aktivitas fisik (hari/minggu)', [0, 1, 2, 3], index=1, key='faf')
        tue = st.selectbox('Penggunaan gawai (jam/hari)', [0, 1, 2], index=1, key='tue')
        mtrans = st.selectbox('Transportasi utama', list(MTRANS_MAP.keys()), index=1, key='mtrans')
    
    # Tombol Prediksi
    st.markdown("---")
    if st.button("🔮 Dapatkan Prediksi dan Penjelasan", type="primary", use_container_width=True):
        try:
            height = float(height_cm) / 100.0
            st.session_state.user_input_raw = {
                'Age': int(age),
                'Gender': str(gender),
                'Height': height,
                'Weight': float(weight),
                'CALC': str(calc_choice),
                'FAVC': str(favc),
                'FCVC': str(fcvc),
                'NCP': str(ncp),
                'SCC': str(scc),
                'SMOKE': str(smoke),
                'CH2O': str(ch2o),
                'family_history_with_overweight': str(family_history),
                'FAF': str(faf),
                'TUE': str(tue),
                'CAEC': str(caec_choice),
                'MTRANS': str(mtrans)
            }
            st.session_state.prediction_done = True
            st.session_state.show_lime = False
            st.session_state.show_dice = False
            st.rerun()
        except (ValueError, TypeError) as e:
            st.error(f"Error pada input: {str(e)}")
    
    # Tampilkan Hasil Prediksi
    if st.session_state.prediction_done:
        user_input_raw = st.session_state.user_input_raw
        input_df_processed = preprocess_input_data(user_input_raw, ALL_FEATURES)
        
        if input_df_processed is not None:
            # Simpan input user ke Supabase (jika terkonfigurasi)
            try:
                user_input_to_save = st.session_state.get('user_input_raw')
                if user_input_to_save and isinstance(user_input_to_save, dict):
                    ok, resp = insert_input_to_supabase(user_input_to_save)
                    if ok:
                        st.info("✅ Data input berhasil disimpan ke database.")
                    else:
                        # Jangan hentikan alur jika gagal, tampilkan notifikasi kecil
                        st.warning(f"Gagal menyimpan data ke database: {resp}")
            except Exception:
                # Jangan biarkan kegagalan penyimpanan menghentikan alur utama
                pass
            # Prediksi
            prediction_proba = model.predict_proba(input_df_processed[ALL_FEATURES]) # type: ignore
            predicted_class_index = np.argmax(prediction_proba[0])
            predicted_class = encoders[TARGET_NAME].classes_[predicted_class_index] # type: ignore
            
            # Tampilkan Hasil
            st.markdown("---")
            st.header("📊 Hasil Analisis")
            
            col_result1, col_result2 = st.columns([2, 1])
            
            with col_result1:
                st.subheader("🎯 Prediksi Model")
                st.markdown(f"### Status berat badan Anda: **{predicted_class.replace('_', ' ')}**")
            
            with col_result2:
                st.subheader("📈 Probabilitas")
                for i, class_name in enumerate(CLASS_NAMES): # type: ignore
                    prob = prediction_proba[0][i]
                    st.write(f"{class_name.replace('_', ' ')}: {prob:.2%}")
            
            # Tombol Analisis XAI
            st.markdown("---")
            st.subheader("🔍 Pilih Analisis Lanjutan (XAI)")
            
            col_lime, col_dice = st.columns(2)
            
            with col_lime:
                if st.button("🔬 Analisis LIME (Faktor Pengaruh)", use_container_width=True):
                    st.session_state.show_lime = True
                    st.session_state.show_dice = False
            
            with col_dice:
                if st.button("💡 Rekomendasi DiCE (Saran Perubahan)", use_container_width=True):
                    st.session_state.show_dice = True
                    st.session_state.show_lime = False
            
            # Tampilkan Analisis LIME
            if st.session_state.get('show_lime', False):
                st.markdown("---")
                st.subheader("🔬 Penjelasan LIME (Faktor-faktor yang Mempengaruhi Prediksi)")
                
                with st.spinner("Membuat penjelasan LIME..."):
                    try:
                        lime_exp = lime_explainer.explain_instance(
                            input_df_processed[ALL_FEATURES].values[0],
                            lambda x: predict_proba_catboost_for_lime(x, model, ALL_FEATURES),
                            num_features=8,
                            top_labels=1
                        )
                        
                        fig = lime_exp.as_pyplot_figure(label=predicted_class_index)
                        st.pyplot(fig, use_container_width=True)
                        
                        st.info("""
                        **Cara Membaca Grafik:**
                        - **Batang Hijau**: Fitur yang mendukung/memperkuat prediksi
                        - **Batang Merah**: Fitur yang menentang/melemahkan prediksi
                        - Semakin panjang batang, semakin besar pengaruhnya
                        """)
                        
                    except Exception as e:
                        st.error(f"Gagal membuat penjelasan LIME: {e}")
            
            # Tampilkan Rekomendasi DiCE
            if st.session_state.get('show_dice', False):
                st.markdown("---")
                st.subheader("💡 Rekomendasi DiCE (Saran Bertahap untuk Mencapai Berat Badan Ideal)")
                
                # Tentukan target berikutnya
                next_target, is_final_goal, all_steps = get_next_target_class(predicted_class, CLASS_NAMES)
                
                if predicted_class == next_target:
                    st.success("""
                    🎉 **Selamat!**
                    
                    Berat badan Anda sudah dalam kategori **Normal** atau lebih baik, sehingga tidak 
                    diperlukan rekomendasi perubahan. Pertahankan gaya hidup sehat Anda!
                    """)
                else:
                    
                    total_steps = len(all_steps) - 1
                    current_step = 1
                    # Opsi perubahan berat badan
                    allow_weight_change = st.checkbox(
                        "Izinkan saran yang mengubah berat badan (Weight)",
                        value=False,
                        help="Centang jika Anda menerima saran yang mengubah berat badan secara langsung"
                    )
                    
                    with st.spinner(f"Mencari rekomendasi untuk mencapai {next_target.replace('_', ' ')}..."):
                        try:
                            desired_class_index = int(encoders[TARGET_NAME].transform([next_target])[0]) # type: ignore
                            
                            dice_result = get_dice_recommendations(
                                x_train_encoded,
                                model,
                                encoders,
                                input_df_processed,
                                desired_class_index,
                                ALL_FEATURES,
                                allow_weight_change=allow_weight_change
                            )
                            
                            if dice_result and dice_result.cf_examples_list and dice_result.cf_examples_list[0].final_cfs_df is not None:
                                cf_df = dice_result.cf_examples_list[0].final_cfs_df
                                cf_df_decoded = decode_dice_dataframe(cf_df, encoders, ALL_FEATURES)
                                
                                # Data saat ini
                                q_df = input_df_processed.copy()
                                q_df[TARGET_NAME] = predicted_class
                                q_decoded = decode_dice_dataframe(q_df, encoders, ALL_FEATURES)
                                
                                # Tampilkan step description
                                st.markdown(f"### {get_step_description(predicted_class, next_target, current_step, total_steps)}")
                                
                                # Tampilkan perbandingan - VERTIKAL (atas-bawah)
                                st.markdown("#### 📋 Data Anda Saat Ini")
                                st.dataframe(q_decoded, use_container_width=True)
                                
                                st.markdown("---")
                                st.markdown("#### ⬇️ Rekomendasi Perubahan")
                                st.markdown(f"**Target: {next_target.replace('_', ' ')}**")
                                
                                st.dataframe(cf_df_decoded, use_container_width=True)
                                
                            else:
                                st.warning("""
                                ⚠️ Tidak dapat menemukan rekomendasi perubahan yang sesuai.
                                
                                **Saran:**
                                - Coba centang opsi "Izinkan perubahan berat badan"
                                - Konsultasikan dengan ahli gizi atau dokter untuk panduan personal
                                """)
                        
                        except ValueError:
                            st.error(f"❌ Kelas target '{next_target}' tidak ditemukan dalam model.")
                            st.info(f"Kelas yang tersedia: {', '.join(encoders[TARGET_NAME].classes_)}") # type: ignore
                        
                        except Exception as e:
                            st.error(f"❌ Gagal menghasilkan rekomendasi: {e}")
                            st.error(f"Traceback: {traceback.format_exc()}")
        
        else:
            st.error("Gagal memproses input data. Silakan periksa kembali data yang Anda masukkan.")

if __name__ == "__main__":
    run_prediction_app()