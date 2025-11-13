
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

def run_prediction_app():
    # Using a function ensures that all variables are locally scoped
    # and the app is cleaner.

    # ======================================================================================
    # 1. KONFIGURASI & SETUP APLIKASI
    # ======================================================================================

    st.set_page_config(page_title="Prediksi Obesitas (XAI)", layout="wide")

    # If authenticated, show user info and logout button in the sidebar
    with st.sidebar:
        st.title(f"👋 Halo, {st.session_state.get('user_name', 'Pengguna')}!")
        st.write(f"**Email:** {st.session_state.get('user')}") # Corrected from 'user_email'
        st.write(f"**Role:** {st.session_state.get('user_role')}")
        
        if st.button("🚪 Logout", use_container_width=True):
            logout() # This function from Login.py handles supabase signout and session state
            st.session_state.page = "login"
            st.rerun()

    # --- Path ke Aset Model (SESUAI DENGAN REPO ANDA) ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR,"Model_Website") 

    MODEL_PATH = os.path.join(MODEL_DIR, "catboost_obesity_model.cbm") 
    TARGET_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_y_v3.pkl")
    FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names_v3.pkl")
    CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names_y_v3.pkl")

    DATA_PATH = r"C:\Users\LENOVO\Documents\DENICO\Skripsi\Python\Dataset\combined_dataset.csv"

    # --- Konfigurasi Tipe Fitur (Berdasarkan Notebook) ---
    TARGET_NAME = 'NObeyesdad'

    CONTINUOUS_COLS = ['Age', 'Height', 'Weight']
    CATEGORICAL_COLS = [
        'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 
        'family_history_with_overweight', 'CAEC', 'MTRANS'
    ]
    ORDINAL_COLS = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    ALL_CATEGORICAL_COLS = CATEGORICAL_COLS + ORDINAL_COLS
    ALL_FEATURES = CONTINUOUS_COLS + ALL_CATEGORICAL_COLS

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

    # ======================================================================================
    # 2. FUNGSI PREPROCESSING & PEMUATAN ASET
    # ======================================================================================

    def preprocess_input_data(input_dict, all_features_list):
        df = pd.DataFrame([input_dict])
        if df.empty: return None
        try:
            for col in CONTINUOUS_COLS:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any() or np.isinf(df[col]).any():
                    if col == 'Age': df[col] = df[col].fillna(25)
                    elif col == 'Height': df[col] = df[col].fillna(170)
                    elif col == 'Weight': df[col] = df[col].fillna(70)
                if col == 'Height' and df[col].iloc[0] < 10:
                    df[col] = df[col] * 100
                if col == 'Age': df[col] = df[col].round().astype(int)
                elif col == 'Height': df[col] = df[col].round().astype(int)
                else: df[col] = df[col].round(3).astype(float)
            df['Gender'] = df['Gender'].map(GENDER_MAP).astype(int)
            df['family_history_with_overweight'] = df['family_history_with_overweight'].map(FAMILY_HISTORY_MAP).astype(int)
            df['FAVC'] = df['FAVC'].map(FAVC_MAP).astype(int)
            df['SCC'] = df['SCC'].map(SCC_MAP).astype(int)
            df['SMOKE'] = df['SMOKE'].map(SMOKE_MAP).astype(int)
            df['CAEC'] = df['CAEC'].map(CAEC_MAP).astype(int)
            df['CALC'] = df['CALC'].map(CALC_MAP).astype(int)
            df['MTRANS'] = df['MTRANS'].map(MTRANS_MAP).astype(int)
            ordinal_defaults = {'FCVC': 2, 'NCP': 3, 'CH2O': 2, 'FAF': 1, 'TUE': 1}
            for col in ORDINAL_COLS:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(ordinal_defaults[col]).round().astype(int)
        except Exception as e:
            st.error(f"Error saat preprocessing data input: {e}")
            return None
        if not all_features_list:
            st.error("Daftar fitur (ALL_FEATURES) kosong.")
            return None
        try:
            df = df[all_features_list]
        except KeyError as e:
            st.error(f"Error: Kolom yang hilang saat preprocessing: {e}")
            return None
        return df

    @st.cache_resource
    def load_all_assets():
        try:
            model = CatBoostClassifier()
            model.load_model(MODEL_PATH)
            target_encoder = joblib.load(TARGET_ENCODER_PATH)
            all_features = joblib.load(FEATURE_NAMES_PATH)
            class_names = joblib.load(CLASS_NAMES_PATH)
            encoders = { TARGET_NAME: target_encoder }
            x_train_raw = pd.read_csv(DATA_PATH, usecols=all_features + [TARGET_NAME])
            x_train_raw = x_train_raw.dropna().reset_index(drop=True)
            x_train_processed = x_train_raw.copy()
            for col in CONTINUOUS_COLS:
                x_train_processed[col] = pd.to_numeric(x_train_processed[col], errors='coerce')
            x_train_processed['Gender'] = x_train_processed['Gender'].map(GENDER_MAP)
            x_train_processed['family_history_with_overweight'] = x_train_processed['family_history_with_overweight'].map(FAMILY_HISTORY_MAP)
            x_train_processed['FAVC'] = x_train_processed['FAVC'].map(FAVC_MAP)
            x_train_processed['SCC'] = x_train_processed['SCC'].map(SCC_MAP)
            x_train_processed['SMOKE'] = x_train_processed['SMOKE'].map(SMOKE_MAP)
            x_train_processed['CAEC'] = x_train_processed['CAEC'].map(CAEC_MAP)
            x_train_processed['CALC'] = x_train_processed['CALC'].map(CALC_MAP)
            x_train_processed['MTRANS'] = x_train_processed['MTRANS'].map(MTRANS_MAP)
            for col in ORDINAL_COLS:
                x_train_processed[col] = pd.to_numeric(x_train_processed[col], errors='coerce')
            x_train_processed = x_train_processed.dropna()
            for col in ALL_CATEGORICAL_COLS:
                x_train_processed[col] = x_train_processed[col].round().astype(int)
            for col in CONTINUOUS_COLS:
                if col == 'Age': x_train_processed[col] = x_train_processed[col].round().astype(int)
                else: x_train_processed[col] = x_train_processed[col].astype(float)
            x_train_encoded = x_train_processed[all_features]
            return model, encoders, all_features, class_names, x_train_encoded
        except Exception as e:
            st.error(f"Gagal memuat aset penting: {e}")
            return None, None, None, None, None

    # ... (The rest of the functions: initialize_lime_explainer, predict_proba_catboost_for_lime, DiceCatBoostWrapper, etc.)
    # Note: These functions are now defined inside run_prediction_app, which is fine.
    
    @st.cache_resource
    def initialize_lime_explainer(_X_train_encoded, all_features_list, class_names_list):
        training_values = _X_train_encoded[all_features_list].values 
        categorical_feature_indices = [_X_train_encoded.columns.get_loc(col) for col in ALL_CATEGORICAL_COLS if col in _X_train_encoded.columns]
        categorical_names_map = {}
        if 'Gender' in all_features_list:
            categorical_names_map[all_features_list.index('Gender')] = list(GENDER_MAP.keys())
        if 'family_history_with_overweight' in all_features_list:
            categorical_names_map[all_features_list.index('family_history_with_overweight')] = list(FAMILY_HISTORY_MAP.keys())
        lime_explainer = LimeTabularExplainer(training_data=training_values, feature_names=all_features_list, class_names=class_names_list, categorical_features=categorical_feature_indices, categorical_names=categorical_names_map, mode='classification', random_state=42)
        return lime_explainer

    def predict_proba_catboost_for_lime(data, model_obj, all_features_list):
        if isinstance(data, (list, np.ndarray)):
            arr = np.array(data)
            df = pd.DataFrame(arr.reshape(-1, len(all_features_list)), columns=all_features_list)
        else:
            df = pd.DataFrame(data)
        input_encoded = df.reindex(columns=all_features_list, fill_value=0)
        for col in ALL_CATEGORICAL_COLS:
            if col in input_encoded.columns:
                numeric_col = pd.to_numeric(input_encoded[col], errors='coerce').fillna(0)
                input_encoded[col] = numeric_col.round().astype(int)
        for col in CONTINUOUS_COLS:
            if col in input_encoded.columns:
                input_encoded[col] = pd.to_numeric(input_encoded[col], errors='coerce').fillna(0).astype(float)
        return model_obj.predict_proba(input_encoded[all_features_list])

    class DiceCatBoostWrapper:
        def __init__(self, model, feature_names, continuous_features, categorical_features):
            self.model, self.feature_names, self.continuous_features, self.categorical_features = model, feature_names, continuous_features, categorical_features
        def predict_proba(self, X):
            if isinstance(X, np.ndarray): X = pd.DataFrame(X, columns=self.feature_names)
            X_copy = X.copy()
            for col in self.continuous_features:
                if col in X_copy.columns:
                    X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(X_copy[col].median() or 0)
            for col in self.categorical_features:
                if col in X_copy.columns:
                    X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(0).round().astype(int)
            X_copy = X_copy[self.feature_names]
            return self.model.predict_proba(X_copy)

    def get_dice_recommendations(_x_train_encoded, model_obj, encoders, input_df_processed, desired_class_index, all_features_list):
        # This function remains complex, keeping its internal logic.
        progress_text = "Mencari rekomendasi perubahan..."
        my_bar = st.progress(0, text=progress_text)
        try:
            my_bar.progress(10, text="Mempersiapkan data training...")
            df_dice = _x_train_encoded.copy()
            string_predictions = model_obj.predict(df_dice[all_features_list]).ravel()
            df_dice[TARGET_NAME] = encoders[TARGET_NAME].transform(string_predictions)
            my_bar.progress(30, text="Mengatur batasan nilai fitur...")
            dice_continuous_features = CONTINUOUS_COLS.copy()
            categorical_features_for_dice = [col for col in all_features_list if col not in dice_continuous_features]
            data_interface = dice_ml.Data(dataframe=df_dice, continuous_features=dice_continuous_features, outcome_name=TARGET_NAME)
            wrapped_model = DiceCatBoostWrapper(model_obj, all_features_list, CONTINUOUS_COLS, ALL_CATEGORICAL_COLS)
            model_interface = dice_ml.Model(model=wrapped_model, backend="sklearn", model_type='classifier')
            my_bar.progress(70, text="Menyiapkan algoritma pencarian...")
            dice_explainer = Dice(data_interface, model_interface, method="genetic")
            query_instance = input_df_processed[all_features_list].copy()
            
            # Mendefinisikan fitur-fitur yang umumnya TIDAK diubah untuk counterfactuals perilaku.
            # Ini termasuk fitur demografi, atribut statis, dan fitur kontinu yang bukan merupakan kebiasaan langsung.
            non_actionable_or_static_features = ['Age', 'Height', 'Weight', 'Gender', 'family_history_with_overweight']

            # Fitur yang akan diubah adalah semua fitur yang bukan kontinu dan tidak termasuk dalam daftar non-actionable/statis.
            # Ini memastikan hanya fitur perilaku/gaya hidup yang dipertimbangkan untuk counterfactuals.
            features_to_vary_list = [f for f in all_features_list if f not in CONTINUOUS_COLS and f not in non_actionable_or_static_features]

            my_bar.progress(90, text="Mencari rekomendasi terbaik...")
            dice_result = dice_explainer.generate_counterfactuals(query_instance, total_CFs=3, desired_class=desired_class_index, features_to_vary=features_to_vary_list, permitted_range=None)
            my_bar.progress(100, text="Selesai!")
            return dice_result
        except Exception as e:
            st.error(f"Error saat mencari rekomendasi DiCE: {e}")
            return None
        finally:
            my_bar.empty()

    def decode_dice_dataframe(df_dice_output, encoders, all_features_list):
        df_decoded = df_dice_output.copy()
        categorical_decode_maps = {
            'Gender': {'0': 'Female', '1': 'Male'}, 'family_history_with_overweight': {'0': 'no', '1': 'yes'},
            'FAVC': {'0': 'no', '1': 'yes'}, 'SCC': {'0': 'no', '1': 'yes'}, 'SMOKE': {'0': 'no', '1': 'yes'},
            'CAEC': {'0': 'no', '1': 'Sometimes', '2': 'Frequently', '3': 'Always'},
            'CALC': {'0': 'no', '1': 'Sometimes', '2': 'Frequently', '3': 'Always'},
            'MTRANS': {'0': 'Walking', '1': 'Public_Transportation', '2': 'Bike', '3': 'Motorbike', '4': 'Automobile'}
        }
        for col_name, mapping in categorical_decode_maps.items():
            if col_name in df_decoded.columns: df_decoded[col_name] = df_decoded[col_name].astype(str).map(mapping)
        for col in CONTINUOUS_COLS:
            if col in df_decoded.columns: df_decoded[col] = pd.to_numeric(df_decoded[col], errors='coerce').round(2)
        for col in ORDINAL_COLS:
            if col in df_decoded.columns: df_decoded[col] = pd.to_numeric(df_decoded[col], errors='coerce').round().astype(int)
        if TARGET_NAME in df_decoded.columns:
            df_decoded[TARGET_NAME] = encoders[TARGET_NAME].inverse_transform(pd.to_numeric(df_decoded[TARGET_NAME], errors='coerce').round().astype(int))
        return df_decoded[all_features_list + [TARGET_NAME]]

    # ======================================================================================
    # 5. APLIKASI UTAMA STREAMLIT
    # ======================================================================================

    loaded_assets = load_all_assets()
    model, encoders, ALL_FEATURES, CLASS_NAMES, x_train_encoded = loaded_assets

    if any(asset is None for asset in loaded_assets):
        st.error("Gagal memuat aset penting. Aplikasi berhenti.")
        st.stop()
    
    lime_explainer = initialize_lime_explainer(x_train_encoded, ALL_FEATURES, CLASS_NAMES)

    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False
        st.session_state.user_input_raw = None

    st.sidebar.title("Informasi")
    st.sidebar.write("Aplikasi ini memprediksi tingkat obesitas berdasarkan kebiasaan sehari-hari.")
    st.sidebar.markdown("---")
    if st.sidebar.button("Hapus Cache & Muat Ulang"):
        st.cache_resource.clear()
        st.rerun()

    st.title("Prediksi Tingkat Obesitas & Rekomendasi Perubahan")
    st.markdown("Masukkan data Anda di bawah ini untuk mendapatkan prediksi dan penjelasan (XAI).")

    st.header("Masukkan Data Anda")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox('Gender', list(GENDER_MAP.keys()), key='gender')
        age = st.number_input('Umur', 1, 100, 22, 1, key='age')
        height_cm = st.number_input('Tinggi Badan (cm)', 100, 250, 168, key='height_cm')
        weight = st.number_input('Berat Badan (kg)', 30, 200, 63, key='weight')
        family_history_with_overweight = st.selectbox('Riwayat keluarga obesitas?', list(FAMILY_HISTORY_MAP.keys()), format_func=lambda x: 'Tidak' if x == 'no' else 'Ya', key='family')
    with col2:
        favc = st.selectbox('Makan makanan tinggi kalori?', list(FAVC_MAP.keys()), index=1, format_func=lambda x: 'Tidak' if x == 'no' else 'Ya', key='favc')
        fcvc = st.selectbox('Frekuensi makan sayur', [1, 2, 3], index=2, format_func=lambda x: {1: 'Tidak Pernah', 2: 'Kadang-kadang', 3: 'Selalu'}[x], key='fcvc')
        ncp = st.number_input('Makan utama per hari?', 1, 4, 3, 1, key='ncp')
        caec_choice = st.selectbox('Makan cemilan?', list(CAEC_MAP.keys()), index=1, key='caec')
        calc_choice = st.selectbox('Minum alkohol?', list(CALC_MAP.keys()), index=0, key='calc')
    with col3:
        smoke = st.selectbox('Merokok?', list(SMOKE_MAP.keys()), format_func=lambda x: 'Tidak' if x == 'no' else 'Ya', key='smoke')
        ch2o = st.selectbox('Minum air per hari?', [1, 2, 3], index=1, format_func=lambda x: {1: '<1L', 2: '1-2L', 3: '>2L'}[x], key='ch2o')
        scc = st.selectbox('Memantau kalori?', list(SCC_MAP.keys()), format_func=lambda x: 'Tidak' if x == 'no' else 'Ya', key='scc')
        faf = st.selectbox('Aktivitas fisik (hari/minggu)?', [0, 1, 2, 3], index=1, key='faf')
        tue = st.selectbox('Penggunaan gawai (jam/hari)?', [0, 1, 2], index=1, key='tue')
        mtrans = st.selectbox('Transportasi utama?', list(MTRANS_MAP.keys()), index=1, key='mtrans')

    if st.button("Dapatkan Prediksi dan Penjelasan", type="primary"):
        try:
            height = float(height_cm) / 100.0
            st.session_state.user_input_raw = {
                'Age': int(age), 'Gender': str(gender), 'Height': height, 'Weight': float(weight),
                'CALC': str(calc_choice), 'FAVC': str(favc), 'FCVC': str(fcvc), 'NCP': str(ncp),
                'SCC': str(scc), 'SMOKE': str(smoke), 'CH2O': str(ch2o),
                'family_history_with_overweight': str(family_history_with_overweight),
                'FAF': str(faf), 'TUE': str(tue), 'CAEC': str(caec_choice), 'MTRANS': str(mtrans)
            }
            st.session_state.prediction_done = True
            st.rerun()
        except (ValueError, TypeError) as e:
            st.error(f"Error pada input: Pastikan semua nilai numerik valid. Detail: {str(e)}")

    if st.session_state.prediction_done:
        user_input_raw = st.session_state.user_input_raw
        input_df_processed = preprocess_input_data(user_input_raw, ALL_FEATURES)
        if input_df_processed is not None and model is not None:
            prediction_proba = model.predict_proba(input_df_processed[ALL_FEATURES])
            predicted_class_index = np.argmax(prediction_proba[0])
            predicted_class = CLASS_NAMES_PATH[predicted_class_index]
            
            st.header("Hasil Analisis")
            st.subheader("Prediksi Model")
            st.markdown(f"Status berat badan Anda diprediksi: **{predicted_class.replace('_', ' ')}**")
            
            tab1, tab2 = st.tabs(["Penjelasan (LIME)", "Rekomendasi (DiCE)"])
            with tab1:
                with st.spinner("Membuat penjelasan LIME..."):
                    try:
                        lime_exp = lime_explainer.explain_instance(input_df_processed[ALL_FEATURES].values[0], lambda x: predict_proba_catboost_for_lime(x, model, ALL_FEATURES), num_features=8, top_labels=1)
                        fig = lime_exp.as_pyplot_figure(label=predicted_class_index)
                        st.pyplot(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Gagal membuat penjelasan LIME: {e}")
            with tab2:
                if predicted_class != 'Normal_Weight':
                    desired_target_class = 'Normal_Weight'
                    st.info(f"Mencari rekomendasi untuk mencapai: **{desired_target_class.replace('_', ' ')}**")
                    with st.spinner("Mencari..."):
                        try:
                            desired_target_index = list(CLASS_NAMES_PATH).index(desired_target_class)
                            dice_result = get_dice_recommendations(x_train_encoded, model, encoders, input_df_processed, desired_target_index, ALL_FEATURES)
                            if dice_result and dice_result.cf_examples_list and dice_result.cf_examples_list[0].final_cfs_df is not None:
                                cf_df_decoded = decode_dice_dataframe(dice_result.cf_examples_list[0].final_cfs_df, encoders, ALL_FEATURES)
                                st.dataframe(cf_df_decoded)
                            else:
        
                                st.warning("Tidak dapat menemukan rekomendasi.")
                        except Exception as e:
                            st.error(f"Gagal menghasilkan rekomendasi DiCE: {e}")
                else:
                    st.success("Selamat! Berat badan Anda sudah Normal.")
