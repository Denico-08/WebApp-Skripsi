import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from matplotlib import pyplot as plt
from catboost import CatBoostClassifier
from lime.lime_tabular import LimeTabularExplainer
import traceback
from Login import logout
from Connection.supabase_client import insert_input_to_supabase
from XAI.dice_helpers import decode_dice_dataframe, get_dice_recommendations
from XAI.lime_helpers import initialize_lime_explainer, predict_proba_catboost_for_lime, get_step_description, get_next_target_class

from config import (
    TARGET_NAME, CONTINUOUS_COLS, ALL_CATEGORICAL_COLS, CATEGORICAL_COLS, ORDINAL_COLS,
    GENDER_MAP, FAMILY_HISTORY_MAP, FAVC_MAP, SCC_MAP, SMOKE_MAP,
    CAEC_MAP, CALC_MAP, MTRANS_MAP
)

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