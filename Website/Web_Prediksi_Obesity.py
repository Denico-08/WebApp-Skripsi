import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from dataclasses import dataclass, asdict
from catboost import CatBoostClassifier

from User import User
from Connection.supabase_client import (
    insert_input_to_supabase,
    insert_faktor_dominan,
    insert_prediction_to_supabase,
    insert_rekomendasi_to_supabase,
)

from config import (
    TARGET_NAME, CONTINUOUS_COLS, ORDINAL_COLS,
    GENDER_MAP, FAMILY_HISTORY_MAP, FAVC_MAP, SCC_MAP, SMOKE_MAP,
    CAEC_MAP, CALC_MAP, MTRANS_MAP
)

# Import Helper XAI (Langsung Function)
from XAI.dice_helpers import (
    decode_dice_dataframe, get_dice_recommendations, 
    summarize_dice_changes)
from XAI.lime_helpers import (
    initialize_lime_explainer, 
    predict_proba_catboost_for_lime, 
    generate_lime_explanation_text, get_step_description, get_next_target_class
)

# ==============================================================================
# 1. CLASS DATASET
# ==============================================================================
class Dataset:
    def __init__(self):
        self.data_train_encoded = None
        self.feature_names = None
        self.class_names = None
        
    def LoadDataset(self):
        """Memuat data training dan metadata yang dibutuhkan untuk XAI"""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(base_dir, "Model_Website")
            
            # Paths
            path_train = os.path.join(model_dir, "X_train_smote.pkl")
            path_features = os.path.join(model_dir, "X_ClassNames.pkl")
            path_classes = os.path.join(model_dir, "Y_ClassNames.pkl")

            # Load Assets
            self.feature_names = joblib.load(path_features)
            self.class_names = joblib.load(path_classes)
            x_train_array = joblib.load(path_train)
            
            # Konversi ke DataFrame
            if isinstance(x_train_array, np.ndarray):
                if self.feature_names:
                    self.data_train_encoded = pd.DataFrame(x_train_array, columns=self.feature_names)
            elif isinstance(x_train_array, pd.DataFrame):
                self.data_train_encoded = x_train_array
            
            return True
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return False

# ==============================================================================
# 2. CLASS INPUT DATA
# ==============================================================================
@dataclass
class InputData:
    # Atribut Data
    Age: int
    Gender: str
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str
    Id: int = 0 

    def Get_raw_Data(self) -> dict:
        return asdict(self)

    def preprocess(self, feature_columns: list) -> pd.DataFrame:
        """Preprocessing data input user agar sesuai format model"""
        input_dict = self.Get_raw_Data()
        df = pd.DataFrame([input_dict])

        if df.empty: return None #type: ignore

        try:
            # 1. Handle Fitur Kontinyu
            for col in CONTINUOUS_COLS:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Handle missing/inf values
                if df[col].isnull().any() or np.isinf(df[col]).any():
                    defaults = {'Age': 25, 'Height': 170, 'Weight': 70}
                    df[col] = df[col].fillna(defaults.get(col, 0))
                
                # Konversi cm ke meter jika perlu
                if col == 'Height' and df[col].iloc[0] < 10:
                    df[col] = df[col] * 100
                
                # Casting
                if col == 'Age' or col == 'Height':
                    df[col] = df[col].round().astype(int)
                else:
                    df[col] = df[col].round(3).astype(float)
            
            # 2. Encode Kategorikal
            df['Gender'] = df['Gender'].map(GENDER_MAP).astype(int)
            df['family_history_with_overweight'] = df['family_history_with_overweight'].map(FAMILY_HISTORY_MAP).astype(int)
            df['FAVC'] = df['FAVC'].map(FAVC_MAP).astype(int)
            df['SCC'] = df['SCC'].map(SCC_MAP).astype(int)
            df['SMOKE'] = df['SMOKE'].map(SMOKE_MAP).astype(int)
            df['CAEC'] = df['CAEC'].map(CAEC_MAP).astype(int)
            df['CALC'] = df['CALC'].map(CALC_MAP).astype(int)
            df['MTRANS'] = df['MTRANS'].map(MTRANS_MAP).astype(int)
            
            # 3. Proses Ordinal
            ordinal_defaults = {'FCVC': 2, 'NCP': 3, 'CH2O': 2, 'FAF': 1, 'TUE': 1}
            for col in ORDINAL_COLS:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(ordinal_defaults[col]).round().astype(int)
            
            # 4. Reorder Kolom
            df = df[feature_columns]
            return df

        except Exception as e:
            st.error(f"Error preprocessing: {e}")
            return None #type: ignore

# ==============================================================================
# 3. CLASS HASIL PREDIKSI
# ==============================================================================
class Hasil_Prediksi:
    def __init__(self, kategori_berat: str, probabilitas: float, data_input: InputData):
        self.id = None 
        self.kategori_berat = kategori_berat
        self.probabilitas = probabilitas
        self.data_input = data_input
        # Atribut tambahan untuk menyimpan hasil analisis sementara (opsional)
        self.lime_explanation = None
        self.dice_recommendation = None

    def Display_Result(self) -> str:
        return self.kategori_berat.replace('_', ' ')

    def Save_Result(self, user_id: str):
        """Menyimpan input dan hasil prediksi ke Supabase"""
        try:
            # 1. Simpan Input
            ok_input, id_input = insert_input_to_supabase(self.data_input.Get_raw_Data(), user_id)
            
            if ok_input and id_input:
                self.data_input.Id = id_input 
                
                # 2. Simpan Prediksi
                ok_pred, resp_pred = insert_prediction_to_supabase(
                    id_input=id_input,
                    hasil_prediksi=self.kategori_berat,
                    probabilitas=self.probabilitas
                )
                
                if ok_pred:
                    # Ekstrak ID Prediksi
                    if isinstance(resp_pred, list) and len(resp_pred) > 0:
                        self.id = resp_pred[0].get('ID_Prediksi') or resp_pred[0].get('id')
                    elif isinstance(resp_pred, dict):
                        self.id = resp_pred.get('ID_Prediksi') or resp_pred.get('id')
                    return True
            return False
        except Exception as e:
            st.warning(f"Gagal menyimpan data: {e}")
            return False

# ==============================================================================
# 4. CLASS MODEL PREDIKSI
# ==============================================================================
class Model_Prediksi:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.feature_names = []
        self.class_names = []
        
    def loadmodel(self):
        """Memuat Model CatBoost dan Encoders"""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(base_dir, "Model_Website")
            
            # Load Model
            self.model = CatBoostClassifier()
            self.model.load_model(os.path.join(model_dir, "CatBoost_Model.cbm"))
            
            # Load Metadata
            self.encoders[TARGET_NAME] = joblib.load(os.path.join(model_dir, "Y_Processed.pkl"))
            self.feature_names = joblib.load(os.path.join(model_dir, "X_ClassNames.pkl"))
            self.class_names = joblib.load(os.path.join(model_dir, "Y_ClassNames.pkl"))
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False

    def predict(self, data: InputData) -> Hasil_Prediksi:
        """Melakukan prediksi berdasarkan InputData"""
        # 1. Preprocess
        df_processed = data.preprocess(self.feature_names)
        if df_processed is None: return None #type: ignore
            
        # 2. Predict Probabilitas
        prediction_proba = self.model.predict_proba(df_processed) #type: ignore
        predicted_class_index = np.argmax(prediction_proba[0])
        
        # 3. Decode
        predicted_class = self.encoders[TARGET_NAME].classes_[predicted_class_index]
        prediction_probability = prediction_proba[0][predicted_class_index]
        
        # 4. Return Object
        return Hasil_Prediksi(
            kategori_berat=predicted_class,
            probabilitas=float(prediction_probability),
            data_input=data
        )

@st.cache_resource
def initialize_system():
    # 1. Init Classes
    dataset = Dataset()
    model = Model_Prediksi()
    
    # 2. Load Data & Model
    if not dataset.LoadDataset(): return None
    if not model.loadmodel(): return None
    
    # 3. Init LIME Explainer (Menggunakan Function Import, bukan Class)
    # Ini disimpan supaya tidak perlu init berulang-ulang
    lime_explainer = None
    if dataset.data_train_encoded is not None:
        lime_explainer = initialize_lime_explainer(
            dataset.data_train_encoded, 
            dataset.feature_names, 
            dataset.class_names
        )
    
    return dataset, model, lime_explainer

def run_prediction_app():
    st.set_page_config(page_title="Prediksi Obesitas (Class Integrated)", layout="wide")
    
    # Sidebar
    with st.sidebar:
        st.title(f"Halo, {st.session_state.get('user_name', 'Pengguna')}!")
        st.write(f"Email: {st.session_state.get('user')}")
        if st.button("Riwayat"): st.session_state.page = "riwayat"; st.rerun()
        if st.button("Logout"): User.logout(); st.session_state.page = "login"; st.rerun()

    if st.session_state.page == "riwayat":
        # Logika untuk menampilkan halaman history via Class User
        current_user_id = st.session_state.get('user_id')
        current_user_name = st.session_state.get('user_name')
        
        if current_user_id:
            # Buat objek user dengan ID yang sedang login
            user_obj = User(user_id=current_user_id, name=current_user_name) # type: ignore
            user_obj.render_history_page()
        else:
            st.warning("Sesi kadaluarsa. Silakan login kembali.")
    # Initialize System
    system = initialize_system()
    if not system: st.stop()
    dataset, model, lime_explainer = system

    # Session State
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
        st.session_state.show_lime = False
        st.session_state.show_dice = False

    # Main UI
    st.title("Prediksi Tingkat Obesitas")
    st.header("Masukkan Data Anda")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Data Pribadi")
        gender = st.selectbox('Gender', list(GENDER_MAP.keys()))
        age = st.number_input('Umur', 1, 100, 25)
        height = st.number_input('Tinggi (cm)', 100, 250, 170)
        weight = st.number_input('Berat (kg)', 30, 200, 70)
        family = st.selectbox(
            'Riwayat keluarga obesitas?', 
            list(FAMILY_HISTORY_MAP.keys()),
            format_func=lambda x: 'Tidak' if x == 'no' else 'Ya',
            key='family'
        )

    with col2:
        st.subheader("Kebiasaan Makan")
        favc = st.selectbox(
            'Apakah anda mengonsumsi makanan cepat saji lebih dari 2x seminggu?',
            list(FAVC_MAP.keys()),
            index=1,
            format_func=lambda x: 'Tidak' if x == 'no' else 'Ya',
            key='favc'
        )
        ncp = st.number_input('Makan utama per hari', min_value=1, max_value=4, value=3, step=1, key='ncp')
        fcvc = st.selectbox(
            'Frekuensi makan sayur setiap makan',
            [1, 2, 3],
            index=2,
            format_func=lambda x: {1: 'Tidak Pernah', 2: 'Setengah dari jumlah makan per hari', 3: 'Setiap Makan'}[x],
            key='fcvc'
        )
        caec = st.selectbox(
            'Seberapa sering makan cemilan tinggi kalori di antara waktu makan dalam seminggu?',
            list(CAEC_MAP.keys()),
            index=1,  # Default: Sometimes
            format_func=lambda x: {'no': 'Tidak Pernah', 'Sometimes': '1-2x/minggu','Frequently': '3-5x/minggu', 'Always': '6-7x/minggu'}[x],
            key='caec'
        )
        calc = st.selectbox('Berapa porsi anda mengonsumsi alkohol bir per hari (1 porsi: 350 ml)?',
            list(CALC_MAP.keys()),
            index=0, # Default: no
            format_func=lambda x: {'no': 'Tidak Pernah', 'Sometimes': '2 porsi', 'Frequently': '3 porsi', 'Always': '> 4 porsi'}[x],
            key='calc'
        )

    with col3:
        st.subheader("Gaya Hidup")
        smoke = st.selectbox(
            'Apakah anda merokok?',
            list(SMOKE_MAP.keys()),
            format_func=lambda x: 'Tidak' if x == 'no' else 'Ya',
            key='smoke'
        )
        ch2o = st.selectbox(
            'Jumlah minum air per hari',
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
        faf = st.selectbox('Berapa menit anda melakukan aktivitas fisik (jalan, bersepeda, olahraga ringan) dalam seminggu', [0, 1, 2, 3], index=1,
            format_func=lambda x: {0: '< 15 menit', 1: '15 - 30 menit', 2: '30 - 60 menit', 3: '60+ menit>'}[x],
            key='faf')
        tue = st.selectbox('Penggunaan gawai (jam/hari)', [0, 1, 2], index=1, 
            format_func=lambda x: {0: '< 1 jam', 1: '1 - 2 jam', 2: '>2 jam'}[x],
            key='tue')
        mtrans = st.selectbox('Transportasi utama', list(MTRANS_MAP.keys()), index=1, key='mtrans')

    # Prediksi
    st.markdown("---")
    if st.button("Prediksi Sekarang", type="primary", use_container_width=True):
        try:
            # Create Input Object
            input_data = InputData(
                Age=int(age), Gender=str(gender), Height=float(height)/100.0, Weight=float(weight),
                family_history_with_overweight=str(family), FAVC=str(favc), FCVC=float(fcvc),
                NCP=float(ncp), CAEC=str(caec), SMOKE=str(smoke), CH2O=float(ch2o), SCC=str(scc),
                FAF=float(faf), TUE=float(tue), CALC=str(calc), MTRANS=str(mtrans)
            )
            
            # Predict
            hasil = model.predict(input_data)
            
            if hasil:
                st.session_state.prediction_result = hasil
                st.session_state.show_lime = False
                st.session_state.show_dice = False
                
                # Save
                user_id = st.session_state.get('user_id')
                if user_id: hasil.Save_Result(user_id)
                st.rerun()
                
        except Exception as e:
            st.error(f"Input Error: {e}")

    # Result Display
    if st.session_state.prediction_result:
        hasil = st.session_state.prediction_result
        st.markdown("---")
        c1, c2 = st.columns([2,1])
        with c1: st.success(f"**Hasil:** {hasil.Display_Result()}")
        with c2: st.info(f"**Probabilitas:** {hasil.probabilitas:.2%}")

        # XAI Section
        st.markdown("### Analisis Lanjutan")
        col_xai1, col_xai2 = st.columns(2)
        with col_xai1: 
            if st.button("🔍 Analisis LIME"): st.session_state.show_lime=True; st.session_state.show_dice=False
        with col_xai2:
            if st.button("💡 Rekomendasi DiCE"): st.session_state.show_dice=True; st.session_state.show_lime=False

        # ------------------------------------------------------------------
        # LOGIKA LIME (LANGSUNG PAKAI FUNCTION IMPORT)
        # ------------------------------------------------------------------
        if st.session_state.show_lime:
            with st.spinner("Analyzing..."):
                if lime_explainer:
                    # Preprocess ulang untuk LIME
                    df_processed = hasil.data_input.preprocess(model.feature_names)
                    pred_idx = list(model.encoders[TARGET_NAME].classes_).index(hasil.kategori_berat)
                    
                    # 1. Generate LIME instance
                    # Kita gunakan 'predict_proba_catboost_for_lime' dari import
                    lime_exp = lime_explainer.explain_instance(
                        df_processed.values[0],
                        lambda x: predict_proba_catboost_for_lime(x, model.model, model.feature_names),
                        num_features=7,
                        top_labels=1
                    )
                    
                    # 2. Generate Text
                    lime_text = generate_lime_explanation_text(
                        lime_exp,
                        pred_idx,
                        hasil.kategori_berat,
                        hasil.data_input.Get_raw_Data()
                    )
                    
                    # Tampilkan
                    st.pyplot(lime_exp.as_pyplot_figure(label=pred_idx))
                    st.info(lime_text)
                    
                    # Simpan Faktor Dominan ke DB
                    if hasil.id:
                        try:
                            top_list = lime_exp.as_list(label=pred_idx)
                            top_features = {str(feat): float(weight) for feat, weight in top_list[:5]}
                            insert_faktor_dominan(hasil.id, top_features)
                        except Exception: pass
                else:
                    st.error("LIME Explainer belum siap (Dataset mungkin gagal dimuat).")

        # ------------------------------------------------------------------
        # LOGIKA DiCE (LANGSUNG PAKAI FUNCTION IMPORT)
        # ------------------------------------------------------------------
        if st.session_state.show_dice:
            with st.spinner("Generating Recommendations..."):
                df_processed = hasil.data_input.preprocess(model.feature_names)
                
                # 1. Tentukan Target
                next_target, is_final, all_steps = get_next_target_class(hasil.kategori_berat, model.class_names)
                
                if hasil.kategori_berat == next_target:
                    st.success("Berat badan sudah ideal.")
                else:
                    # 2. Generate Recommendations
                    desired_class_index = int(model.encoders[TARGET_NAME].transform([next_target])[0])
                    
                    dice_result = get_dice_recommendations(
                        dataset.data_train_encoded, model.model, model.encoders,
                        df_processed, desired_class_index, model.feature_names
                    )
                    
                    if dice_result and dice_result.cf_examples_list and dice_result.cf_examples_list[0].final_cfs_df is not None:
                        cf_df = dice_result.cf_examples_list[0].final_cfs_df
                        
                        # Decode & Summarize
                        cf_decoded = decode_dice_dataframe(cf_df, model.encoders, model.feature_names)
                        summary = summarize_dice_changes(dice_result, df_processed, model.encoders, model.feature_names)
                        step_desc = get_step_description(hasil.kategori_berat, next_target, 1, len(all_steps)-1)
                        
                        # Tampilkan
                        st.write(step_desc)
                        if summary:
                            st.info("💡 **Saran Perubahan:**")
                            for msg in summary: st.markdown(f"- {msg}")
                        st.dataframe(cf_decoded)
                        
                        # Simpan Rekomendasi
                        if hasil.id:
                            try:
                                insert_rekomendasi_to_supabase(
                                    id_prediksi=hasil.id, target_prediksi=next_target,
                                    perubahan_prediksi=cf_decoded.to_dict('records')
                                )
                            except Exception: pass
                    else:
                        st.warning("Tidak ada rekomendasi spesifik.")

if __name__ == "__main__":
    run_prediction_app()