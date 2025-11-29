import streamlit as st
import pandas as pd
import numpy as np

import os
import seaborn as sns
from matplotlib import pyplot as plt
from Role import Role
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Import Class User (Parent Class)
from User import User

# Import Class Pendukung dari file utama
# Pastikan Web_Prediksi_Obesity.py ada di folder yang sama
from Web_Prediksi_Obesity import Dataset, Model_Prediksi 
from Connection.supabase_client import get_supabase_client
from config import TARGET_NAME

# ==============================================================================
# CLASS ADMIN (Inherits from User)
# ==============================================================================
class Admin(User):
    def __init__(self, email=None, name=None, user_id=None):
        # Panggil konstruktor parent (User)
        super().__init__(email=email, name=name)
        self.id = user_id
        self.role = Role.ADMIN.value

    # --------------------------------------------------------------------------
    # FITUR 1: MANAJEMEN DATASET
    # --------------------------------------------------------------------------
    def _get_dataset_path(self):
        base = os.path.dirname(os.path.abspath(__file__))
        # Sesuaikan path ini dengan struktur folder Anda
        dataset = r"C:\Users\LENOVO\Documents\DENICO\Skripsi\Python\Dataset\combined_dataset.csv"
        return os.path.normpath(os.path.join(base,dataset))

    def view_dataset_stats(self):
        """Menampilkan statistik dan visualisasi dataset."""
        st.header("Segmen Dataset")
        csv_path = self._get_dataset_path()
        
        if not os.path.exists(csv_path):
            st.error(f"File dataset tidak ditemukan di: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        if TARGET_NAME not in df.columns:
            st.error(f"Kolom target '{TARGET_NAME}' tidak ditemukan.")
            return

        st.subheader("Informasi Ringkas")
        st.write(f"Jumlah baris: {len(df):,}")
        class_counts = df[TARGET_NAME].value_counts()
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("Jumlah tiap kelas:")
            st.dataframe(class_counts.rename('count'))
        
        with c2:
            st.subheader("Distribusi Kelas")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=class_counts.index, y=class_counts.values, palette='mako', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylabel('Count')
            st.pyplot(fig)

        st.subheader("Preview Data")
        st.dataframe(df.head(100), use_container_width=True)

    # --------------------------------------------------------------------------
    # FITUR 2: EVALUASI MODEL
    # --------------------------------------------------------------------------
    def _evaluate_and_display(self, model, encoder, ALL_FEATURES, df_features, y_true, label_prefix=""):
        # Helper internal untuk menghitung metrik
        try:
            proba = model.predict_proba(df_features)
            y_pred_idx = np.argmax(proba, axis=1)
        except Exception:
            y_pred = model.predict(df_features)
            try:
                y_pred_idx = encoder.transform(y_pred)
            except Exception:
                y_pred_idx = np.array(y_pred)

        y_true_idx = encoder.transform(y_true)

        # Metrics
        acc = accuracy_score(y_true_idx, y_pred_idx)
        prec = precision_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0)
        rec = recall_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0)
        f1 = f1_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0)

        # Display Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{label_prefix} Accuracy", f"{acc:.4f}")
        col2.metric("Precision", f"{prec:.4f}")
        col3.metric("Recall", f"{rec:.4f}")
        col4.metric("F1-Score", f"{f1:.4f}")

        # Confusion Matrix
        st.subheader(f"{label_prefix} Confusion Matrix")
        cm = confusion_matrix(y_true_idx, y_pred_idx)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap='Blues', ax=ax)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

        # ROC Curve
        st.subheader(f"{label_prefix} ROC Curve")
        try:
            y_true_bin = label_binarize(y_true_idx, classes=np.arange(len(encoder.classes_)))
            n_classes = y_true_bin.shape[1]
            fpr, tpr, roc_auc = dict(), dict(), dict()
            
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            colors = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, n_classes))) # type: ignore
            ax_roc.set_prop_cycle(colors)

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], proba[:, i]) # type: ignore
                roc_auc[i] = auc(fpr[i], tpr[i])
                ax_roc.plot(fpr[i], tpr[i], lw=2, label=f'Class {encoder.classes_[i]} (AUC={roc_auc[i]:.2f})')

            ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
        except Exception as e:
            st.warning(f"Gagal membuat ROC: {e}")

    def view_model_performance(self):
        """Menampilkan evaluasi kinerja model."""
        st.header("Segmen Model")
        st.write("Mengevaluasi model CatBoost yang sedang aktif.")

        # Instantiate Class Model & Dataset dari file Web_Prediksi_Obesity
        model_obj = Model_Prediksi()
        dataset_obj = Dataset()
        
        if not model_obj.loadmodel() or not dataset_obj.LoadDataset():
            st.error("Gagal memuat model atau dataset pendukung.")
            return

        # Data Loading untuk Test/Val
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_path, 'Model_Website', 'X dan Y')
        
        try:
            X_val = pd.read_csv(os.path.join(data_path, 'X_val.csv'))[model_obj.feature_names]
            y_val = pd.read_csv(os.path.join(data_path, 'y_val.csv'))
            X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))[model_obj.feature_names]
            y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))
            
            tab1, tab2, tab3 = st.tabs(["Evaluasi Validasi", "Evaluasi Test", "Info Model"])
            
            with tab1:
                self._evaluate_and_display(model_obj.model, model_obj.encoders[TARGET_NAME], model_obj.feature_names, X_val, y_val, "Validation")
            
            with tab2:
                self._evaluate_and_display(model_obj.model, model_obj.encoders[TARGET_NAME], model_obj.feature_names, X_test, y_test, "Test")
            
            with tab3:
                st.write("Parameter Model:")
                try:
                    st.json(model_obj.model.get_all_params()) # type: ignore
                except:
                    st.write(str(model_obj.model))

        except Exception as e:
            st.error(f"Error memuat data evaluasi: {e}")

    # --------------------------------------------------------------------------
    # FITUR 3: MANAJEMEN USER
    # --------------------------------------------------------------------------
    def view_all_users(self):
        """Menampilkan daftar user dan data input mereka."""
        st.header("Segmen User")
        
        try:
            client = get_supabase_client()
            users = client.table('User').select('*').execute().data
            inputs = client.table('DataInput').select('*').execute().data

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total User Terdaftar", len(users) if users else 0)
            with col2:
                st.metric("Total Data Input Masuk", len(inputs) if inputs else 0)

            st.markdown("---")
            if users:
                st.subheader('Daftar Akun User')
                st.dataframe(pd.DataFrame(users), use_container_width=True)
            
            if inputs:
                st.subheader('Log Data Input & Prediksi User')
                st.dataframe(pd.DataFrame(inputs), use_container_width=True)

        except Exception as e:
            st.error(f"Gagal mengambil data user: {e}")

    # --------------------------------------------------------------------------
    # UI DASHBOARD UTAMA
    # --------------------------------------------------------------------------
    def render_dashboard(self):
        """Menampilkan Dashboard Admin."""
        st.set_page_config(page_title='Admin Panel', layout='wide')

        # Sidebar Admin
        with st.sidebar:
            st.title(f"Admin Panel")
            st.info(f"Login sebagai: **{self.name}**")
            
            if st.button('🚪 Logout Admin', use_container_width=True):
                self.logout() # Memanggil method dari Parent Class User

        st.title('🔐 Halaman Administrator')
        
        # Tab Navigasi
        tabs = st.tabs(["📊 Statistik Dataset", "🤖 Evaluasi Model", "👥 Manajemen User"])

        with tabs[0]:
            self.view_dataset_stats()
        
        with tabs[1]:
            self.view_model_performance()
        
        with tabs[2]:
            self.view_all_users()

# ==============================================================================
# FUNGSI MAIN (ENTRY POINT)
# ==============================================================================
def run_admin_page():
    # 1. Cek Login (Method Static dari User)
    User.require_auth()

    # 2. Cek Role (Harus Admin)
    current_role = st.session_state.get('user_role')
    if current_role != 'Admin':
        st.warning('⛔ Akses Ditolak: Halaman ini khusus untuk Admin.')
        if st.button("Kembali ke Halaman Utama"):
            st.session_state.page = "prediksi"
            st.rerun()
        return

    # 3. Instansiasi Objek Admin
    # Mengambil data dari session state yang diset saat login
    current_admin = Admin(
        email=st.session_state.get('user'),
        name=st.session_state.get('user_name'),
        user_id=st.session_state.get('user_id')
    )

    # 4. Render Dashboard
    current_admin.render_dashboard()

if __name__ == '__main__':
    run_admin_page()