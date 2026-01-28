import streamlit as st
import pandas as pd
import numpy as np
import joblib

import os
from Role import Role
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Plotly for interactive charts
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Import Class User (Parent Class)
from User import User

from Web_Prediksi_Obesity import Dataset, Model_Prediksi 
from Connection.db_client import get_db_connection
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
        dataset = r"C:\Users\LENOVO\Documents\DENICO\Skripsi\Python\Dataset\combined_df_trimmed.csv"
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
            fig = px.bar(
                class_counts,
                x=class_counts.index, 
                y=class_counts.values,
                labels={'x': 'Kelas Obesitas', 'y': 'Jumlah'},
                color=class_counts.index,
                text_auto=True
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="Kelas",
                yaxis_title="Jumlah",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Tampilkan distribusi Train, Validation, dan Test
        st.subheader("Distribusi Data Train, Validation, dan Test")
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_path, 'Model_Website', 'X dan Y')
            
            # Load data
            y_train_smote_path = os.path.join(base_path, 'Model_Website', 'Y_train_smote.pkl')
            y_val_path = os.path.join(data_path, 'y_val.csv')
            y_test_path = os.path.join(data_path, 'y_test.csv')
            
            datasets_available = {}
            datasets_counts = {}
            
            # Load Train SMOTE
            if os.path.exists(y_train_smote_path):
                try:
                    y_train_smote = joblib.load(y_train_smote_path)
                    # Decode jika perlu
                    if hasattr(y_train_smote, 'dtype') and y_train_smote.dtype == 'int64':
                        # Asumsikan y_class_names sudah ada dari loading sebelumnya
                        if 'y_class_names' in locals():
                            y_train_decoded = [y_class_names[int(y)] for y in y_train_smote] #type: ignore
                            train_counts = pd.Series(y_train_decoded).value_counts()
                        else:
                            train_counts = pd.Series(y_train_smote).value_counts()
                    else:
                        train_counts = pd.Series(y_train_smote).value_counts()
                    datasets_available['Train (SMOTE)'] = train_counts
                except Exception as e:
                    st.warning(f"Tidak dapat memuat y_train_smote: {e}")
            
            # Load Validation
            if os.path.exists(y_val_path):
                try:
                    y_val_df = pd.read_csv(y_val_path)
                    y_val_col = y_val_df.columns[0]
                    val_counts = y_val_df[y_val_col].value_counts()
                    datasets_available['Validation'] = val_counts
                except Exception as e:
                    st.warning(f"Tidak dapat memuat y_val: {e}")
            
            # Load Test
            if os.path.exists(y_test_path):
                try:
                    y_test_df = pd.read_csv(y_test_path)
                    y_test_col = y_test_df.columns[0]
                    test_counts = y_test_df[y_test_col].value_counts()
                    datasets_available['Test'] = test_counts
                except Exception as e:
                    st.warning(f"Tidak dapat memuat y_test: {e}")
            
            if datasets_available:
                # Buat kolom untuk setiap dataset
                num_cols = len(datasets_available)
                cols = st.columns(num_cols)
                
                for idx, (dataset_name, counts) in enumerate(datasets_available.items()):
                    with cols[idx]:
                        st.write(f"**{dataset_name}**")
                        fig_dist = px.bar(
                            counts,
                            x=counts.index,
                            y=counts.values,
                            labels={'x': 'Kelas', 'y': 'Jumlah'},
                            color=counts.index,
                            text_auto=True
                        )
                        fig_dist.update_layout(
                            showlegend=False,
                            xaxis_title="Kelas",
                            yaxis_title="Jumlah",
                            xaxis_tickangle=-45,
                            height=400
                        )
                        st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("Data Train, Validation, dan Test tidak tersedia.")
        except Exception as e:
            st.warning(f"Tidak dapat memuat distribusi data: {e}")

        st.subheader("Preview Data")
        st.dataframe(df.head(100), use_container_width=True)

    # --------------------------------------------------------------------------
    # FITUR 2: EVALUASI MODEL
    # --------------------------------------------------------------------------
    def _evaluate_and_display(self, model, encoder, ALL_FEATURES, df_features, y_true, label_prefix=""):
        from sklearn.preprocessing import LabelEncoder
        from itertools import cycle

        # encoder sekarang adalah LabelEncoder yang sudah di-fit (dari y_label_encoder.pkl)
        le = encoder
        
        # Validasi
        if not hasattr(le, 'classes_'):
            st.error("Encoder tidak valid. Harus memiliki atribut 'classes_'")
            return

        try:
            proba = model.predict_proba(df_features)
            y_pred_idx = np.argmax(proba, axis=1)
        except Exception:
            y_pred = model.predict(df_features)
            try:
                y_pred_idx = le.transform(np.array(y_pred).ravel())
            except Exception:
                y_pred_idx = np.array(y_pred).ravel()

        # Transform y_true (dari CSV) menjadi index numerik
        y_true_idx = le.transform(np.array(y_true).ravel())

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

        # Confusion Matrix (dengan Plotly)
        st.subheader(f"{label_prefix} Confusion Matrix")
        cm = confusion_matrix(y_true_idx, y_pred_idx, labels=np.arange(len(le.classes_)))
        
        x = le.classes_.tolist()
        y = le.classes_.tolist()
        z_text = [[str(y) for y in x] for x in cm]
        
        fig_cm = ff.create_annotated_heatmap(
            np.flipud(cm), # flip untuk orientasi yg benar
            x=x, 
            y=y[::-1], # flip untuk orientasi yg benar
            annotation_text=np.flipud(z_text),
            colorscale='Blues'
        )
        fig_cm.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True)
        )
        st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})


        # ROC Curve & AUC
        try:
            y_true_bin = label_binarize(y_true_idx, classes=np.arange(len(le.classes_)))
            n_classes = y_true_bin.shape[1]
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], proba[:, i]) #type: ignore
                roc_auc[i] = auc(fpr[i], tpr[i])

            # --- Tampilkan AUC Scores ---
            st.subheader(f"{label_prefix} AUC Score (One-vs-Rest)")
            
            # Buat dataframe untuk menampilkan skor AUC
            auc_data = {'Class': [le.classes_[i] for i in range(n_classes)],
                        'AUC Score': [roc_auc[i] for i in range(n_classes)]}
            auc_df = pd.DataFrame(auc_data)

            # Tambahkan rata-rata macro
            if n_classes > 0:
                macro_avg_auc = np.mean(list(roc_auc.values()))
                # Create a new DataFrame for the average row
                avg_row = pd.DataFrame([{'Class': 'Macro Average', 'AUC Score': macro_avg_auc}])
                # Concatenate the original DataFrame with the average row
                auc_df = pd.concat([auc_df, avg_row], ignore_index=True)

            # Tampilkan menggunakan st.dataframe dengan format
            st.dataframe(
                auc_df.style.format({'AUC Score': '{:.4f}'}),
                use_container_width=True,
                hide_index=True
            )
            # --- Tampilkan ROC Curves ---
            st.subheader(f"{label_prefix} ROC Curve (One-vs-Rest)")
            # Buat plot terpisah untuk setiap kelas
            for i in range(n_classes):
                class_name = le.classes_[i]
                fig_roc_class = go.Figure()

                fig_roc_class.add_trace(go.Scatter(
                    x=fpr[i], y=tpr[i],
                    mode='lines',
                    line=dict(width=3),
                    name=f'AUC = {roc_auc[i]:.2f}'
                ))

                fig_roc_class.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

                fig_roc_class.update_layout(
                    title=f'ROC Curve for: <b>{class_name}</b>',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                    xaxis=dict(fixedrange=True),
                    yaxis=dict(fixedrange=True)
                )
                st.plotly_chart(fig_roc_class, use_container_width=True, config={'displayModeBar': False})

        except Exception as e:
            st.warning(f"Gagal membuat ROC curve atau AUC Score: {e}")
            import traceback
            st.text(traceback.format_exc())

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
                st.code("""
                    cat_features=categorical_features,
                    loss_function='MultiClass',
                    eval_metric='Accuracy',
                    learning_rate=0.05,
                    random_state=42,
                    iterations=1000,
                    depth=6,
                    l2_leaf_reg=5
                """, language='python')

        except Exception as e:
            st.error(f"Error memuat data evaluasi: {e}")

    # --------------------------------------------------------------------------
    # FITUR 3: MANAJEMEN USER
    # --------------------------------------------------------------------------
    def view_all_users(self):
        """Menampilkan daftar user dan data input mereka."""
        st.header("Segmen User")
        conn = get_db_connection()
        try:
            users = pd.read_sql('SELECT * FROM "User"', conn)
            inputs = pd.read_sql('SELECT * FROM "DataInput"', conn)
            # ... sisa kode visualisasi dataframe sama ...
        except Exception as e:
            st.error(f"Gagal: {e}")

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

        st.title('Halaman Administrator')
        
        # Tab Navigasi
        tabs = st.tabs(["Statistik Dataset", "Evaluasi Model", "Manajemen User"])

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
    
    current_admin = Admin(
        email=st.session_state.get('user'),
        name=st.session_state.get('user_name'),
        user_id=st.session_state.get('user_id')
    )

    # 4. Render Dashboard
    current_admin.render_dashboard()

if __name__ == '__main__':
    run_admin_page()