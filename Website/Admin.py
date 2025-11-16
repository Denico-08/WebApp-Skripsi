import streamlit as st
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from Web_Prediksi_Obesity import load_all_assets
from Login import require_auth, logout
from Connection.supabase_client import get_supabase_client
from config import TARGET_NAME, CONTINUOUS_COLS, CATEGORICAL_COLS, ORDINAL_COLS, GENDER_MAP, FAMILY_HISTORY_MAP, FAVC_MAP, SCC_MAP, SMOKE_MAP, CAEC_MAP, CALC_MAP, MTRANS_MAP


def _get_dataset_path():
    base = os.path.dirname(os.path.abspath(__file__))
    # dataset is at project root under 'Dataset'
    return os.path.normpath(os.path.join(base, '..', 'Dataset', 'combined_dataset.csv'))


def _preprocess_dataframe(df: pd.DataFrame, feature_order: list):
    # Work on a copy
    df = df.copy()

    # Ensure continuous cols numeric
    for c in CONTINUOUS_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(df[c].median())

    # Map categorical
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map(GENDER_MAP).fillna(0).astype(int)
    if 'family_history_with_overweight' in df.columns:
        df['family_history_with_overweight'] = df['family_history_with_overweight'].map(FAMILY_HISTORY_MAP).fillna(0).astype(int)
    if 'FAVC' in df.columns:
        df['FAVC'] = df['FAVC'].map(FAVC_MAP).fillna(0).astype(int)
    if 'SCC' in df.columns:
        df['SCC'] = df['SCC'].map(SCC_MAP).fillna(0).astype(int)
    if 'SMOKE' in df.columns:
        df['SMOKE'] = df['SMOKE'].map(SMOKE_MAP).fillna(0).astype(int)
    if 'CAEC' in df.columns:
        df['CAEC'] = df['CAEC'].map(CAEC_MAP).fillna(0).astype(int)
    if 'CALC' in df.columns:
        df['CALC'] = df['CALC'].map(CALC_MAP).fillna(0).astype(int)
    if 'MTRANS' in df.columns:
        df['MTRANS'] = df['MTRANS'].map(MTRANS_MAP).fillna(1).astype(int)

    # Ordinal
    ordinal_defaults = {'FCVC': 2, 'NCP': 3, 'CH2O': 2, 'FAF': 1, 'TUE': 1}
    for col in ORDINAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(ordinal_defaults.get(col, 1)).astype(int)

    # Keep only features expected by model and in correct order
    missing = [f for f in feature_order if f not in df.columns]
    if missing:
        # create missing with default 0
        for m in missing:
            df[m] = 0

    df = df[feature_order]

    return df


def _show_dataset_segment():
    st.header("Segmen Dataset")
    csv_path = _get_dataset_path()
    if not os.path.exists(csv_path):
        st.error(f"File dataset tidak ditemukan: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if TARGET_NAME not in df.columns:
        st.error(f"Kolom target '{TARGET_NAME}' tidak ditemukan di dataset.")
        return

    st.subheader("Informasi Ringkas")
    st.write(f"Jumlah baris: {len(df):,}")
    class_counts = df[TARGET_NAME].value_counts()
    st.write("Jumlah tiap kelas:")
    st.dataframe(class_counts.rename('count'))

    st.subheader("Distribusi Kelas (Bar Chart)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='mako', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    st.subheader("Preview Data")
    st.dataframe(df.head(200))


def _evaluate_and_display(model, encoder, ALL_FEATURES, df_features, y_true, label_prefix=""):
    # Predict
    try:
        proba = model.predict_proba(df_features)
        y_pred_idx = np.argmax(proba, axis=1)
    except Exception:
        # fallback to predict if predict_proba not available
        y_pred = model.predict(df_features)
        # if already labels, transform
        try:
            y_pred_idx = encoder.transform(y_pred)
        except Exception:
            y_pred_idx = np.array(y_pred)

    y_true_idx = encoder.transform(y_true)

    cm = confusion_matrix(y_true_idx, y_pred_idx)
    report = classification_report(y_true_idx, y_pred_idx, target_names=encoder.classes_, zero_division=0)

    st.subheader(f"{label_prefix} Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    st.pyplot(fig)

    st.subheader(f"{label_prefix} Classification Report")
    st.text(report)

    acc = accuracy_score(y_true_idx, y_pred_idx)
    prec = precision_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0)
    rec = recall_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0)
    f1 = f1_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0)

    st.write(f"Accuracy: {acc:.4f}")
    st.write(f"Precision (weighted): {prec:.4f}")
    st.write(f"Recall (weighted): {rec:.4f}")
    st.write(f"F1-score (weighted): {f1:.4f}")

    # --- ROC Curve ---
    st.subheader(f"{label_prefix} ROC Curve & AUC")
    try:
        y_true_bin = label_binarize(y_true_idx, classes=np.arange(len(encoder.classes_)))
        n_classes = y_true_bin.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Using a color cycle
        colors = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, n_classes)))
        ax.set_prop_cycle(colors)

        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], lw=2,
                    label='ROC of class {0} (AUC = {1:0.2f})'
                    ''.format(encoder.classes_[i], roc_auc[i]))

        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (One-vs-Rest)')
        ax.legend(loc="lower right")
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Gagal membuat ROC curve: {e}")


def _show_model_segment():
    st.header("Segmen Model")
    st.write("Menggunakan model CatBoost yang ada di `Website/Model_Website`.")

    model, encoders, ALL_FEATURES, CLASS_NAMES, x_train_encoded = load_all_assets()
    if any(a is None for a in [model, encoders, ALL_FEATURES]):
        st.error("Gagal memuat aset model. Periksa file model dan asset terkait.")
        return

    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'Model_Website', 'X dan Y')
    # Preprocess dataset to model features
    try:
        X_val = pd.read_csv(os.path.join(data_path, 'X_val.csv'))
        X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
        y_val = pd.read_csv(os.path.join(data_path, 'y_val.csv'))
        y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))
        
        if ALL_FEATURES:
            X_val = X_val[ALL_FEATURES]
            X_test = X_test[ALL_FEATURES]

    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        return
    except Exception as e:
        st.error(f"Error saat memuat dataset: {e}")
        return

    st.subheader("Evaluasi pada Validation Data")
    _evaluate_and_display(model, encoders[TARGET_NAME], ALL_FEATURES, X_val, y_val, label_prefix="Validation")# type: ignore

    st.subheader("Evaluasi pada Test Data")
    _evaluate_and_display(model, encoders[TARGET_NAME], ALL_FEATURES, X_test, y_test, label_prefix="Test")# type: ignore

    # Show model parameters
    st.subheader("Parameter Model CatBoost")
    try:
        params = model.get_params()# type: ignore
    except Exception:
        try:
            params = model.get_all_params()# type: ignore
        except Exception:
            params = str(model)

    st.write(params)

    # --- CatBoost Learning Curve: Training vs Validation ---
    st.subheader("Kurva Pembelajaran (Learning Curve)")
    try:
        evals_result = None
        try:
            evals_result = model.get_evals_result()# type: ignore
        except Exception:
            evals_result = None

        if evals_result and isinstance(evals_result, dict) and 'learn' in evals_result and evals_result['learn']:
            # determine metric name (e.g., 'Accuracy')
            metric_name = next(iter(evals_result['learn'].keys()))
            train_history = list(evals_result['learn'].get(metric_name, []))
            val_history = list(evals_result.get('validation', {}).get(metric_name, []))

            if not val_history:
                st.warning('Data histori validasi tidak ditemukan di model. Pastikan model dilatih dengan `eval_set` dan menyimpan histori.')
            else:
                # ensure same length
                common_len = min(len(train_history), len(val_history))
                if common_len == 0:
                    st.warning('Histori training/validation kosong setelah pemotongan. Tidak dapat menampilkan learning curve.')
                else:
                    it = np.arange(1, common_len + 1)
                    tr = np.array(train_history[:common_len], dtype=float)
                    va = np.array(val_history[:common_len], dtype=float)

                    # best iteration (may be None)
                    try:
                        best_iter = model.get_best_iteration()# type: ignore
                    except Exception:
                        best_iter = None

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(it, tr, label=f'Training {metric_name}', color='green', lw=2)
                    ax.plot(it, va, label=f'Validation {metric_name}', color='orange', lw=2)
                    if best_iter is not None and 0 <= best_iter < common_len:
                        ax.axvline(x=best_iter + 1, color='red', linestyle='--', label=f'Best Iteration ({best_iter + 1})')

                    ax.set_xlabel('Iterasi')
                    ax.set_ylabel(metric_name)
                    ax.set_title(f'Learning Curve: Training vs Validation {metric_name}')
                    ax.legend()
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

                    # adjust y-limits to focus on relevant range
                    min_y = min(float(np.min(tr)), float(np.min(va))) - 0.05
                    ax.set_ylim(bottom=max(0.0, min_y), top=1.01)

                    st.pyplot(fig)

                    # show metrics at best iteration when available
                    if best_iter is not None and 0 <= best_iter < common_len:
                        st.subheader('Skor Terbaik')
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric('Akurasi Training Terbaik', f'{tr[best_iter]:.4f}')
                        with col2:
                            st.metric('Akurasi Validasi Terbaik', f'{va[best_iter]:.4f}')
                        
                        st.subheader("Analisis Learning Curve")
                        st.markdown(f"""
                        Grafik di atas menunjukkan performa model pada data training dan validasi selama proses training.
                        - **Garis Hijau (Training {metric_name})**: Menunjukkan seberapa baik model mempelajari data training.
                        - **Garis Oranye (Validation {metric_name})**: Menunjukkan seberapa baik model dapat menggeneralisasi pengetahuannya ke data baru (data validasi).

                        **Mengapa Iterasi Terbaik di {best_iter + 1}?**

                        Iterasi terbaik ({best_iter + 1}) adalah titik di mana model mencapai **skor validasi tertinggi ({va[best_iter]:.4f})**. Ini dianggap sebagai titik optimal sebelum model mulai *overfitting*.

                        - **Sebelum Iterasi {best_iter + 1}**: Kedua kurva (training dan validasi) sama-sama naik, menunjukkan model sedang dalam proses belajar yang sehat.
                        - **Setelah Iterasi {best_iter + 1}**: Perhatikan bahwa kurva training kemungkinan akan terus naik (atau mendatar di nilai tinggi), sementara kurva validasi mulai **stagnan atau bahkan menurun**. Ini adalah tanda klasik **overfitting**. Artinya, model mulai "menghafal" data training alih-alih belajar pola umum yang bisa diterapkan pada data baru.

                        Dengan memilih model pada iterasi {best_iter + 1}, kita mendapatkan versi model dengan kemampuan generalisasi terbaik untuk data yang belum pernah dilihat sebelumnya.
                        """)
        else:
            st.info('Tidak ada histori training/validation pada model untuk menampilkan learning curve CatBoost.')
    except Exception as e:
        st.error(f'Gagal membuat learning curve CatBoost: {e}')

    
                
def _show_user_segment():
    st.header("Segmen User")
    st.write("Menampilkan daftar user yang sudah terdaftar dan data input/prediksi mereka (jika tersedia).")

    try:
        supabase = get_supabase_client()
        users_resp = supabase.table('User').select('*').execute()
        inputs_resp = supabase.table('DataInput').select('*').execute()

        users = users_resp.data if hasattr(users_resp, 'data') else (users_resp.get('data') if isinstance(users_resp, dict) else [])
        inputs = inputs_resp.data if hasattr(inputs_resp, 'data') else (inputs_resp.get('data') if isinstance(inputs_resp, dict) else [])

        if users:
            st.subheader('Daftar User')
            df_users = pd.DataFrame(users)
            st.dataframe(df_users)
        else:
            st.info('Tidak ada data user ditemukan di Supabase.')

        if inputs:
            st.subheader('Data Input & Prediksi')
            df_inputs = pd.DataFrame(inputs)
            st.dataframe(df_inputs)
        else:
            st.info('Tidak ada data input/prediksi ditemukan di Supabase.')

    except Exception as e:
        st.error(f"Gagal mengambil data user dari Supabase: {e}")


def admin_page_():
    require_auth()

    # Ensure only admin can access
    if st.session_state.get('user_role') != 'Admin':
        st.warning('Akses ditolak: Anda bukan admin.')
        return

    st.set_page_config(page_title='Admin Panel', layout='wide')

    with st.sidebar:
        st.title(f"Admin: {st.session_state.get('user_name')}")
        st.write(f"Email: {st.session_state.get('user')}")
        if st.button('🚪 Logout'):
            logout()
            st.session_state.page = 'login'
            st.rerun()

    st.title('Halaman Admin')
    st.markdown('Tiga segmen: `Dataset`, `Model`, `User`.')

    tab = st.selectbox('Pilih Segmen', ['Dataset', 'Model', 'User'])

    if tab == 'Dataset':
        _show_dataset_segment()
    elif tab == 'Model':
        _show_model_segment()
    else:
        _show_user_segment()


if __name__ == '__main__':
    admin_page_()