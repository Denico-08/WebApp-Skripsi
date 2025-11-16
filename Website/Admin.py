import streamlit as st
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
<<<<<<< HEAD
from catboost import CatBoostClassifier
=======
>>>>>>> 6d5bbb45cea0b2c96f7eb716b2495b3b1ef8f797
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


def _show_model_segment():
    st.header("Segmen Model")
    st.write("Menggunakan model CatBoost yang ada di `Website/Model_Website`.")

    model, encoders, ALL_FEATURES, CLASS_NAMES, x_train_encoded = load_all_assets()
    if any(a is None for a in [model, encoders, ALL_FEATURES]):
        st.error("Gagal memuat aset model. Periksa file model dan asset terkait.")
        return

    csv_path = _get_dataset_path()
    df = pd.read_csv(csv_path)

    # Preprocess dataset to model features
    df_features_all = df.copy()
    if TARGET_NAME not in df_features_all.columns:
        st.error(f"Kolom target '{TARGET_NAME}' tidak ditemukan di dataset.")
        return

    y = df_features_all[TARGET_NAME].astype(str)
    X = _preprocess_dataframe(df_features_all, ALL_FEATURES) # type: ignore

    # Stratified splits: train 70 / val 15 / test 15
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    except Exception:
        # fallback to simple split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, y_val = X_test.copy(), y_test.copy()

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

    # --- Learning curve for SVC (train vs val and train vs test)
    st.subheader("Learning Curve SVC (Train vs Val & Train vs Test)")
    st.write("Bandingkan akurasi training dengan validation dan training dengan test menggunakan SVC.")
    if st.button("Hitung Learning Curve SVC"):
        with st.spinner("Menghitung learning curve SVC — ini mungkin memakan waktu beberapa menit..."):
            try:
                fractions = [0.1, 0.3, 0.5, 0.7, 1.0]
                train_scores = []
                val_scores = []
                test_scores = []

                # SVC parameters (moderate size to avoid extremely long runs)
                svc_params = {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale',
                    'max_iter': 10000,
                }

                # limit samples to speed up
                max_samples = min(len(X_train), 2000)

                for frac in fractions:
                    n = max(10, int(frac * max_samples))
                    X_sub = X_train.sample(n, random_state=42)
                    y_sub = y_train.loc[X_sub.index]

                    # Scale continuous columns using scaler fit on training subset
                    scaler = StandardScaler()
                    cont_cols = [c for c in CONTINUOUS_COLS if c in X_sub.columns]
                    if cont_cols:
                        scaler.fit(X_sub[cont_cols])
                        X_sub_scaled = X_sub.copy()
                        X_sub_scaled[cont_cols] = scaler.transform(X_sub[cont_cols])
                        X_val_scaled = X_val.copy()
                        X_val_scaled[cont_cols] = scaler.transform(X_val[cont_cols])
                        X_test_scaled = X_test.copy()
                        X_test_scaled[cont_cols] = scaler.transform(X_test[cont_cols])
                    else:
                        X_sub_scaled = X_sub
                        X_val_scaled = X_val
                        X_test_scaled = X_test

                    svc = SVC(**svc_params)
                    svc.fit(X_sub_scaled, y_sub)

                    train_pred = svc.predict(X_sub_scaled)
                    val_pred = svc.predict(X_val_scaled)
                    test_pred = svc.predict(X_test_scaled)

                    train_scores.append(accuracy_score(y_sub, train_pred))
                    val_scores.append(accuracy_score(y_val, val_pred))
                    test_scores.append(accuracy_score(y_test, test_pred))

                # Plot: train vs val
                fig1, ax1 = plt.subplots()
                ax1.plot(fractions, train_scores, label='Train Accuracy', marker='o')
                ax1.plot(fractions, val_scores, label='Validation Accuracy', marker='o')
                ax1.set_xlabel('Fraction of Training Data Used')
                ax1.set_ylabel('Accuracy')
                ax1.set_title('SVC Learning Curve: Train vs Validation')
                ax1.legend()
                st.pyplot(fig1)

                # Plot: train vs test
                fig2, ax2 = plt.subplots()
                ax2.plot(fractions, train_scores, label='Train Accuracy', marker='o')
                ax2.plot(fractions, test_scores, label='Test Accuracy', marker='o')
                ax2.set_xlabel('Fraction of Training Data Used')
                ax2.set_ylabel('Accuracy')
                ax2.set_title('SVC Learning Curve: Train vs Test')
                ax2.legend()
                st.pyplot(fig2)

                st.subheader('SVC Parameters')
                st.write(svc_params)

            except Exception as e:
                st.error(f"Gagal menghitung learning curve SVC: {e}")

<<<<<<< HEAD
    # --- Learning curve for CatBoost (train vs val and train vs test)
    st.subheader("Learning Curve CatBoost (Train vs Val & Train vs Test)")
    st.write("Melatih CatBoost pada beberapa fraksi data training untuk melihat tren akurasi.")
    if st.button("Hitung Learning Curve CatBoost"):
        with st.spinner("Menghitung learning curve CatBoost — ini mungkin memakan waktu cukup lama..."):
            try:
                from catboost import CatBoostClassifier

                fractions = [0.1, 0.3, 0.5, 0.7, 1.0]
                train_scores = []
                val_scores = []
                test_scores = []

                # Parameters per user's request (depth set to 8, learning_rate to 0.04)
                cb_params = {
                    'iterations': 1000,
                    'learning_rate': 0.04,
                    'depth': 8,
                    'cat_features': [c for c in (ALL_FEATURES or []) if c in ((CATEGORICAL_COLS or []) + (ORDINAL_COLS or []))],
                    'use_best_model': True,
                    'eval_metric': 'Accuracy',
                    'od_wait': 50,
                    'od_type': 'Iter',
                    'loss_function': 'MultiClass',
                    'random_seed': 42,
                    'verbose': 100
                }

                max_samples = min(len(X_train), 2000)

                for frac in fractions:
                    n = max(10, int(frac * max_samples))
                    X_sub = X_train.sample(n, random_state=42)
                    y_sub = y_train.loc[X_sub.index]

                    cb = CatBoostClassifier(**cb_params)
                    cb.fit(X_sub, y_sub)

                    train_pred = cb.predict(X_sub)
                    val_pred = cb.predict(X_val)
                    test_pred = cb.predict(X_test)

                    train_scores.append(accuracy_score(y_sub, train_pred))
                    val_scores.append(accuracy_score(y_val, val_pred))
                    test_scores.append(accuracy_score(y_test, test_pred))

                # Plot: train vs val
                fig_cb1, ax_cb1 = plt.subplots()
                ax_cb1.plot(fractions, train_scores, label='Train Accuracy', marker='o')
                ax_cb1.plot(fractions, val_scores, label='Validation Accuracy', marker='o')
                ax_cb1.set_xlabel('Fraction of Training Data Used')
                ax_cb1.set_ylabel('Accuracy')
                ax_cb1.set_title('CatBoost Learning Curve: Train vs Validation')
                ax_cb1.legend()
                st.pyplot(fig_cb1)

                # Plot: train vs test
                fig_cb2, ax_cb2 = plt.subplots()
                ax_cb2.plot(fractions, train_scores, label='Train Accuracy', marker='o')
                ax_cb2.plot(fractions, test_scores, label='Test Accuracy', marker='o')
                ax_cb2.set_xlabel('Fraction of Training Data Used')
                ax_cb2.set_ylabel('Accuracy')
                ax_cb2.set_title('CatBoost Learning Curve: Train vs Test')
                ax_cb2.legend()
                st.pyplot(fig_cb2)

                st.subheader('CatBoost Parameters')
                st.write(cb_params)

            except Exception as e:
                st.error(f"Gagal menghitung learning curve CatBoost: {e}")

=======
>>>>>>> 6d5bbb45cea0b2c96f7eb716b2495b3b1ef8f797

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
