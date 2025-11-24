import numpy as np
import pandas as pd
import dice_ml
from dice_ml import Dice
import streamlit as st
from collections import Counter
from config import ALL_CATEGORICAL_COLS, CONTINUOUS_COLS, ORDINAL_COLS, TARGET_NAME, DECODE_MAPS
import traceback

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
                            desired_class_index, all_features_list, allow_weight_change=True):

    progress_bar = st.progress(0, text="Mencari rekomendasi perubahan...")
    
    try:
        # Persiapkan data
        progress_bar.progress(10, text="Mempersiapkan data training...")
        df_dice = prepare_dice_data(x_train_encoded, model_obj, encoders, all_features_list)
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
        
            # Tentukan fitur yang tidak boleh diubah
            base_immutables = ['Gender', 'Age', 'Height', 'family_history_with_overweight']
            immutable_features = base_immutables if strategy['allow_weight'] else base_immutables + ['Weight']
            features_to_vary = [col for col in all_features_list if col not in immutable_features]
            
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
            
            # Coba berbagai metode dalam strategi ini
            for method_idx, method in enumerate(strategy['methods']):
                try:
                    
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
                        st.write(f"❌ Metode {method} tidak menghasilkan counterfactual")
                        
                except Exception as e:
                    error_msg = str(e)                    
                    # Jika error terkait target class, coba dengan numeric encoding
                    if 'could not be identified' in error_msg.lower() or 'target' in error_msg.lower():
                        try:
                            
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

def summarize_dice_changes(dice_result, input_df_processed, encoders, all_features_list):
    """
    Menganalisis hasil rekomendasi DiCE dan mengembalikan daftar fitur
    yang paling sering disarankan untuk diubah, beserta NILAI SARAN PALING UMUM
    yang sudah diterjemahkan ke teks bahasa Indonesia yang mudah dipahami.
    """
    try:
        if not dice_result or not dice_result.cf_examples_list or dice_result.cf_examples_list[0].final_cfs_df is None:
            return []

        # 1. Decode hasil rekomendasi (Counterfactuals)
        cf_df = dice_result.cf_examples_list[0].final_cfs_df
        cf_decoded = decode_dice_dataframe(cf_df, encoders, all_features_list)
        
        # 2. Decode data asli user
        q_df = input_df_processed.copy()
        if TARGET_NAME not in q_df.columns:
            q_df[TARGET_NAME] = cf_decoded[TARGET_NAME].iloc[0] # Dummy target
            
        q_decoded = decode_dice_dataframe(q_df, encoders, all_features_list)
        original_data = q_decoded.iloc[0]

        feature_changes = {}

        # 3. Kumpulkan semua perubahan (simpan nilai barunya)
        for _, row in cf_decoded.iterrows():
            for col in all_features_list:
                if col == TARGET_NAME or col in ['Gender', 'Age', 'Height', 'family_history_with_overweight']:
                    continue
                
                val_original = original_data[col]
                val_new = row[col]
                
                # Bandingkan nilai
                if str(val_original) != str(val_new):
                    if col not in feature_changes:
                        feature_changes[col] = []
                    feature_changes[col].append(val_new)

        # 4. Analisis Statistik (Cari nilai terbanyak/modus untuk setiap fitur)
        summary_list = []
        
        # Urutkan fitur berdasarkan total frekuensi perubahan (terbanyak ke sedikit)
        sorted_features = sorted(feature_changes.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Mapping nama fitur ke bahasa Indonesia
        readable_names = {
            'Weight': 'Berat Badan',
            'FCVC': 'Konsumsi Sayur',
            'NCP': 'Jadwal Makan Utama',
            'CH2O': 'Minum Air Putih',
            'FAF': 'Aktivitas Fisik',
            'TUE': 'Penggunaan Gawai',
            'MTRANS': 'Transportasi',
            'FAVC': 'Konsumsi Makanan Tinggi Kalori',
            'CAEC': 'Ngemil di Antara Makan',
            'SMOKE': 'Merokok',
            'SCC': 'Memantau Kalori',
            'CALC': 'Konsumsi Alkohol'
        }

        # Mapping nilai numerik/kode ke deskripsi teks
        human_readable_map = {
            'FCVC': {1: 'Jarang (Tidak Pernah)', 2: 'Kadang-kadang', 3: 'Sering (Setiap Makan)'},
            'NCP': {1: '1x sehari', 2: '2x sehari', 3: '3x sehari', 4: 'Lebih dari 3x sehari'},
            'CH2O': {1: 'Kurang (< 1 Liter)', 2: 'Cukup (1-2 Liter)', 3: 'Banyak (> 2 Liter)'},
            'FAF': {0: 'Tidak ada (< 15 mnt)', 1: 'Ringan (15-30 mnt)', 2: 'Sedang (30-60 mnt)', 3: 'Rutin/Tinggi (> 60 mnt)'},
            'TUE': {0: 'Rendah (0-1 jam)', 1: 'Sedang (1-2 jam)', 2: 'Tinggi (> 2 jam)'},
            'CAEC': {'no': 'Tidak Pernah', 'Sometimes': 'Kadang-kadang', 'Frequently': 'Sering', 'Always': 'Selalu'},
            'CALC': {'no': 'Tidak Pernah', 'Sometimes': 'Kadang-kadang', 'Frequently': 'Sering', 'Always': 'Selalu'},
            'FAVC': {'no': 'Tidak', 'yes': 'Ya'},
            'SCC': {'no': 'Tidak', 'yes': 'Ya'},
            'SMOKE': {'no': 'Tidak', 'yes': 'Ya'}
        }

        for feature, values in sorted_features[:5]: # Top 5 fitur
            count_total = len(values)
            total_recommendations = len(cf_decoded)
            
            # Cari nilai yang paling sering disarankan (Modus)
            value_counts = Counter(values)
            most_common_val, most_common_count = value_counts.most_common(1)[0]
            
            feature_label = readable_names.get(feature, feature)
            
            # Terjemahkan nilai jika ada di map
            translated_val = most_common_val
            if feature in human_readable_map:
                # Coba parsing ke int jika memungkinkan (karena key di map bisa int)
                try:
                    key_val = int(float(most_common_val))
                except:
                    key_val = str(most_common_val)
                
                translated_val = human_readable_map[feature].get(key_val, most_common_val)
            
            # Tambahan satuan untuk Berat Badan
            if feature == 'Weight':
                translated_val = f"{float(most_common_val):.1f} kg"

            # Format pesan: Menampilkan nilai spesifik yang paling banyak disarankan
            msg = f"**{feature_label}**: Disarankan menjadi **{translated_val}** (muncul di {most_common_count} dari {total_recommendations} opsi)"
            summary_list.append(msg)
            
        return summary_list

    except Exception as e:
        print(f"Error generating summary: {e}")
        return []