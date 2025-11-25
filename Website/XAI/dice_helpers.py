import numpy as np
import pandas as pd
import dice_ml
from dice_ml import Dice
import streamlit as st
from collections import Counter
from config import ALL_CATEGORICAL_COLS, CONTINUOUS_COLS, ORDINAL_COLS, TARGET_NAME, DECODE_MAPS
import traceback

class DiceHelper:

    def __init__(self, model=None, feature_names=None, continuous_features=None, categorical_features=None, class_labels=None):

        self.model = model
        self.feature_names = feature_names
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.class_labels = class_labels
        
        if class_labels is not None:
            self.classes_ = list(class_labels)

    def predict_proba(self, X):
        """
        Implementasi fungsi prediksi yang dibutuhkan oleh dice_ml.Model.
        Menggunakan model yang disimpan di atribut self.model.
        """
        if self.model is None:
            raise ValueError("Model belum diinisialisasi di DiceHelper.")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        X_copy = X.copy()
        
        # Preprocessing numeric
        if self.continuous_features:
            for col in self.continuous_features:
                if col in X_copy.columns:
                    X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(X_copy[col].median() or 0)
        
        # Preprocessing categorical
        if self.categorical_features:
            for col in self.categorical_features:
                if col in X_copy.columns:
                    X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(0).round().astype(int)
        
        X_copy = X_copy[self.feature_names]
        probs = self.model.predict_proba(X_copy)
        
        # Reorder probabilities if needed
        try:
            model_cls = list(getattr(self.model, 'classes_', []))
            if hasattr(self, 'class_labels') and self.class_labels is not None and model_cls:
                if model_cls != list(self.class_labels):
                    idx = [model_cls.index(c) for c in self.class_labels]
                    probs = np.array(probs)[:, idx]
        except Exception:
            pass
        
        return probs

    def _prepare_data(self, x_train_encoded, encoders, all_features_list):
        """Internal: Menyiapkan dataset pelatihan untuk DiCE."""
        df_dice = x_train_encoded.copy()
        for col in ALL_CATEGORICAL_COLS:
            if col in df_dice.columns:
                df_dice[col] = pd.to_numeric(df_dice[col], errors='coerce').fillna(0).round().astype(int)
        
        try:
            # Gunakan self.predict_proba yang sudah didefinisikan di kelas ini
            probs_all = self.predict_proba(df_dice[all_features_list])
            preds_idx = np.argmax(probs_all, axis=1)
            class_labels = np.array(encoders[TARGET_NAME].classes_)
            derived_labels = class_labels[preds_idx]
            df_dice[TARGET_NAME] = pd.Categorical(derived_labels, categories=list(encoders[TARGET_NAME].classes_))
        except Exception:
            try:
                string_predictions = self.model.predict(df_dice[all_features_list]).ravel() #type: ignore
                df_dice[TARGET_NAME] = string_predictions
            except Exception:
                df_dice[TARGET_NAME] = np.full((len(df_dice),), encoders[TARGET_NAME].classes_[0])
        
        return df_dice

    def get_dice_recommendations(self, x_train_encoded, model_obj, encoders, input_df_processed,desired_class_index, all_features_list, allow_weight_change=True):

        progress_bar = st.progress(0, text="Mencari rekomendasi perubahan...")
        
        try:
            # Persiapkan data
            progress_bar.progress(10, text="Mempersiapkan data training...")
            df_dice = self._prepare_data(x_train_encoded, encoders, self.feature_names)
            dice_continuous_features = CONTINUOUS_COLS + ORDINAL_COLS
            
            data_interface = dice_ml.Data(
                dataframe=df_dice,
                continuous_features=dice_continuous_features,
                outcome_name=TARGET_NAME
            )
            
            model_interface = dice_ml.Model(
                model=self,
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
                                
                                model_interface_num = dice_ml.Model(
                                    model=self,
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

    def decode_dataframe(self, df_dice_output, encoders, all_features_list):
        """Menerjemahkan hasil dataframe DiCE (numerik) ke format yang mudah dibaca."""
        df_decoded = df_dice_output.copy()
        for col_name, mapping in DECODE_MAPS.items():
            if col_name in df_decoded.columns:
                df_decoded[col_name] = df_decoded[col_name].astype(str).map({str(k): v for k, v in mapping.items()})
        
        for col in CONTINUOUS_COLS:
            if col in df_decoded.columns:
                df_decoded[col] = pd.to_numeric(df_decoded[col], errors='coerce').round(2)
        for col in ORDINAL_COLS:
            if col in df_decoded.columns:
                df_decoded[col] = pd.to_numeric(df_decoded[col], errors='coerce').round().astype(int)
        
        if TARGET_NAME in df_decoded.columns:
            try:
                numeric_vals = pd.to_numeric(df_decoded[TARGET_NAME], errors='coerce')
                if not numeric_vals.isna().all():
                    df_decoded[TARGET_NAME] = encoders[TARGET_NAME].inverse_transform(numeric_vals.round().astype(int))
            except: pass
        
        return df_decoded[all_features_list + [TARGET_NAME]]

    def summarize_changes(self, dice_result, input_df_processed, encoders, all_features_list):
        """Membuat ringkasan teks dari perubahan yang disarankan."""
        try:
            if not dice_result or not dice_result.cf_examples_list or dice_result.cf_examples_list[0].final_cfs_df is None:
                return []

            cf_df = dice_result.cf_examples_list[0].final_cfs_df
            cf_decoded = self.decode_dataframe(cf_df, encoders, all_features_list)
            
            q_df = input_df_processed.copy()
            if TARGET_NAME not in q_df.columns:
                q_df[TARGET_NAME] = cf_decoded[TARGET_NAME].iloc[0]
            q_decoded = self.decode_dataframe(q_df, encoders, all_features_list)
            original_data = q_decoded.iloc[0]

            feature_changes = {}
            for _, row in cf_decoded.iterrows():
                for col in all_features_list:
                    if col in [TARGET_NAME, 'Gender', 'Age', 'Height', 'family_history_with_overweight']: continue
                    if str(original_data[col]) != str(row[col]):
                        if col not in feature_changes: feature_changes[col] = []
                        feature_changes[col].append(row[col])

            summary_list = []
            sorted_features = sorted(feature_changes.items(), key=lambda x: len(x[1]), reverse=True)
            
            human_readable_map = {
                'FCVC': {1: 'Jarang', 2: 'Kadang', 3: 'Sering'},
                'NCP': {1: '1x/hari', 2: '2x/hari', 3: '3x/hari', 4: '>3x/hari'},
                'CH2O': {1: 'Kurang', 2: 'Cukup', 3: 'Banyak'},
                'FAF': {0: 'Tidak ada', 1: 'Ringan', 2: 'Sedang', 3: 'Tinggi'},
                'TUE': {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'},
                'CAEC': {'no': 'Tidak Pernah', 'Sometimes': 'Kadang', 'Frequently': 'Sering', 'Always': 'Selalu'},
                'CALC': {'no': 'Tidak Pernah', 'Sometimes': 'Kadang', 'Frequently': 'Sering', 'Always': 'Selalu'},
                'FAVC': {'no': 'Tidak', 'yes': 'Ya'}, 'SCC': {'no': 'Tidak', 'yes': 'Ya'}, 'SMOKE': {'no': 'Tidak', 'yes': 'Ya'}
            }
            readable_names = {
                'Weight': 'Berat Badan', 'FCVC': 'Makan Sayur', 'NCP': 'Makan Utama', 'CH2O': 'Minum Air',
                'FAF': 'Olahraga', 'TUE': 'Main HP', 'FAVC': 'Makan Tinggi Kalori', 'CAEC': 'Ngemil'
            }

            for feature, values in sorted_features[:5]:
                most_common_val, count = Counter(values).most_common(1)[0]
                label = readable_names.get(feature, feature)
                
                trans_val = most_common_val
                if feature in human_readable_map:
                    try:
                        k = int(float(most_common_val))
                    except:
                        k = str(most_common_val)
                    trans_val = human_readable_map[feature].get(k, most_common_val)
                
                if feature == 'Weight':
                    trans_val = f"{float(most_common_val):.1f} kg"

                summary_list.append(f"**{label}**: Disarankan menjadi **{trans_val}** (muncul di {count}/{len(cf_decoded)} opsi)")
            
            return summary_list
        except Exception as e:
            print(f"Error summary: {e}")
            return []


def get_dice_recommendations(x_train_encoded, model_obj, encoders, input_df_processed, desired_class_index, all_features_list, allow_weight_change=True):
    # Instansiasi Helper dengan Data Model untuk sesi ini
    helper = DiceHelper(
        model=model_obj,
        feature_names=all_features_list,
        continuous_features=CONTINUOUS_COLS,
        categorical_features=ALL_CATEGORICAL_COLS,
        class_labels=encoders[TARGET_NAME].classes_
    )
    # Panggil method instance
    return helper.get_dice_recommendations(x_train_encoded, model_obj, encoders, input_df_processed, desired_class_index, all_features_list, allow_weight_change)

def summarize_dice_changes(dice_result, input_df_processed, encoders, all_features_list):
    # Helper statis (tanpa model, hanya butuh utils)
    helper = DiceHelper() 
    return helper.summarize_changes(dice_result, input_df_processed, encoders, all_features_list)

def decode_dice_dataframe(df_dice_output, encoders, all_features_list):
    # Helper statis (tanpa model)
    helper = DiceHelper()
    return helper.decode_dataframe(df_dice_output, encoders, all_features_list)
