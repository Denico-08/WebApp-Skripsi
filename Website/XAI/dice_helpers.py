import numpy as np
import pandas as pd
import dice_ml
from dice_ml import Dice
import streamlit as st
from config import ALL_CATEGORICAL_COLS, CONTINUOUS_COLS, ORDINAL_COLS, TARGET_NAME, DECODE_MAPS
import traceback

class DiceCatBoostWrapper:

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
                            desired_class_index, all_features_list, allow_weight_change=False, progress_bar=None):

    # Membuat progress bar jika tidak disediakan
    if progress_bar is None:
        progress_bar = st.progress(0, text="Mencari rekomendasi perubahan...")
        
    close_progress_bar = progress_bar is None
    
    try:
        # Persiapkan data
        progress_bar.progress(10, text="Mempersiapkan data training...")
        df_dice = prepare_dice_data(x_train_encoded, model_obj, encoders, all_features_list)
        
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
                'proximity_weight': 0.2,
                'diversity_weight': 2.0
            },
            {
                'name': 'very_relaxed',
                'allow_weight': True,
                'total_cfs': 30,
                'methods': ['random', 'genetic'],
                'proximity_weight': 0.1,
                'diversity_weight': 3.0
            }
        ]
        
        for strategy_idx, strategy in enumerate(strategies):
            progress_bar.progress(
                60 + (strategy_idx * 10), 
                text=f"Mencoba strategi {strategy_idx + 1}/{len(strategies)}: {strategy['name']}..."
            )
            
            base_immutables = ['Gender', 'Age', 'Height', 'family_history_with_overweight']
            immutable_features = base_immutables if strategy['allow_weight'] else base_immutables + ['Weight']
            features_to_vary = [col for col in all_features_list if col not in immutable_features]
            
            current_weight = float(query_instance['Weight'].iloc[0])
            if strategy['name'] == 'very_relaxed':
                weight_min = max(30.0, round(current_weight - 100.0, 1))
            elif strategy['name'] == 'with_weight_relaxed':
                weight_min = max(30.0, round(current_weight - 70.0, 1))
            else:
                weight_min = max(30.0, round(current_weight - 50.0, 1))
            weight_max = current_weight
            
            permitted_range = {
                **({'Weight': [weight_min, weight_max]} if strategy['allow_weight'] else {}),
                'FCVC': [1, 3], 'NCP': [1, 4], 'CH2O': [1, 3],
                'FAF': [0, 3], 'TUE': [0, 2]
            }
            
            for method in strategy['methods']:
                try:
                    dice_explainer = Dice(data_interface, model_interface, method=method)
                    dice_result = dice_explainer.generate_counterfactuals(
                        query_instance,
                        total_CFs=strategy['total_cfs'],
                        desired_class=desired_class_label,
                        features_to_vary=features_to_vary,
                        permitted_range=permitted_range,
                        proximity_weight=strategy['proximity_weight'],
                        diversity_weight=strategy['diversity_weight']
                    )
                    
                    if (dice_result and dice_result.cf_examples_list and 
                        dice_result.cf_examples_list[0].final_cfs_df is not None and 
                        len(dice_result.cf_examples_list[0].final_cfs_df) > 0):
                        
                        num_cfs = len(dice_result.cf_examples_list[0].final_cfs_df)
                        st.success(f"✅ Berhasil! Ditemukan {num_cfs} counterfactual(s) dengan strategi '{strategy['name']}' metode '{method}'")
                        
                        if close_progress_bar:
                            progress_bar.progress(100, text="Selesai!")
                        
                        return dice_result
                        
                except Exception as e:
                    error_msg = str(e)
                    
                    if 'could not be identified' in error_msg.lower() or 'target' in error_msg.lower():
                        # ... (error handling for numeric encoding)
                        pass
            
        if close_progress_bar:
            progress_bar.progress(100, text="Selesai!")
            st.error("❌ Semua strategi pencarian gagal menemukan counterfactual yang valid.")
        
        return None
        
    except Exception as e:
        st.error(f"Error saat mencari rekomendasi DiCE: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
        if close_progress_bar and progress_bar is not None:
            progress_bar.empty()

def generate_multi_step_recommendations(x_train_encoded, model_obj, encoders, input_df_processed, 
                                        all_steps, all_features_list, allow_weight_change=False):
    
    recommendations_per_step = []
    current_input = input_df_processed.copy()
    
    total_steps = len(all_steps) - 1
    progress_bar = st.progress(0, text="Memulai rekomendasi multi-tahap...")

    for i in range(total_steps):
        current_class = all_steps[i]
        next_class = all_steps[i+1]
        
        step_progress = int(((i + 1) / total_steps) * 100)
        progress_bar.progress(step_progress, text=f"Langkah {i+1}/{total_steps}: Mencari jalan dari {current_class} ke {next_class}...")
        
        st.markdown(f"### Langkah {i+1}: Dari **{current_class.replace('_', ' ')}** ke **{next_class.replace('_', ' ')}**")
        
        try:
            desired_class_index = int(encoders[TARGET_NAME].transform([next_class])[0])
            
            dice_result_step = get_dice_recommendations(
                x_train_encoded, model_obj, encoders, current_input,
                desired_class_index, all_features_list, allow_weight_change, progress_bar=progress_bar
            )
            
            if dice_result_step and dice_result_step.cf_examples_list and dice_result_step.cf_examples_list[0].final_cfs_df is not None:
                cf_df = dice_result_step.cf_examples_list[0].final_cfs_df
                
                # Simpan hasil (DataFrame yang sudah di-decode)
                recommendations_per_step.append({
                    'from': current_class,
                    'to': next_class,
                    'original_input_processed': current_input.copy(),
                    'recommendations_df': cf_df
                })
                
                # Update input untuk langkah selanjutnya
                current_input = cf_df.iloc[[0]].drop(columns=[TARGET_NAME])
                
            else:
                st.warning(f"Tidak dapat menemukan rekomendasi untuk langkah dari {current_class} ke {next_class}. Proses berhenti.")
                break
                
        except Exception as e:
            st.error(f"Terjadi error pada langkah {i+1}: {e}")
            break
            
    progress_bar.empty()
    return recommendations_per_step


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