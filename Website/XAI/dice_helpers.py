import numpy as np
import pandas as pd
import dice_ml
from dice_ml import Dice
import streamlit as st
from collections import Counter
from config import (
    ALL_CATEGORICAL_COLS, CONTINUOUS_COLS, ORDINAL_COLS, TARGET_NAME, 
    DECODE_MAPS, ACTION_VERBS, OBESITY_GAIN_HIERARCHY, OBESITY_REDUCTION_HIERARCHY
)
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

    def get_next_target_class(self, current_class, class_names):
        """
        Menentukan kelas target berikutnya dalam hirarki obesitas (untuk rekomendasi).
        """
        # Tentukan hirarki yang akan digunakan
        if current_class == 'Insufficient_Weight':
            active_hierarchy = [c for c in OBESITY_GAIN_HIERARCHY if c in class_names]
        elif current_class in OBESITY_REDUCTION_HIERARCHY:
            active_hierarchy = [c for c in OBESITY_REDUCTION_HIERARCHY if c in class_names]
        else:
            # Jika kelas saat ini adalah Normal_Weight atau tidak ada di hirarki
            return current_class, True, [current_class]

        # Temukan posisi saat ini di hirarki aktif
        try:
            current_index = active_hierarchy.index(current_class)
        except ValueError:
            return current_class, True, [current_class] # Fallback jika tidak ditemukan

        # Target selalu 'Normal_Weight'
        NORMAL_WEIGHT_CLASS = 'Normal_Weight'
        
        # Periksa apakah sudah mencapai atau melewati target
        if current_class == NORMAL_WEIGHT_CLASS:
            return current_class, True, [current_class]
        
        # Tentukan path dan target berikutnya
        if current_index < len(active_hierarchy) - 1:
            next_class = active_hierarchy[current_index + 1]
            
            # Buat path lengkap dari posisi saat ini ke Normal_Weight
            try:
                normal_index_in_hierarchy = active_hierarchy.index(NORMAL_WEIGHT_CLASS)
                all_steps = active_hierarchy[current_index : normal_index_in_hierarchy + 1]
            except ValueError:
                all_steps = [current_class, next_class]

            is_final = (next_class == NORMAL_WEIGHT_CLASS)
            
            return next_class, is_final, all_steps
        else:
            # Sudah di ujung hirarki (seharusnya Normal_Weight)
            return current_class, True, [current_class]

    def predict_proba(self, X):
        """Wrapper prediksi untuk DiCE dengan handling yang lebih robust"""
        if self.model is None:
            raise ValueError("Model belum diinisialisasi di DiceHelper.")

        # Konversi ke DataFrame jika perlu
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Pastikan urutan kolom
        X_ordered = X[self.feature_names].copy()
        
        # PENTING: Pastikan tipe data konsisten
        for col in self.feature_names: #type: ignore
            if col in CONTINUOUS_COLS:
                X_ordered[col] = pd.to_numeric(X_ordered[col], errors='coerce').fillna(0).astype(float)
            else:
                X_ordered[col] = pd.to_numeric(X_ordered[col], errors='coerce').fillna(0).astype(int)
        
        # Prediksi
        probs = self.model.predict_proba(X_ordered)
        
        # Handle class ordering jika perlu
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
        """Menyiapkan dataset untuk DiCE dengan validasi yang lebih ketat"""
        df_dice = x_train_encoded.copy()
        
        # STEP 0: ENCODE KOLOM YANG MASIH STRING (JIKA ADA)
        # Ini handle kasus di mana X_train_smote.pkl belum di-encode
        from config import (GENDER_MAP, FAMILY_HISTORY_MAP, FAVC_MAP, SCC_MAP, 
                           SMOKE_MAP, CAEC_MAP, CALC_MAP, MTRANS_MAP)
        
        # Cek dan encode manual jika perlu
        encoding_maps = {
            'Gender': GENDER_MAP,
            'family_history_with_overweight': FAMILY_HISTORY_MAP,
            'FAVC': FAVC_MAP,
            'SCC': SCC_MAP,
            'SMOKE': SMOKE_MAP,
            'CAEC': CAEC_MAP,
            'CALC': CALC_MAP,
            'MTRANS': MTRANS_MAP
        }
        
        for col, mapping in encoding_maps.items():
            if col in df_dice.columns:
                # Cek apakah kolom masih string
                if df_dice[col].dtype == 'object' or df_dice[col].dtype == 'string':
                    df_dice[col] = df_dice[col].map(mapping)
        
        # Pastikan semua kolom kategorikal adalah integer
        for col in ALL_CATEGORICAL_COLS:
            if col in df_dice.columns:
                df_dice[col] = pd.to_numeric(df_dice[col], errors='coerce').fillna(0).round().astype(int)
        
        # Pastikan kolom continuous adalah float
        for col in CONTINUOUS_COLS:
            if col in df_dice.columns:
                df_dice[col] = pd.to_numeric(df_dice[col], errors='coerce').fillna(0).astype(float)
        
        
        # Template row untuk membuat dummy data
        template_row = df_dice.iloc[0].copy()
        dummy_rows = []
        
        # Mapping lengkap untuk setiap fitur kategorikal
        categorical_ranges = {
            'Gender': [0, 1],
            'family_history_with_overweight': [0, 1],
            'FAVC': [0, 1],
            'SCC': [0, 1],
            'SMOKE': [0, 1],
            'CAEC': [0, 1, 2, 3],
            'CALC': [0, 1, 2, 3],
            'MTRANS': [0, 1, 2, 3, 4],
            'FCVC': [1, 2, 3],
            'NCP': [1, 2, 3, 4],
            'CH2O': [1, 2, 3],
            'FAF': [0, 1, 2, 3],
            'TUE': [0, 1, 2]
        }
        
        # Cek dan tambahkan missing values
        missing_values = {}
        for col, valid_range in categorical_ranges.items():
            if col in df_dice.columns:
                existing_vals = set(df_dice[col].unique())
                missing_vals = [v for v in valid_range if v not in existing_vals]
                
                if missing_vals:
                    missing_values[col] = missing_vals
                    # Tambahkan dummy row untuk setiap missing value
                    for val in missing_vals:
                        dummy_row = template_row.copy()
                        dummy_row[col] = val
                        dummy_rows.append(dummy_row)
        
        # Report missing values
        if missing_values:
            st.warning(f"""
            ⚠️ **Missing categorical values detected:**
            {chr(10).join([f'- **{col}**: missing {vals}' for col, vals in missing_values.items()])}
            
            Adding {len(dummy_rows)} dummy rows to ensure full coverage...
            """)
        
        dummy_df = pd.DataFrame(dummy_rows)
        df_dice = pd.concat([df_dice, dummy_df], ignore_index=True)
        # Generate target labels
        try:
            probs_all = self.predict_proba(df_dice[all_features_list])
            preds_idx = np.argmax(probs_all, axis=1)
            class_labels = np.array(encoders[TARGET_NAME].classes_)
            derived_labels = class_labels[preds_idx]
            df_dice[TARGET_NAME] = derived_labels
        except Exception as e:
            st.warning(f"Gagal generate target otomatis: {e}")
            # Fallback: gunakan prediksi string langsung
            try:
                string_predictions = self.model.predict(df_dice[all_features_list]).ravel() # type: ignore
                df_dice[TARGET_NAME] = string_predictions
            except Exception:
                # Ultimate fallback
                df_dice[TARGET_NAME] = encoders[TARGET_NAME].classes_[0]
        
        return df_dice

    def get_dice_recommendations(self, x_train_encoded, model_obj, encoders, input_df_processed, 
                                 desired_class_index, all_features_list, allow_weight_change=True):
        """Generate counterfactual recommendations dengan strategi bertahap"""
        
        progress_bar = st.progress(0, text="Memulai pencarian counterfactual...")
        
        try:
            # === STEP 1: Persiapan Data ===
            progress_bar.progress(10, text="Mempersiapkan data training...")
            
            # Cek sample dari training data
            sample_check = x_train_encoded.head(5) if hasattr(x_train_encoded, 'head') else None
            
            needs_encoding = False
            if sample_check is not None:
                # Cek apakah ada nilai string di kolom yang seharusnya numeric
                for col in ['Gender', 'FAVC', 'SMOKE']:
                    if col in sample_check.columns:
                        sample_vals = sample_check[col].head(3).tolist()
                        if any(isinstance(v, str) for v in sample_vals):
                            needs_encoding = True
                            break
            
            df_dice = self._prepare_data(x_train_encoded, encoders, all_features_list)
            
            # Definisikan fitur continuous untuk DiCE
            dice_continuous_features = CONTINUOUS_COLS + ORDINAL_COLS
            
            # Buat Data Interface
            data_interface = dice_ml.Data(
                dataframe=df_dice,
                continuous_features=dice_continuous_features,
                outcome_name=TARGET_NAME
            )
            
            # Buat Model Interface
            model_interface = dice_ml.Model(
                model=self,
                backend="sklearn",
                model_type='classifier'
            )
            
            progress_bar.progress(30, text="Validasi input dan target...")
            
            # === STEP 2: Persiapkan Query Instance ===
            query_instance = input_df_processed[all_features_list].copy()
            
            # CRITICAL: Pastikan tipe data konsisten dan nilai valid
            for col in ALL_CATEGORICAL_COLS:
                if col in query_instance.columns:
                    # Konversi ke int dan pastikan dalam range yang valid
                    val = int(query_instance[col].iloc[0])
                    
                    # Validasi range untuk setiap fitur
                    if col in ['Gender', 'family_history_with_overweight', 'FAVC', 'SCC', 'SMOKE']:
                        query_instance[col] = min(max(val, 0), 1)  # Binary: 0 atau 1
                    elif col in ['CAEC', 'CALC']:
                        query_instance[col] = min(max(val, 0), 3)  # 0-3
                    elif col == 'MTRANS':
                        query_instance[col] = min(max(val, 0), 4)  # 0-4
                    elif col in ORDINAL_COLS:
                        # FCVC, NCP, CH2O: 1-3, FAF: 0-3, TUE: 0-2
                        if col in ['FCVC', 'NCP', 'CH2O']:
                            query_instance[col] = min(max(val, 1), 3)
                        elif col == 'FAF':
                            query_instance[col] = min(max(val, 0), 3)
                        elif col == 'TUE':
                            query_instance[col] = min(max(val, 0), 2)
                    
                    query_instance[col] = query_instance[col].astype(int)
            
            # Pastikan continuous features adalah float
            for col in CONTINUOUS_COLS:
                if col in query_instance.columns:
                    query_instance[col] = float(query_instance[col].iloc[0])
                    query_instance[col] = query_instance[col].astype(float)
            
            # Validasi prediksi awal
            current_pred_proba = model_obj.predict_proba(query_instance)
            current_pred_idx = np.argmax(current_pred_proba[0])
            current_pred_label = encoders[TARGET_NAME].classes_[current_pred_idx]
            
            # Dapatkan target label
            desired_class_label = encoders[TARGET_NAME].classes_[desired_class_index]
            
            # Cek apakah sudah di target
            if current_pred_label == desired_class_label:
                st.success("✅ Anda sudah berada di kategori target!")
                progress_bar.empty()
                return None
                
                for col in ['FAVC', 'CAEC', 'CALC']:
                    if col in df_dice.columns:
                        unique_vals = sorted(df_dice[col].unique())
                        st.write(f"- {col}: {unique_vals}")
            
            st.info(f"🎯 Target: **{current_pred_label.replace('_', ' ')}** → **{desired_class_label.replace('_', ' ')}**")
            
            # === STEP 3: Strategi Pencarian Bertahap ===
            progress_bar.progress(50, text="Mencari solusi optimal...")
            
            # Definisikan strategi dari yang paling strict ke paling relaxed
            # NOTE: Setiap method punya parameter yang berbeda!
            strategies = [
                {
                    'name': 'Optimal (Tanpa Ubah Berat)',
                    'allow_weight': False,
                    'total_cfs': 5,
                    'method': 'genetic',  # Hanya 1 method
                    'params': {
                        'proximity_weight': 0.5,
                        'diversity_weight': 1.0
                    }
                },
                {
                    'name': 'Balanced (Dengan Berat)',
                    'allow_weight': True,
                    'total_cfs': 10,
                    'method': 'genetic',
                    'params': {
                        'proximity_weight': 0.3,
                        'diversity_weight': 1.5
                    }
                },
                {
                    'name': 'Relaxed Genetic',
                    'allow_weight': True,
                    'total_cfs': 15,
                    'method': 'genetic',
                    'params': {
                        'proximity_weight': 0.1,
                        'diversity_weight': 2.0
                    }
                },
                {
                    'name': 'Random Search',
                    'allow_weight': True,
                    'total_cfs': 20,
                    'method': 'random',
                    'params': {
                        'num_cfs': 20  # Random method uses num_cfs instead of total_CFs
                    }
                },
                {
                    'name': 'KD-Tree (Experimental)',
                    'allow_weight': True,
                    'total_cfs': 10,
                    'method': 'kdtree',
                    'params': {}
                }
            ]
            
            # Filter strategi berdasarkan user preference
            if not allow_weight_change:
                # Jika user tidak mau ubah berat, coba tanpa berat dulu
                pass
            else:
                # Skip strategi pertama jika user OK dengan perubahan berat
                strategies = strategies[1:]
            
            # === STEP 4: Loop Strategi ===
            for strategy_idx, strategy in enumerate(strategies):
                progress_percent = 50 + (strategy_idx * 10)
                progress_bar.progress(min(progress_percent, 90), 
                                    text=f"Strategi {strategy_idx + 1}/{len(strategies)}: {strategy['name']}")
                
                # Tentukan fitur yang tidak boleh diubah
                base_immutables = ['Gender', 'Age', 'Height', 'family_history_with_overweight']
                if not strategy['allow_weight']:
                    base_immutables.append('Weight')
                
                features_to_vary = [col for col in all_features_list if col not in base_immutables]
                
                # Batasan range yang lebih luas (HANYA untuk Weight)
                permitted_range = {}
                if 'Weight' in features_to_vary:
                    current_weight = float(query_instance['Weight'].iloc[0])
                    permitted_range['Weight'] = [
                        max(30.0, current_weight * 0.5),  # Min: 50% atau 30kg
                        min(200.0, current_weight * 1.5)  # Max: 150% atau 200kg
                    ]
                
                method = strategy['method']
                
                try:
                              
                    # Buat explainer
                    dice_explainer = Dice(data_interface, model_interface, method=method)
                    
                    # Prepare parameters berdasarkan method
                    gen_params = {
                        'query_instances': query_instance,
                        'total_CFs': strategy['total_cfs'],
                        'desired_class': desired_class_label,
                        'features_to_vary': features_to_vary
                    }
                    
                    # Tambahkan permitted_range jika ada
                    if permitted_range:
                        gen_params['permitted_range'] = permitted_range
                    
                    # Tambahkan parameter spesifik method
                    if method == 'genetic':
                        gen_params.update(strategy['params'])
                    elif method == 'random':
                        # Random method tidak support proximity_weight/diversity_weight
                        pass
                    
                    # Generate counterfactuals
                    dice_result = dice_explainer.generate_counterfactuals(**gen_params)
                    
                    # Validasi hasil
                    if (dice_result and 
                        dice_result.cf_examples_list and 
                        len(dice_result.cf_examples_list) > 0 and
                        dice_result.cf_examples_list[0].final_cfs_df is not None and 
                        len(dice_result.cf_examples_list[0].final_cfs_df) > 0):
                        
                        cf_df = dice_result.cf_examples_list[0].final_cfs_df
                        num_cfs = len(cf_df)
                        
                        # Verifikasi bahwa CF benar-benar mencapai target
                        cf_predictions = model_obj.predict(cf_df[all_features_list])
                        valid_cfs = sum(cf_predictions == desired_class_label)
                        
                        if valid_cfs > 0:
                            st.success(f"✅ **Berhasil!** Ditemukan {valid_cfs}/{num_cfs} counterfactual yang valid")
                            progress_bar.progress(100, text="Selesai! ✨")
                            
                            # Filter hanya CF yang valid
                            if valid_cfs < num_cfs:
                                valid_indices = [i for i, pred in enumerate(cf_predictions) if pred == desired_class_label]
                                dice_result.cf_examples_list[0].final_cfs_df = cf_df.iloc[valid_indices].reset_index(drop=True)
                                st.info(f"ℹ️ Menampilkan {valid_cfs} rekomendasi yang tervalidasi")
                            
                            return dice_result
                        else:
                            st.warning(f"⚠️ Ditemukan {num_cfs} CF, tapi tidak ada yang mencapai target")
                    else:
                        st.write(f"   ❌ Tidak menghasilkan CF")
                
                except Exception as e:
                    error_msg = str(e)
                    
                    # Coba dengan numeric target sebagai fallback
                    if 'could not be identified' in error_msg.lower() or 'outside the dataset' in error_msg.lower():
                        try:
                            df_dice_numeric = df_dice.copy()
                            df_dice_numeric[TARGET_NAME] = encoders[TARGET_NAME].transform(
                                df_dice_numeric[TARGET_NAME].astype(str)
                            )
                            
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
                            
                            # Parameters untuk numeric
                            gen_params_num = {
                                'query_instances': query_instance,
                                'total_CFs': strategy['total_cfs'],
                                'desired_class': desired_class_index,
                                'features_to_vary': features_to_vary
                            }
                            
                            if permitted_range:
                                gen_params_num['permitted_range'] = permitted_range
                            
                            if method == 'genetic':
                                gen_params_num.update(strategy['params'])
                            
                            dice_result = dice_explainer_num.generate_counterfactuals(**gen_params_num)
                            
                            if (dice_result and 
                                dice_result.cf_examples_list and 
                                dice_result.cf_examples_list[0].final_cfs_df is not None and 
                                len(dice_result.cf_examples_list[0].final_cfs_df) > 0):
                                
                                progress_bar.progress(100, text="Selesai!")
                                return dice_result
                        
                        except Exception as e2:
                            st.write(f"   ❌ Numeric encoding gagal: {str(e2)[:100]}")
                    
                    continue
            
            # Jika semua strategi gagal
            progress_bar.progress(100, text="Selesai")
            st.error("""
            ❌ **Tidak dapat menemukan rekomendasi yang valid**
            
            Kemungkinan penyebab:
            - Gap antara kondisi saat ini dan target terlalu besar
            - Constraint fitur terlalu ketat
            - Model kesulitan menemukan kombinasi fitur yang valid
            
            💡 **Saran:**
            - Coba targetkan kelas yang lebih dekat terlebih dahulu
            - Izinkan perubahan berat badan jika memungkinkan
            """)
            return None
            
        except Exception as e:
            st.error(f"❌ Error sistem DiCE: {e}")
            st.code(traceback.format_exc())
            return None
        finally:
            progress_bar.empty()

    def decode_dataframe(self, df_dice_output, encoders, all_features_list):
        """Decode hasil DiCE ke format human-readable"""
        df_decoded = df_dice_output.copy()
        
        # Decode kategorikal
        for col_name, mapping in DECODE_MAPS.items():
            if col_name in df_decoded.columns:
                df_decoded[col_name] = df_decoded[col_name].astype(str).map(
                    {str(k): v for k, v in mapping.items()}
                )
        
        # Format continuous
        for col in CONTINUOUS_COLS:
            if col in df_decoded.columns:
                df_decoded[col] = pd.to_numeric(df_decoded[col], errors='coerce').round(2)
        
        # Format ordinal
        for col in ORDINAL_COLS:
            if col in df_decoded.columns:
                df_decoded[col] = pd.to_numeric(df_decoded[col], errors='coerce').round().astype(int)
        
        # Decode target
        if TARGET_NAME in df_decoded.columns:
            try:
                numeric_vals = pd.to_numeric(df_decoded[TARGET_NAME], errors='coerce')
                if not numeric_vals.isna().all():
                    df_decoded[TARGET_NAME] = encoders[TARGET_NAME].inverse_transform(
                        numeric_vals.round().astype(int)
                    )
            except:
                pass
        
        # Return dengan kolom yang diminta
        available_cols = [col for col in all_features_list + [TARGET_NAME] if col in df_decoded.columns]
        return df_decoded[available_cols]

    def summarize_changes(self, dice_result, input_df_processed, encoders, all_features_list):
        """Generate human-readable summary dengan action verbs"""
        try:
            if not dice_result or not dice_result.cf_examples_list or \
               dice_result.cf_examples_list[0].final_cfs_df is None:
                return []

            cf_df = dice_result.cf_examples_list[0].final_cfs_df
            cf_decoded = self.decode_dataframe(cf_df, encoders, all_features_list)
            
            q_df = input_df_processed.copy()
            if TARGET_NAME not in q_df.columns:
                q_df[TARGET_NAME] = cf_decoded[TARGET_NAME].iloc[0]
            q_decoded = self.decode_dataframe(q_df, encoders, all_features_list)
            original_data = q_decoded.iloc[0]

            # Kumpulkan perubahan dari semua CF
            feature_changes = {}
            for _, row in cf_decoded.iterrows():
                for col in all_features_list:
                    # Skip fitur yang tidak boleh diubah
                    if col in [TARGET_NAME, 'Gender', 'Age', 'Height', 'family_history_with_overweight']:
                        continue
                    
                    if str(original_data[col]) != str(row[col]):
                        if col not in feature_changes:
                            feature_changes[col] = []
                        feature_changes[col].append(row[col])

            # Generate summary dengan action verbs
            summary_list = []
            sorted_features = sorted(feature_changes.items(), key=lambda x: len(x[1]), reverse=True)
            
            # Mapping untuk display yang lebih baik
            human_readable_map = {
                'FCVC': {1: 'Tidak Pernah', 2: 'Setengah dari jumlah makan per hari', 3: 'Setiap Makan'},
                'NCP': {1: '1x/hari', 2: '2x/hari', 3: '3x/hari', 4: '4x/hari'},
                'CH2O': {1: '<1 Liter', 2: '1-2 Liter', 3: '>2 Liter'},
                'FAF': {0: '< 15 menit', 1: '15 - 30 menit', 2: '30 - 60 menit', 3: '+ 60 menit'},
                'TUE': {0: '< 1 jam', 1: '1-2 jam', 2: '>2 jam'},
                'CAEC': {'no': 'Tidak Pernah', 'Sometimes': '1-2x/minggu', 'Frequently': '3-5x/minggu', 'Always': '6-7x/minggu'},
                'CALC': {'no': 'Tidak Pernah', 'Sometimes': '2 Porsi', 'Frequently': '3 Porsi', 'Always': '>4 Porsi'},
                'FAVC': {'no': 'Tidak', 'yes': 'Ya'},
                'SCC': {'no': 'Tidak', 'yes': 'Ya'},
                'SMOKE': {'no': 'Tidak', 'yes': 'Ya'},
                'MTRANS': {
                    'Walking': 'Jalan Kaki',
                    'Public_Transportation': 'Transport Umum',
                    'Bike': 'Sepeda',
                    'Motorbike': 'Motor',
                    'Automobile': 'Mobil'
                }
            }

            # Ambil top 7 perubahan
            for feature, values in sorted_features[:7]:
                most_common_val, count = Counter(values).most_common(1)[0]
                
                # Gunakan action verb dari config
                action = ACTION_VERBS.get(feature, f"Ubah {feature}")
                
                # Format nilai
                trans_val = most_common_val
                if feature in human_readable_map:
                    try:
                        k = int(float(most_common_val)) if isinstance(most_common_val, (int, float, str)) and str(most_common_val).replace('.','').isdigit() else str(most_common_val)
                    except:
                        k = str(most_common_val)
                    trans_val = human_readable_map[feature].get(k, most_common_val)
                
                if feature == 'Weight':
                    trans_val = f"{float(most_common_val):.1f} kg"
                
                # Format persentase kemunculan
                percentage = (count / len(cf_decoded)) * 100
                
                summary_list.append(
                    f"**{action}** menjadi **{trans_val}** "
                    f"({percentage:.0f}% rekomendasi menyarankan ini)"
                )
            
            return summary_list
            
        except Exception as e:
            st.error(f"Error generating summary: {e}")
            return []

    def get_step_description(self, current_class, target_class, is_final_step, all_steps):
        """Generate deskripsi langkah rekomendasi berdasarkan hirarki"""
        try:
            if current_class == target_class:
                return f"Anda sudah berada di kategori **{current_class.replace('_', ' ')}**."
            
            if is_final_step:
                return f"Langkah terakhir: Pindah ke kategori **{target_class.replace('_', ' ')}** untuk mencapai berat badan ideal."
            else:
                next_index = all_steps.index(current_class) + 1
                next_class = all_steps[next_index] if next_index < len(all_steps) else target_class
                return f"Langkah selanjutnya: Pindah ke kategori **{next_class.replace('_', ' ')}**."
        except Exception as e:
            st.error(f"Error generating step description: {e}")
            return ""
# ==============================================================================
# GLOBAL FUNCTIONS (untuk kompatibilitas dengan import)
# ==============================================================================


def get_step_description(current_class, target_class, is_final_step, all_steps):
    """Wrapper function untuk mendapatkan deskripsi langkah"""
    helper = DiceHelper()
    return helper.get_step_description(current_class, target_class, is_final_step, all_steps)

def get_dice_recommendations(x_train_encoded, model_obj, encoders, input_df_processed, 
                             desired_class_index, all_features_list, allow_weight_change=True):
    """Wrapper function untuk generate recommendations"""
    helper = DiceHelper(
        model=model_obj,
        feature_names=all_features_list,
        continuous_features=CONTINUOUS_COLS,
        categorical_features=ALL_CATEGORICAL_COLS,
        class_labels=encoders[TARGET_NAME].classes_
    )
    return helper.get_dice_recommendations(
        x_train_encoded, model_obj, encoders, input_df_processed, 
        desired_class_index, all_features_list, allow_weight_change
    )

def summarize_dice_changes(dice_result, input_df_processed, encoders, all_features_list):
    """Wrapper function untuk generate summary"""
    helper = DiceHelper()
    return helper.summarize_changes(dice_result, input_df_processed, encoders, all_features_list)

def decode_dice_dataframe(df_dice_output, encoders, all_features_list):
    """Wrapper function untuk decode dataframe"""
    helper = DiceHelper()
    return helper.decode_dataframe(df_dice_output, encoders, all_features_list)

def get_next_target_class(current_class, class_names):
    """Wrapper function untuk mendapatkan target kelas berikutnya"""
    helper = DiceHelper()
    return helper.get_next_target_class(current_class, class_names)