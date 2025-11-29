import streamlit as st
import pandas as pd
from Connection.supabase_client import get_supabase_client
from Role import Role

class User:
    def __init__(self, email=None, password=None, name=None):
        self.email = email
        self.password = password
        self.name = name
        self.role = Role.USER.value
        self.id = None

    # ==========================================================================
    # LOGIKA BISNIS (BACKEND - AUTH)
    # ==========================================================================

    def login_action(self) -> bool:
        """Logika autentikasi ke Supabase."""
        if not self.email or not self.password:
            st.error("Email dan password harus diisi")
            return False
        
        try:
            supabase = get_supabase_client()
            response = supabase.table("User").select("*").eq("Email", self.email).eq("Password", self.password).execute()
            
            if response.data and len(response.data) > 0:
                user_data = response.data[0]
                self.id = user_data.get("ID_User") #type: ignore
                self.name = user_data.get("Nama") #type: ignore
                
                # Mengambil role dari database
                db_role = user_data.get("Role") #type: ignore
                
                # --- UPDATE 3: Pastikan role valid (Opsional tapi bagus) ---
                # Jika di DB tertulis "Admin", kita simpan sebagai string "Admin"
                self.role = db_role if db_role else Role.USER.value
                
                st.session_state.user_authenticated = True
                st.session_state.user = self.email
                st.session_state.user_id = self.id
                st.session_state.user_role = self.role
                st.session_state.user_name = self.name
                return True
            return False
        except Exception as e:
            st.error(f"Error saat login: {str(e)}")
            return False

    def register_action(self, confirm_password: str) -> bool:
        """Logika pendaftaran pengguna baru."""
        if not all([self.email, self.name, self.password, confirm_password]):
            st.error("Semua field harus diisi.")
            return False

        if self.password != confirm_password:
            st.error("Password tidak cocok.")
            return False
        
        if len(self.password) < 6: #type: ignore
            st.error("Password minimal 6 karakter.")
            return False

        try:
            supabase = get_supabase_client()
            auth_response = supabase.auth.sign_up({"email": self.email, "password": self.password}) #type: ignore

            if auth_response.user:
                user_id = auth_response.user.id
                insert_data = {
                    "ID_User": user_id, "Email": self.email, 
                    "Password": self.password, "Nama": self.name, "Role": Role.USER.value
                }
                supabase.table("User").insert(insert_data).execute()
                return True
            else:
                st.error("Pendaftaran gagal.")
                return False
        except Exception as e:
            st.error(f"Error register: {e}")
            return False

    def get_history_data(self) -> pd.DataFrame:

        if not self.id:
            return pd.DataFrame()

        client = get_supabase_client()
        try:
            # 1. Ambil Data Input milik User
            input_resp = client.table("DataInput").select("*").eq("ID_User", self.id).execute()
            input_data = input_resp.data
            
            if not input_data:
                return pd.DataFrame()

            # 2. Ambil ID Input untuk query Prediksi
            input_ids = [item.get('ID_Input') for item in input_data if item.get('ID_Input') is not None] #type: ignore
            
            pred_data = []
            if input_ids:
                pred_resp = client.table("Prediksi").select("*").in_("ID_DataInput", input_ids).execute()
                pred_data = pred_resp.data if hasattr(pred_resp, 'data') else []
            
            df_inputs = pd.DataFrame(input_data)
            df_preds = pd.DataFrame(pred_data)

            # Pastikan tipe data ID konsisten (string) agar merge berhasil
            if 'ID_Input' in df_inputs.columns:
                df_inputs['ID_Input'] = df_inputs['ID_Input'].astype(str)
            if 'ID_DataInput' in df_preds.columns:
                df_preds['ID_DataInput'] = df_preds['ID_DataInput'].astype(str)

            # 3. Merge Data
            if not df_preds.empty:
                right = df_preds.copy()
                # Hindari duplikasi ID_User
                if 'ID_User' in right.columns:
                    right = right.drop(columns=['ID_User'])

                # Normalisasi nama kolom hasil prediksi
                cols_lower = {c.lower(): c for c in right.columns}
                if 'hasil_prediksi' not in cols_lower:
                    candidate = None
                    for k in cols_lower:
                        if 'hasil' in k or 'predik' in k or 'prediction' in k:
                            candidate = cols_lower[k]
                            break
                    if candidate:
                        right = right.rename(columns={candidate: 'Hasil_Prediksi'})
                    else:
                        right['Hasil_Prediksi'] = None

                df_history = pd.merge(df_inputs, right, left_on='ID_Input', right_on='ID_DataInput', how='left')

                # Fallback check column names after merge
                if 'Hasil_Prediksi' not in df_history.columns:
                    candidates = [c for c in df_history.columns if 'hasil' in c.lower() or 'predik' in c.lower() or 'prediction' in c.lower()]
                    if candidates:
                        df_history['Hasil_Prediksi'] = df_history[candidates[0]]
                    else:
                        df_history['Hasil_Prediksi'] = None
            else:
                df_history = df_inputs.copy()
                df_history['Hasil_Prediksi'] = "N/A"

            # Normalize missing hasil values
            try:
                df_history['Hasil_Prediksi'] = df_history['Hasil_Prediksi'].fillna('N/A')
            except Exception:
                pass

            return df_history

        except Exception as e:
            st.error(f"Gagal mengambil history: {e}")
            return pd.DataFrame()

    # ==========================================================================
    # UTILITIES (LOGOUT & AUTH CHECK)
    # ==========================================================================

    @staticmethod
    def logout():
        try:
            get_supabase_client().auth.sign_out()
        except Exception: pass
        for key in ['user_authenticated', 'user', 'user_id', 'user_role', 'user_name']:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

    @staticmethod
    def require_auth():
        if not st.session_state.get("user_authenticated"):
            st.warning("Silakan login terlebih dahulu.")
            st.stop()

    # ==========================================================================
    # UI METHODS (TAMPILAN HALAMAN)
    # ==========================================================================

    @staticmethod
    def render_login_page():
        """Tampilan Halaman Login."""
        if st.session_state.get("user_authenticated"):
            return # Sudah login

        st.title("Selamat Datang!")
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.form_submit_button("Login", use_container_width=True):
                user = User(email=email, password=password)
                if user.login_action():
                    st.success(f"Login berhasil! Halo {st.session_state.user_name}")
                    st.rerun()
                else:
                    st.error("Login gagal.")
        
        st.markdown("---")
        if st.button("Daftar Akun Baru"):
            st.session_state.page = "signup"
            st.rerun()

    @staticmethod
    def render_signup_page():
        """Tampilan Halaman Registrasi."""
        st.title("Daftar Akun")
        if st.button("Kembali ke Login"):
            st.session_state.page = "login"
            st.rerun()

        with st.form("signup_form"):
            email = st.text_input("Email")
            name = st.text_input("Nama Lengkap")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Konfirmasi Password", type="password")
            
            if st.form_submit_button("Daftar", use_container_width=True):
                user = User(email=email, password=password, name=name)
                if user.register_action(confirm):
                    st.success("Berhasil! Silakan login.")

    def render_history_page(self):
        """Tampilan Halaman History User yang lebih informatif."""
        st.title(f"Riwayat Prediksi: {self.name}")

        if st.button("⬅️ Kembali ke Halaman Prediksi"):
            st.session_state.page = "prediksi"
            st.rerun()

        df = self.get_history_data()

        if not df.empty:
            # Urutkan berdasarkan tanggal, dari yang terbaru
            if 'CreateInput' in df.columns and pd.api.types.is_datetime64_any_dtype(df['CreateInput']):
                df = df.sort_values(by='CreateInput', ascending=False)

            # Kamus untuk nama kolom yang lebih deskriptif
            column_mapping = {
                'Gender': 'Jenis Kelamin',
                'Age': 'Usia',
                'Height': 'Tinggi (cm)',
                'Weight': 'Berat (kg)',
                'family_history_with_overweight': 'Riwayat Keluarga Obesitas',
                'FAVC': 'Sering Konsumsi Makanan Tinggi Kalori',
                'FCVC': 'Frekuensi Konsumsi Sayuran',
                'NCP': 'Jumlah Makanan Utama per Hari',
                'CALC': 'Konsumsi Alkohol',
                'SCC': 'Memantau Konsumsi Kalori',
                'SMOKE': 'Merokok',
                'CH2O': 'Konsumsi Air per Hari (Liter)',
                'FAF': 'Frekuensi Aktivitas Fisik per Minggu',
                'TUE': 'Waktu Menggunakan Gadget per Hari (Jam)',
                'MTRANS': 'Transportasi yang Digunakan',
                'CreateInput': 'Tanggal Prediksi'
            }

            for index, row in df.iterrows():
                try:
                    # Format tanggal untuk judul expander
                    date_str = "Tanggal tidak tersedia"
                    if 'CreateInput' in row and pd.notna(row['CreateInput']):
                        date_str = pd.to_datetime(row['CreateInput']).strftime('%d %B %Y')
                    
                    hasil_prediksi = row.get('Hasil_Prediksi', 'N/A')
                    expander_title = f"Prediksi pada {date_str} — Hasil: **{hasil_prediksi}**"

                    with st.expander(expander_title):
                        st.markdown("##### Detail Data yang Anda Masukkan:")
                        
                        # Bagi detail menjadi dua kolom agar tidak terlalu panjang
                        col1, col2 = st.columns(2)
                        
                        # Iterasi melalui kolom yang relevan untuk ditampilkan
                        input_data_cols = [col for col in column_mapping if col in row]
                        
                        # Bagi kolom menjadi dua list untuk ditampilkan di dua kolom
                        mid_point = (len(input_data_cols) + 1) // 2
                        
                        with col1:
                            for col_name in input_data_cols[:mid_point]:
                                label = column_mapping.get(col_name, col_name)
                                value = row[col_name]
                                st.text(f"{label}: {value}")
                                
                        with col2:
                            for col_name in input_data_cols[mid_point:]:
                                label = column_mapping.get(col_name, col_name)
                                value = row[col_name]
                                st.text(f"{label}: {value}")

                except Exception as e:
                    st.warning(f"Gagal menampilkan salah satu riwayat. Data mentah di bawah.")
                    st.write(row) 

        else:
            st.info("Belum ada riwayat prediksi.")