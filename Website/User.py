import streamlit as st
import bcrypt
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from Connection.db_client import get_db_connection
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
        conn = get_db_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor) # type: ignore
            cur.execute('SELECT * FROM "User" WHERE "Email" = %s', (self.email,))
            user_data = cur.fetchone()

            if user_data:
                # Verifikasi password hash
                stored_password = user_data['Password']
                if bcrypt.checkpw(self.password.encode('utf-8'), stored_password.encode('utf-8')): # type: ignore
                    self.id = user_data["ID_User"]
                    self.name = user_data["Nama"]
                    self.role = user_data["Role"] if user_data["Role"] else Role.USER.value

                    # Set session state (sama seperti kode lama)
                    st.session_state.user_authenticated = True
                    st.session_state.user = self.email
                    st.session_state.user_id = self.id
                    st.session_state.user_role = self.role
                    st.session_state.user_name = self.name
                    return True
            return False
        except Exception as e:
            return False

    def register_action(self, confirm_password: str) -> bool:
        """Logika pendaftaran pengguna baru."""
        if not all([self.email, self.name, self.password, confirm_password]):
            st.error("Semua kolom harus diisi.")
            return False

        if self.password != confirm_password:
            st.error("Password dan Konfirmasi Password tidak cocok.")
            return False
        
        if len(self.password) < 6: # type: ignore
            st.error("Password minimal 6 karakter.")
            return False

        # Hash password
        import bcrypt
        hashed = bcrypt.hashpw(self.password.encode('utf-8'), bcrypt.gensalt()) # type: ignore
        
        conn = get_db_connection()
        
        # --- PERBAIKAN: Cek koneksi sebelum lanjut ---
        if conn is None:
            st.error("Gagal terhubung ke database. Periksa koneksi internet atau konfigurasi server.")
            return False
        # ---------------------------------------------

        try:
            cur = conn.cursor()
            
            # Generate ID manual (UUID)
            import uuid
            new_id = str(uuid.uuid4())
            
            # Query Insert
            sql = """INSERT INTO "User" ("ID_User", "Email", "Password", "Nama", "Role") 
                     VALUES (%s, %s, %s, %s, %s)"""
            
            # Jalankan query
            cur.execute(sql, (new_id, self.email, hashed.decode('utf-8'), self.name, Role.USER.value))
            
            conn.commit()
            cur.close()
            conn.close()
            return True

        except Exception as e:
            # Jika conn terbuka tapi query gagal, tutup koneksi
            if conn: conn.close()
            st.error(f"Error register: {e}")
            return False

    def get_history_data(self) -> pd.DataFrame:
        conn = get_db_connection()
        query = f"""
        SELECT i.*, p."Hasil_Prediksi", p."Probabilitas"
        FROM "DataInput" i
        LEFT JOIN "Prediksi" p ON i."ID_Input" = p."ID_DataInput"
        WHERE i."ID_User" = '{self.id}'
        ORDER BY i."CreateInput" DESC
        """
        return pd.read_sql(query, conn)

    # ==========================================================================
    # UTILITIES (LOGOUT & AUTH CHECK)
    # ==========================================================================

    @staticmethod
    def logout():
        # Daftar key yang perlu dihapus dari session state
        keys_to_clear = ['user_authenticated', 'user', 'user_id', 'user_role', 'user_name']
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
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
                    st.error("Login Tidak Valid")
        
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
            password = st.text_input("Password 6 karakter", type="password")
            confirm = st.text_input("Konfirmasi Password", type="password")
            
            if st.form_submit_button("Daftar", use_container_width=True):
                user = User(email=email, password=password, name=name)
                if user.register_action(confirm):
                    st.success("Sign Up Berhasil")
                else:
                    st.error("Pendaftaran Tidak Valid")

    def render_history_page(self):
        """Tampilan Halaman History User yang lebih informatif."""
        st.title(f"Riwayat Prediksi: {self.name}")

        if st.button("⬅️ Kembali ke Halaman Prediksi"):
            st.session_state.page = "prediksi"
            st.rerun()

        df = self.get_history_data()

        if not df.empty:
            # Urutkan berdasarkan tanggal, dari yang terbaru
            try:
                if 'CreateInput' in df.columns:
                    # Konversi ke datetime jika belum
                    if not pd.api.types.is_datetime64_any_dtype(df['CreateInput']):
                        df['CreateInput'] = pd.to_datetime(df['CreateInput'], errors='coerce')
                    df = df.sort_values(by='CreateInput', ascending=False, na_position='last')
            except Exception as e:
                pass

            column_mapping = {
                'Gender': 'Jenis Kelamin',
                'Age': 'Usia',
                'Height': 'Tinggi (cm)',
                'Weight': 'Berat (kg)',
                'FamilyHistory': 'Riwayat Keluarga Obesitas',
                'FAVC': 'Sering Konsumsi Makanan Tinggi Kalori',
                'FCVC': 'Frekuensi Konsumsi Sayuran',
                'NCP': 'Jumlah Makanan Utama per Hari',
                'CALC': 'Konsumsi Alkohol',
                'CH20': 'Konsumsi Air per Hari (Liter)',
                'FAF': 'Frekuensi Aktivitas Fisik per Minggu',
                'TUE': 'Waktu Menggunakan Gadget per Hari (Jam)',
                'MTRANS': 'Transportasi yang Digunakan',
                'CreateInput': 'Tanggal Prediksi'
            }
            
            # Mapping untuk decode nilai ordinal
            ordinal_decoders = {
                'FCVC': {1: 'Tidak Pernah', 2: 'Setengah dari jumlah makan per hari', 3: 'Setiap Makan', '1': 'Tidak Pernah', '2': 'Setengah dari jumlah makan per hari', '3': 'Setiap Makan'},
                'NCP': {1: '1x/hari', 2: '2x/hari', 3: '3x/hari', 4: '4x/hari', '1': '1x/hari', '2': '2x/hari', '3': '3x/hari', '4': '4x/hari'},
                'CH20': {1: '<1 Liter', 2: '1-2 Liter', 3: '>2 Liter', '1': '<1 Liter', '2': '1-2 Liter', '3': '>2 Liter'},
                'FAF': {0: '< 15 menit', 1: '15 - 30 menit', 2: '30 - 60 menit', 3: '> 60 menit', '0': '< 15 menit', '1': '15 - 30 menit', '2': '> 60 Menit'},
                'TUE': {0: '< 1 jam', 1: '1-2 jam', 2: '>2 jam', '0' : '< 1 jam', '1': '1-2 jam', '2': '>2 jam'},
                'CAEC': {'no': 'Tidak Pernah', 'Sometimes': '1-2x/minggu', 'Frequently': '3-5x/minggu', 'Always': '6-7x/minggu'},
                'CALC': {'no': 'Tidak Pernah', 'Sometimes': '2 Porsi', 'Frequently': '3 Porsi', 'Always': '>4 Porsi'},
                'FAVC': {'no': 'Tidak', 'yes': 'Ya', 0: 'Tidak', 1: 'Ya', '0': 'Tidak', '1': 'Ya'},
                'FamilyHistory': {True: 'Ya', False: 'Tidak', 'true': 'Ya', 'false': 'Tidak', 'True': 'Ya', 'False': 'Tidak', 1: 'Ya', 0: 'Tidak', '1': 'Ya', '0': 'Tidak'},
                'MTRANS': {
                    'Walking': 'Jalan Kaki',
                    'Public_Transportation': 'Transport Umum',
                    'Bike': 'Sepeda',
                    'Motorbike': 'Motor',
                    'Automobile': 'Mobil'
                }
            }
            
            # Helper function untuk decode nilai
            def decode_value(col_name, value):
                """Dekode nilai ordinal menjadi teks yang deskriptif"""
                if col_name in ordinal_decoders:
                    try:
                        # Try berbagai format: original value, string, dan integer
                        decoded = ordinal_decoders[col_name].get(value, None)
                        if decoded is None and isinstance(value, (int, float)):
                            decoded = ordinal_decoders[col_name].get(str(int(value)), None)
                            if decoded is None:
                                decoded = ordinal_decoders[col_name].get(int(value), None)
                        if decoded is None and isinstance(value, str):
                            try:
                                decoded = ordinal_decoders[col_name].get(int(value), None)
                            except:
                                pass
                        return decoded if decoded else str(value)
                    except:
                        return str(value)
                return str(value) if pd.notna(value) else 'N/A'

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
                                # Decode nilai jika ordinal
                                display_value = decode_value(col_name, value)
                                st.text(f"{label}: {display_value}")
                                
                        with col2:
                            for col_name in input_data_cols[mid_point:]:
                                label = column_mapping.get(col_name, col_name)
                                value = row[col_name]
                                # Decode nilai jika ordinal
                                display_value = decode_value(col_name, value)
                                st.text(f"{label}: {display_value}")

                except Exception as e:
                    st.warning(f"Gagal menampilkan salah satu riwayat. Data mentah di bawah.")
                    st.write(row) 

        else:
            st.info("Mohon melakukan prediksi terlebih dahulu")