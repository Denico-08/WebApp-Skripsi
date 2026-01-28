import os
from datetime import date
import json
from typing import Optional, Any, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(env_path, encoding="latin-1") # atau encoding="utf-8"

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT")
        )
        return conn
    except Exception as e:
        print(f"Error connecting to DB: {e}")
        return None

def insert_input_to_supabase(user_input_raw: dict, user_id: Optional[Any] = None) -> Tuple[bool, Any]:
    
    conn = get_db_connection()
    if conn is None:
        return False, "Database connection failed"

    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # --- PERBAIKAN: Menambahkan tanda kutip " setelah ID_User ---
        query = """
            INSERT INTO "DataInput" (
                "Age", "Gender", "Height", "Weight", "FamilyHistory", 
                "FAVC", "CAEC", "CALC", "MTRANS", "CH20", 
                "FCVC", "NCP", "FAF", "TUE", "CreateInput", "ID_User"
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING "ID_Input";
        """
        # -------------------------------------------------------------

        # Ambil ID User (jika ada)
        id_user_value = user_id if user_id is not None else os.environ.get('SUPABASE_DEFAULT_USER')
        
        values = (
            int(user_input_raw.get('Age', 0)),
            str(user_input_raw.get('Gender', '')),
            float(user_input_raw.get('Height', 0.0)),
            float(user_input_raw.get('Weight', 0.0)),
            True if str(user_input_raw.get('family_history_with_overweight', '')).lower() in ['yes', 'y', 'true', '1'] else False,
            str(user_input_raw.get('FAVC', '')),
            str(user_input_raw.get('CAEC', '')),
            str(user_input_raw.get('CALC', '')),
            str(user_input_raw.get('MTRANS', '')),
            int(user_input_raw.get('CH2O', 0)), # Cek: Apakah form mengirim 'CH2O' (huruf) atau 'CH20' (angka)?
            int(user_input_raw.get('FCVC', 0)),
            int(user_input_raw.get('NCP', 0)),
            int(user_input_raw.get('FAF', 0)),
            int(user_input_raw.get('TUE', 0)),
            date.today().isoformat(),
            id_user_value
        )

        # Eksekusi
        cur.execute(query, values)
        
        # Ambil ID yang baru dibuat
        result = cur.fetchone()
        new_id = result['ID_Input'] if result else None
        
        # Commit (Simpan Permanen)
        conn.commit()
        
        # Tutup Koneksi
        cur.close()
        conn.close()
        
        return True, new_id

    except Exception as e:
        if conn: conn.rollback()
        # Print error ke terminal supaya terlihat jelas penyebabnya
        print(f"❌ GAGAL INSERT INPUT: {e}") 
        return False, str(e)

def insert_faktor_dominan(id_prediksi: int, top_features: Any) -> Tuple[bool, Any]:
    conn = get_db_connection()
    if conn is None:
        return False, "Database connection failed"

    try:
        cur = conn.cursor()
        
        # Konversi data dictionary/list ke string JSON
        features_json = json.dumps(top_features)

        query = """
            INSERT INTO "Faktor_Dominan" ("ID_Prediksi", "TopFeature")
            VALUES (%s, %s)
            RETURNING "ID_Prediksi";
        """
        
        cur.execute(query, (id_prediksi, features_json))
        
        # Ambil data yang baru diinsert (opsional, untuk konfirmasi)
        result_id = cur.fetchone()[0] # type: ignore
        
        conn.commit()
        cur.close()
        conn.close()
        
        # Mengembalikan data dalam format yang mirip dengan response Supabase sebelumnya
        return True, [{'ID_Prediksi': result_id, 'TopFeature': top_features}]

    except Exception as e:
        if conn: conn.rollback()
        return False, str(e)

def insert_rekomendasi_to_supabase(id_prediksi: int, target_prediksi: str, perubahan_prediksi: Any) -> Tuple[bool, Any]:
    # Catatan: Nama fungsi bisa Anda ganti jadi insert_rekomendasi_to_postgres jika mau
    conn = get_db_connection()
    if conn is None:
        return False, "Database connection failed"

    try:
        # Hitung jumlah perubahan (logika sama seperti sebelumnya)
        jumlah = 0
        if isinstance(perubahan_prediksi, dict):
            jumlah = len(perubahan_prediksi)
        elif isinstance(perubahan_prediksi, list):
            jumlah = len(perubahan_prediksi)

        # Konversi data perubahan ke string JSON
        perubahan_json = json.dumps(perubahan_prediksi)

        cur = conn.cursor()
        query = """
            INSERT INTO "Rekomendasi" 
            ("ID_Prediksi", "Target_Prediksi", "Jumlah_Perubahan", "Perubahan_Minimal")
            VALUES (%s, %s, %s, %s)
            RETURNING "ID_Rekomendasi"; 
        """
        # Catatan: Pastikan nama kolom Primary Key tabel Rekomendasi benar (misal ID_Rekomendasi)
        # Jika tabel tidak punya auto-increment PK yang perlu dikembalikan, RETURNING bisa disesuaikan.

        cur.execute(query, (id_prediksi, target_prediksi, jumlah, perubahan_json))
        
        conn.commit()
        cur.close()
        conn.close()
        
        # Mengembalikan success state
        return True, "Data rekomendasi berhasil disimpan"

    except Exception as e:
        if conn: conn.rollback()
        return False, str(e)

def insert_prediction_to_supabase(id_input: int, hasil_prediksi: str, probabilitas: float) -> Tuple[bool, Any]:
    conn = get_db_connection()
    if conn is None:
        return False, "Database connection failed"

    try:
        cur = conn.cursor()
        
        query = """
            INSERT INTO "Prediksi" ("ID_DataInput", "Hasil_Prediksi", "Probabilitas")
            VALUES (%s, %s, %s)
            RETURNING "ID_Prediksi";
        """
        
        cur.execute(query, (id_input, hasil_prediksi, probabilitas))
        
        # Mengambil ID hasil insert
        new_id = cur.fetchone()[0] # type: ignore
        
        conn.commit()
        cur.close()
        conn.close()

        # Mengembalikan data dummy yang strukturnya mirip response Supabase agar tidak error di frontend
        return True, [{'ID_Prediksi': new_id, 'Hasil_Prediksi': hasil_prediksi, 'Probabilitas': probabilitas}]

    except Exception as e:
        if conn: conn.rollback()
        return False, str(e)