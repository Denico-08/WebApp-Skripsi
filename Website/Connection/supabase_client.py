import os
from datetime import date
from typing import Optional, Any, Tuple
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Validate credentials
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")

# Clean up URL if it has duplicate parts
if "https://" in SUPABASE_URL and SUPABASE_URL.count("https://") > 1:
    SUPABASE_URL = "https://" + SUPABASE_URL.split("https://")[1].split("https://")[0]

def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)  # type: ignore

def insert_input_to_supabase(user_input_raw: dict, user_id: Optional[Any] = None) -> Tuple[bool, Any]:

    client = get_supabase_client()
    if client is None:
        return False, "Supabase client not configured (SUPABASE_URL/KEY missing or library not installed)"

    try:
        id_user_value = user_id if user_id is not None else os.environ.get('SUPABASE_DEFAULT_USER')

        payload = {
            'Age': int(user_input_raw.get('Age', 0)),
            'Gender': str(user_input_raw.get('Gender', '')),
            'Height': float(user_input_raw.get('Height', 0.0)),
            'Weight': float(user_input_raw.get('Weight', 0.0)),
            'FamilyHistory': True if str(user_input_raw.get('family_history_with_overweight', '')).lower() in ['yes', 'y', 'true', '1', 'ya', 'ya'] else False,
            'FAVC': str(user_input_raw.get('FAVC', '')),
            'CAEC': str(user_input_raw.get('CAEC', '')),
            'SMOKE': str(user_input_raw.get('SMOKE', '')),
            'SCC': str(user_input_raw.get('SCC', '')),
            'CALC': str(user_input_raw.get('CALC', '')),
            'MTRANS': str(user_input_raw.get('MTRANS', '')),
            'CH20': int(user_input_raw.get('CH2O', 0)),
            'FCVC': int(user_input_raw.get('FCVC', 0)),
            'NCP': int(user_input_raw.get('NCP', 0)),
            'FAF': int(user_input_raw.get('FAF', 0)),
            'TUE': int(user_input_raw.get('TUE', 0)),
            'CreateInput': date.today().isoformat(),
            'ID_User': id_user_value,
        }

        resp = client.table('DataInput').insert(payload).execute()
        
        # Check for errors first
        if hasattr(resp, 'data') and resp.data:
            # Assuming the response contains the inserted data in a list
            inserted_data = resp.data[0]
            id_input = inserted_data.get('ID_Input')#type: ignore
            return True, id_input
        else:
            # Handle cases where insertion might have failed or returned no data
            error_message = "Failed to insert data or no data returned from Supabase."
            if hasattr(resp, 'error') and resp.error:#type: ignore
                error_message = resp.error.message#type: ignore
            return False, error_message

    except Exception as e:
        return False, str(e)


def insert_faktor_dominan(id_prediksi: int, top_features: Any) -> Tuple[bool, Any]:
    client = get_supabase_client()
    if client is None:
        return False, "Supabase client not configured"

    try:
        payload = {
            'ID_Prediksi': id_prediksi,
            'TopFeature': top_features,
        }
        resp = client.table('Faktor_Dominan').insert(payload).execute()
        if hasattr(resp, 'data') and resp.data:
            return True, resp.data
        else:
            error_message = 'Failed to insert faktor dominan or no data returned.'
            if hasattr(resp, 'error') and resp.error: #type: ignore
                try:
                    error_message = resp.error.message #type: ignore
                except Exception:
                    error_message = str(resp.error) #type: ignore
            return False, error_message
    except Exception as e:
        return False, str(e)


def insert_rekomendasi_to_supabase(id_prediksi: int, target_prediksi: str, perubahan_prediksi: Any) -> Tuple[bool, Any]:

    client = get_supabase_client()
    if client is None:
        return False, "Supabase client not configured"

    try:
        jumlah = 0
        if isinstance(perubahan_prediksi, dict):
            jumlah = len(perubahan_prediksi)
        elif isinstance(perubahan_prediksi, list):
            jumlah = len(perubahan_prediksi)

        payload = {
            'ID_Prediksi': id_prediksi,
            'Target_Prediksi': target_prediksi,
            'Jumlah_Perubahan': jumlah,
            'Perubahan_Minimal': perubahan_prediksi,
            'ID_Prediksi': id_prediksi,
        }

        resp = client.table('Rekomendasi').insert(payload).execute()
        if hasattr(resp, 'data') and resp.data:
            return True, resp.data
        else:
            error_message = 'Failed to insert rekomendasi or no data returned.'
            if hasattr(resp, 'error') and resp.error: #type: ignore
                try:
                    error_message = resp.error.message #type: ignore
                except Exception:
                    error_message = str(resp.error) #type: ignore
            return False, error_message
    except Exception as e:
        return False, str(e)

def insert_prediction_to_supabase(id_input: int, hasil_prediksi: str, probabilitas: float) -> Tuple[bool, Any]:

    client = get_supabase_client()
    if client is None:
        return False, "Supabase client not configured"

    try:
        payload = {
            'ID_DataInput': id_input,
            'Hasil_Prediksi': hasil_prediksi,
            'Probabilitas': probabilitas,
        }

        resp = client.table('Prediksi').insert(payload).execute()

        if hasattr(resp, 'data') and resp.data:
            return True, resp.data
        else:
            error_message = "Failed to insert prediction or no data returned from Supabase."
            if resp.error: #type: ignore
                error_message = resp.error.message if resp.error.message else str(resp.error) #type: ignore
            return False, error_message

    except Exception as e:
        return False, str(e)
