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
        # Determine ID_User: prefer provided user_id, else try environment
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

        # Interpret response (supabase-py may return an object or dict)
        try:
            if hasattr(resp, 'status_code'):
                if getattr(resp, 'status_code') >= 400:
                    return False, f"Supabase returned status {resp}"
                return True, resp
        except Exception:
            pass

        # If dict-like
        try:
            if isinstance(resp, dict):
                if resp.get('error'):
                    return False, resp.get('error')
                return True, resp.get('data', resp)
        except Exception:
            pass

        # Fallback: assume success
        return True, resp

    except Exception as e:
        return False, str(e)
