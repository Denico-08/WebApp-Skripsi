import streamlit as st
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file in the parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- Supabase Client Initialization ---
@st.cache_resource
def get_supabase_client() -> Client:
    """
    Initializes and returns a cached Supabase client.
    Raises ValueError if Supabase credentials are not set.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
    
    # Clean up URL if it has duplicate parts
    clean_url = SUPABASE_URL
    if "https://" in clean_url and clean_url.count("https://") > 1:
        clean_url = "https://" + clean_url.split("https://")[-1]
        
    return create_client(clean_url, SUPABASE_KEY)

# --- Signup Page UI and Logic ---
def signup_page():
    """
    Renders the complete signup page with a form and links to the login page.
    """
    st.set_page_config(page_title="Daftar Akun", layout="centered")
    st.title("📝 Daftar Akun Baru")
    st.write("Silakan isi form di bawah ini untuk membuat akun baru.")

    # Button to go back to Login page
    if st.button("‹ Kembali ke Login"):
        st.session_state.page = "login"
        st.rerun()

    with st.form("signup_form"):
        email = st.text_input(
            "Email",
            placeholder="your-email@example.com",
            key="signup_email"
        )
        full_name = st.text_input(
            "Nama Lengkap",
            placeholder="Nama Anda",
            key="signup_name"
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Minimal 6 karakter",
            key="signup_password"
        )
        password_confirm = st.text_input(
            "Konfirmasi Password",
            type="password",
            placeholder="Masukkan ulang password",
            key="signup_password_confirm"
        )
        
        submitted = st.form_submit_button("Daftar", use_container_width=True)
        
        if submitted:
            # --- Input Validation ---
            if not all([email, full_name, password, password_confirm]):
                st.error("❌ Semua field harus diisi.")
                return

            if password != password_confirm:
                st.error("❌ Password dan konfirmasi password tidak cocok.")
                return
            
            if len(password) < 6:
                st.error("❌ Password minimal harus 6 karakter.")
                return

            # --- Signup Process ---
            try:
                supabase = get_supabase_client()
                
                # Step 1: Sign up the user in Supabase Auth
                auth_response = supabase.auth.sign_up({
                    "email": email,
                    "password": password,
                })

                # Step 2: If auth is successful, insert profile into the 'User' table
                if auth_response.user:
                    user_id = auth_response.user.id
                    
                    insert_data = {
                        "ID_User": user_id,
                        "Email": email,
                        "Password": password,  # WARNING: Storing plain password is not secure
                        "Nama": full_name,
                        "Role": "User"
                    }
                    
                    # Insert profile data into the public 'User' table
                    supabase.table("User").insert(insert_data).execute()

                    st.success("✅ Pendaftaran berhasil! Silakan kembali ke halaman login untuk masuk.")
                    # The button to go back to login is already present outside the form.

                else:
                    st.error("Pendaftaran gagal. Silakan coba lagi.")

            except Exception as e:
                error_msg = str(e)
                if "user already registered" in error_msg.lower():
                    st.error("❌ Email ini sudah terdaftar. Silakan gunakan email lain atau login.")
                elif "invalid" in error_msg.lower():
                    st.error("❌ Email atau password tidak valid.")
                else:
                    st.error(f"Terjadi kesalahan: {error_msg}")
