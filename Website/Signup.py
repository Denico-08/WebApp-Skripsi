import streamlit as st
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Validate credentials
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")

# Clean up URL if it has duplicate parts
if "https://" in SUPABASE_URL and SUPABASE_URL.count("https://") > 1:
    # Take only the first valid URL part
    SUPABASE_URL = SUPABASE_URL.split("https://")[1]
    SUPABASE_URL = "https://" + SUPABASE_URL.split("https://")[0]


@st.cache_resource
def get_supabase_client() -> Client:
    """Initialize and return cached Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY) # type: ignore


def signup_widget() -> bool:
    """Render a signup form with Supabase integration.
    
    Returns True if signup was successful, False otherwise.
    """
    st.subheader("📝 Daftar Akun Baru")
    
    with st.form("signup_form"):
        email = st.text_input(
            "Email",
            placeholder="your-email@example.com",
            key="signup_email"
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Minimum 6 characters",
            key="signup_password"
        )
        password_confirm = st.text_input(
            "Konfirmasi Password",
            type="password",
            placeholder="Masukkan ulang password",
            key="signup_password_confirm"
        )
        full_name = st.text_input(
            "Nama Lengkap (Opsional)",
            placeholder="Nama Anda",
            key="signup_name"
        )
        
        submitted = st.form_submit_button("Daftar", use_container_width=True)
        
        if submitted:
            # Validation
            if not email or not password or not password_confirm:
                st.error("Email dan password harus diisi")
                return False
            
            if password != password_confirm:
                st.error("Password dan konfirmasi password tidak cocok")
                return False
            
            if len(password) < 6:
                st.error("Password minimal 6 karakter")
                return False
            
            # Attempt signup
            try:
                supabase = get_supabase_client()
                
                # Sign up user with email and password
                response = supabase.auth.sign_up({
                    "email": email,
                    "password": password,
                })
                
                if response.user:
                    st.success(f"✅ Pendaftaran berhasil! Silakan periksa email Anda untuk verifikasi.")
                    
                    # Simpan data user ke tabel User di Supabase (termasuk password)
                    try:
                        insert_data = {
                            "ID_User": response.user.id,
                            "Email": email,
                            "Password": password,
                            "Nama": full_name if full_name else email.split("@")[0],
                            "Role": "User"  # Default role untuk user baru
                        }
                        
                        result = supabase.table("User").insert(insert_data).execute()
                        
                        if result.data:
                            st.success("✅ Profil berhasil disimpan ke database!")
                        else:
                            st.success("✅ Data berhasil disimpan!")
                        
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"❌ Gagal menyimpan profil: {error_msg}")
                    
                    return True
                else:
                    st.error("Pendaftaran gagal. Silakan coba lagi.")
                    return False
                    
            except Exception as e:
                error_msg = str(e)
                if "already" in error_msg.lower():
                    st.error("Email ini sudah terdaftar. Silakan gunakan email lain.")
                elif "invalid" in error_msg.lower():
                    st.error("Email tidak valid.")
                else:
                    st.error(f"Terjadi kesalahan: {error_msg}")
                return False
    
    return False


def login_with_email(email: str, password: str) -> bool:
    """Authenticate user with email and password using Supabase.
    
    Returns True on success, False otherwise.
    """
    try:
        supabase = get_supabase_client()
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if response.user:
            st.session_state.user_authenticated = True
            st.session_state.user = response.user.email
            st.session_state.user_id = response.user.id
            return True
        else:
            return False
            
    except Exception as e:
        st.error(f"Login gagal: {str(e)}")
        return False


def logout() -> None:
    """Log out the current user."""
    try:
        supabase = get_supabase_client()
        supabase.auth.sign_out()
    except Exception:
        pass
    
    st.session_state.user_authenticated = False
    st.session_state.user = None
    st.session_state.user_id = None


def get_user_profile(user_id: str):
    """Ambil data profil user dari tabel User di Supabase.
    
    Args:
        user_id: UUID dari user (dari Supabase Auth)
    
    Returns:
        Dictionary berisi data user atau None jika tidak ditemukan
    """
    try:
        supabase = get_supabase_client()
        response = supabase.table("User").select("*").eq("ID_User", user_id).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        st.error(f"Gagal mengambil data profil: {e}")
        return None


def update_user_profile(user_id: str, data: dict) -> bool:
    """Update data profil user di Supabase.
    
    Args:
        user_id: UUID dari user
        data: Dictionary dengan field yang ingin diupdate (misal: {"Nama": "John Doe"})
    
    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        supabase = get_supabase_client()
        supabase.table("User").update(data).eq("ID_User", user_id).execute()
        st.success("✅ Profil berhasil diupdate!")
        return True
    except Exception as e:
        st.error(f"Gagal update profil: {e}")
        return False


def require_auth(message: str = "Silakan login terlebih dahulu") -> None:
    """Block execution with a message if user is not authenticated."""
    if not st.session_state.get("user_authenticated"):
        st.warning(message)
        st.stop()


if __name__ == "__main__":
    # Test UI when running this file directly
    st.set_page_config(page_title="Signup Test", layout="centered")
    st.title("Signup Module Test")
    
    if "user_authenticated" not in st.session_state:
        st.session_state.user_authenticated = False
        st.session_state.user = None
        st.session_state.user_id = None
    
    if st.session_state.user_authenticated:
        st.success(f"Logged in as: {st.session_state.user}")
        if st.button("Logout"):
            logout()
            st.rerun()
    else:
        tab1, tab2 = st.tabs(["Login", "Signup"])
        
        with tab1:
            st.subheader("Login")
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                if login_with_email(email, password):
                    st.success("Login berhasil!")
                    st.rerun()
                else:
                    st.error("Email atau password salah")
        
        with tab2:
            signup_widget()
