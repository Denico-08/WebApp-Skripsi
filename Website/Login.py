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
    SUPABASE_URL = "https://" + SUPABASE_URL.split("https://")[1].split("https://")[0]


@st.cache_resource
def get_supabase_client() -> Client:
    """Initialize and return cached Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)  # type: ignore


def authenticate(email: str, password: str) -> bool:
    
    if not email or not password:
        st.error("❌ Email dan password harus diisi")
        return False
    
    try:
        supabase = get_supabase_client()
        
        # DEBUG: Show what we're searching for
        print(f"DEBUG: Searching for Email='{email}', Password='{password}'")
        
        # Step 1: Query ALL users to see what's in the table
        all_users_response = supabase.table("User").select("Email, Password, Nama, Role").execute()
        print(f"DEBUG: All users in table: {all_users_response.data}")
        
        # Step 2: Try exact match query
        response = supabase.table("User").select("*").eq("Email", email).eq("Password", password).execute()
        
        print(f"DEBUG: Query response: {response.data}")
        print(f"DEBUG: Response data exists: {response.data is not None}")
        print(f"DEBUG: Response data length: {len(response.data) if response.data else 0}")
        
        if response.data and len(response.data) > 0:
            # Credentials valid - simpan user info ke session state
            user_data = response.data[0]
            print(f"DEBUG: User found! Data: {user_data}")
            
            st.session_state.user_authenticated = True
            st.session_state.user = email
            st.session_state.user_id = user_data["ID_User"] if isinstance(user_data, dict) else None
            st.session_state.user_role = user_data["Role"] if isinstance(user_data, dict) else "User"
            st.session_state.user_name = user_data["Nama"] if isinstance(user_data, dict) else email
            return True
        else:
            # No match found - show available emails
            available_emails = [u["Email"] if isinstance(u, dict) else "N/A" for u in (all_users_response.data or [])]
            print(f"DEBUG: No match found. Available emails: {available_emails}")
            
            with st.expander("🔍 DEBUG INFO"):
                st.write(f"**Email yang dicari:** {email}")
                st.write(f"**Password yang dicari:** {password}")
                st.write(f"**Email yang tersedia di database:** {available_emails}")
                st.write(f"**Semua data users:**")
                st.dataframe(all_users_response.data)
            
            return False
            
    except Exception as e:
        print(f"DEBUG ERROR: {type(e).__name__}: {str(e)}")
        st.error(f"❌ Error saat login: {str(e)}")
        
        with st.expander("❌ ERROR DETAILS"):
            st.write(f"**Error Type:** {type(e).__name__}")
            st.write(f"**Error Message:** {str(e)}")
        
        return False


def logout() -> None:
    """Log the current user out (modifies Streamlit session state)."""
    try:
        supabase = get_supabase_client()
        supabase.auth.sign_out()
    except Exception:
        pass
    
    st.session_state.user_authenticated = False
    st.session_state.user = None
    st.session_state.user_id = None
    st.session_state.user_role = None
    st.session_state.user_name = None


def require_auth(message: str = "Silakan login terlebih dahulu") -> None:
    """Block execution with a message if the user is not authenticated.

    Call this at the top of pages that require login.
    """
    if not st.session_state.get("user_authenticated"):
        st.warning(message)
        st.stop()


def login_widget() -> bool:
    """Render a login widget yang terhubung dengan tabel "User" Supabase.

    - If already logged in, shows user info and a logout button.
    - If not logged in, shows email/password input and Login button.

    Returns True if the user is (now) authenticated, False otherwise.
    """
    if "user_authenticated" not in st.session_state:
        st.session_state.user_authenticated = False
        st.session_state.user = None
        st.session_state.user_id = None
        st.session_state.user_role = None
        st.session_state.user_name = None

    if st.session_state.user_authenticated:
        # User sudah login - tampilkan info
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.success(f"✅ Logged in: {st.session_state.user_name}")
        
        with col2:
            st.write(f"*{st.session_state.user}* ({st.session_state.user_role})")
        
        with col3:
            if st.button("🚪 Logout", use_container_width=True):
                logout()
                st.rerun()
        
        return True

    # User belum login - tampilkan form login
    st.subheader("🔑 Login")
    
    with st.form("login_form"):
        email = st.text_input(
            "Email",
            placeholder="your-email@example.com",
            key="login_email"
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Your password",
            key="login_password"
        )
        submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("❌ Email dan password harus diisi")
            else:
                if authenticate(email, password):
                    st.success(f"✅ Login berhasil! Selamat datang, {st.session_state.user_name}!")
                    st.rerun()
                    return True
                else:
                    st.error("❌ Email atau password salah")

    return False


if __name__ == "__main__":
    # Quick manual test UI when running this file directly
    st.set_page_config(page_title="Login Test", layout="centered")
    st.title("Login Module Test")
    
    login_widget()
