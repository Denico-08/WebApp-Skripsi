import streamlit as st
import os
from supabase_client import get_supabase_client

def authenticate(email: str, password: str) -> bool:
    
    if not email or not password:
        st.error("❌ Email dan password harus diisi")
        return False
    
    try:
        supabase = get_supabase_client()
        
        # Step 1: Query ALL users to see what's in the table
        all_users_response = supabase.table("User").select("Email, Password, Nama, Role").execute()
        
        #Step 2: Try exact match query
        response = supabase.table("User").select("*").eq("Email", email).eq("Password", password).execute()
        
        
        if response.data and len(response.data) > 0:
            user_data = response.data[0]
            
            st.session_state.user_authenticated = True
            st.session_state.user = email
            st.session_state.user_id = user_data["ID_User"] if isinstance(user_data, dict) else None
            st.session_state.user_role = user_data["Role"] if isinstance(user_data, dict) else "User"
            st.session_state.user_name = user_data["Nama"] if isinstance(user_data, dict) else email
            return True
        else:
            available_emails = [u["Email"] if isinstance(u, dict) else "N/A" for u in (all_users_response.data or [])]
            
            return False
            
    except Exception as e:
        st.error(f"❌ Error saat login: {str(e)}")
        
        with st.expander("❌ ERROR DETAILS"):
            st.write(f"**Error Type:** {type(e).__name__}")
            st.write(f"**Error Message:** {str(e)}")
        
        return False


def logout() -> None:
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


def require_auth(message: str = "Silakan login terlebih dahulu:") -> None:
    if not st.session_state.get("user_authenticated"):
        st.warning(message)
        st.stop()


def login_page() -> None:
    """Renders the login page and handles authentication logic."""
    if "user_authenticated" not in st.session_state:
        st.session_state.user_authenticated = False
        st.session_state.user = None
        st.session_state.user_id = None
        st.session_state.user_role = None
        st.session_state.user_name = None

    # This part should not be reached if the user is already authenticated,
    # as the main app router would have already directed them away.
    # However, it's good practice to keep it.
    if st.session_state.user_authenticated:
        st.success(f"✅ Anda sudah login sebagai: {st.session_state.user_name}")
        st.info("Mengalihkan ke halaman utama...")
        # In a multi-page app, the main router handles the redirection.
        # A rerun is enough to trigger the check in the main app.
        st.rerun()
        return

    # User belum login - tampilkan form login
    st.set_page_config(page_title="Login", layout="centered")
    st.title("🔑 Selamat Datang!")
    st.subheader("Silakan Login untuk Melanjutkan")
    
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
                    # Set session state and rerun to let the main app router redirect
                    st.session_state.user_authenticated = True
                    st.success(f"✅ Login berhasil! Selamat datang, {st.session_state.user_name}!")
                    st.rerun()
                else:
                    st.error("❌ Email atau password salah")

    st.markdown("---")
    st.write("Belum punya akun?")
    if st.button("Buat Akun Baru (Sign Up)", use_container_width=True):
        st.session_state.page = "signup"
        st.rerun()
