
import streamlit as st
from Login import login_page
from Signup import signup_page
from Web_Prediksi_Obesity import run_prediction_app

# Initialize session state
if "user_authenticated" not in st.session_state:
    st.session_state.user_authenticated = False
if "page" not in st.session_state:
    st.session_state.page = "login"

# Page routing
if not st.session_state.user_authenticated:
    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "signup":
        signup_page()
else:
    run_prediction_app()
