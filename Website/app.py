import streamlit as st
from User import User
from Web_Prediksi_Obesity import run_prediction_app

# Initialize session state
if "user_authenticated" not in st.session_state:
    st.session_state.user_authenticated = False
if "page" not in st.session_state:
    st.session_state.page = "login"

# Page routing
if not st.session_state.user_authenticated:
    if st.session_state.page == "login":
        User.render_login_page()
    elif st.session_state.page == "signup":
        User.render_signup_page()
    else:
        User.render_login_page() # Default to login
else:
    # If authenticated, route based on user role and page state
    user_role = st.session_state.get('user_role', None)
    page = st.session_state.get('page')

    if user_role == "Admin":
        from Admin import admin_page_
        admin_page_()
    else:  # For regular users
        if page == "riwayat":
            current_user_id = st.session_state.get('user_id')
            current_user_name = st.session_state.get('user_name')
            
            # Buat instance User dan set ID secara manual
            user_instance = User(name=current_user_name)
            user_instance.id = current_user_id
            user_instance.render_history_page()

        elif page == "prediksi":
            run_prediction_app()
        else:
            # Default to the prediction app for any other state
            st.session_state.page = "prediksi"
            run_prediction_app()
