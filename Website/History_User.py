import streamlit as st
from Connection.supabase_client import get_supabase_client
import pandas as pd

def get_user_history(user_id: str):

    client = get_supabase_client()
    user_name = st.session_state.get('user_name', 'Pengguna')

    try:
        input_resp = client.table("DataInput").select("*").eq("ID_User", user_id).execute()
        input_data = input_resp.data
        
        if not input_data:
            st.info(f"Tidak ada riwayat prediksi yang ditemukan untuk {user_name}.")
            return
        input_ids = [item.get('ID_Input') for item in input_data if item.get('ID_Input') is not None] #type: ignore
        if input_ids:
            pred_resp = client.table("Prediksi").select("*").in_("ID_DataInput", input_ids).execute()
            pred_data = pred_resp.data if hasattr(pred_resp, 'data') else []
            # Debugging: if no predictions found, show sample input id(s) and first input row
        else:
            pred_data = []

        df_inputs = pd.DataFrame(input_data)
        df_preds = pd.DataFrame(pred_data)

        # Ensure IDs are comparable types (cast to str) to avoid merge mismatch
        if 'ID_Input' in df_inputs.columns:
            df_inputs['ID_Input'] = df_inputs['ID_Input'].astype(str)
        if 'ID_DataInput' in df_preds.columns:
            df_preds['ID_DataInput'] = df_preds['ID_DataInput'].astype(str)

        if not df_preds.empty:
            # Merge the two dataframes: DataInput.ID_Input <-> Prediksi.ID_DataInput
            right = df_preds.copy()
            # avoid duplicating ID_User from predictions (if present)
            if 'ID_User' in right.columns:
                right = right.drop(columns=['ID_User'])

            cols_lower = {c.lower(): c for c in right.columns}
            if 'hasil_prediksi' not in cols_lower:
                # find candidate that contains 'hasil' or 'predik' or 'prediction'
                candidate = None
                for k in cols_lower:
                    if 'hasil' in k or 'predik' in k or 'prediction' in k:
                        candidate = cols_lower[k]
                        break
                if candidate:
                    right = right.rename(columns={candidate: 'Hasil_Prediksi'})
                else:
                    # no matching column found; create placeholder to make merge predictable
                    right['Hasil_Prediksi'] = None

            df_history = pd.merge(df_inputs, right, left_on='ID_Input', right_on='ID_DataInput', how='left')

            if 'Hasil_Prediksi' not in df_history.columns:
                # search for candidate columns in merged dataframe
                candidates = [c for c in df_history.columns if 'hasil' in c.lower() or 'predik' in c.lower() or 'prediction' in c.lower()]
                if candidates:
                    # pick first candidate and rename
                    df_history['Hasil_Prediksi'] = df_history[candidates[0]]
                else:
                    df_history['Hasil_Prediksi'] = None
        else:
            df_history = df_inputs.copy()
            # create Hasil_Prediksi column as placeholder if predictions not found
            df_history['Hasil_Prediksi'] = "N/A"

        # Normalize missing hasil values to 'N/A' for display
        try:
            df_history['Hasil_Prediksi'] = df_history['Hasil_Prediksi'].fillna('N/A')
        except Exception:
            pass

        # 5. Display the history
        st.subheader(f"Riwayat Prediksi untuk {user_name}")
        
        # Select and reorder columns for a cleaner display
        display_columns = [
            'Gender', 'Age', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC',
            'NCP', 'SCC', 'SMOKE', 'CH2O', 'family_history_with_overweight',
            'FAF', 'TUE', 'MTRANS', 'Hasil_Prediksi', 'created_at'
        ]
        
        # Filter dataframe to only columns that exist, in case some are missing
        existing_display_columns = [col for col in display_columns if col in df_history.columns]
        df_display = df_history[existing_display_columns]
        
        # Rename columns for better readability
        # rename created_at if it came from the Prediksi table
        if 'created_at_x' in df_display.columns:
            df_display = df_display.rename(columns={'created_at_x': 'Tanggal Prediksi'})
        elif 'created_at' in df_display.columns:
            df_display = df_display.rename(columns={'created_at': 'Tanggal Prediksi'})

        df_display = df_display.rename(columns={'Hasil_Prediksi': 'Hasil Prediksi'})

        st.dataframe(df_display)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengambil riwayat: {e}")

def history_page():

    st.title("Riwayat Prediksi Obesitas")

    # Add a button to navigate back to the prediction page
    if st.button("⬅️ Kembali ke Halaman Prediksi"):
        st.session_state.page = "prediksi"
        st.rerun()

    # Check if user is logged in by checking session_state
    if 'user_id' in st.session_state and st.session_state['user_id']:
        user_id = st.session_state['user_id']
        get_user_history(user_id)
    else:
        st.warning("Silakan login terlebih dahulu untuk melihat riwayat prediksi Anda.")
        st.info("Navigasi ke halaman Login untuk masuk atau mendaftar.")
        if st.button("Pergi ke Halaman Login"):
            st.session_state.page = "login"
            st.rerun()