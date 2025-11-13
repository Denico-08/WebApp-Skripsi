# 🔗 Integrasi Login.py dengan Web_Prediksi_Obesity.py

## Status Update

✅ **Login.py** sudah ter-update dan terhubung dengan tabel "User" Supabase
✅ **Signup.py** sudah ter-update dan menyimpan ke tabel "User" Supabase

🔄 **Next**: Integrate Login.py ke Web_Prediksi_Obesity.py

---

## Perubahan yang Diperlukan di Web_Prediksi_Obesity.py

### 1. Ganti Import (Baris ~16)

**❌ SEBELUM:**
```python
from Signup import signup_widget, login_with_email, logout, require_auth
```

**✅ SESUDAH:**
```python
from Signup import signup_widget
from Login import login_widget, logout, require_auth
```

---

### 2. Ganti Authentication Gate (Baris ~30-75)

**❌ SEBELUM (Gunakan Supabase Auth):**
```python
if not st.session_state.user_authenticated:
    st.title("🔐 Sistem Prediksi Obesitas - Login Required")

    if "show_signup" not in st.session_state:
        st.session_state.show_signup = False

    if not st.session_state.show_signup:
        st.subheader("🔑 Login")
        email = st.text_input("Email", key="login_email", placeholder="your-email@example.com")
        password = st.text_input("Password", type="password", key="login_password", placeholder="Your password")

        if st.button("Login", use_container_width=True, type="primary"):
            if login_with_email(email, password):
                st.success("✅ Login berhasil! Mengalihkan...")
                st.rerun()
            else:
                st.error("❌ Email atau password salah")

        st.markdown("---")
        st.write("Belum punya akun?")
        if st.button("Doesn't have accounts", use_container_width=True):
            st.session_state.show_signup = True
            st.rerun()

    else:
        st.subheader("✍️ Daftar Akun Baru")
        signed = signup_widget()
        if signed:
            st.success("Pendaftaran berhasil. Silakan login menggunakan akun Anda.")
            st.session_state.show_signup = False
            st.rerun()

        if st.button("Kembali ke Login"):
            st.session_state.show_signup = False
            st.rerun()

    st.stop()
```

**✅ SESUDAH (Gunakan Login.py):**
```python
if not st.session_state.user_authenticated:
    st.title("🔐 Sistem Prediksi Obesitas")
    
    # Tab untuk Login dan Sign Up
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
    
    with tab1:
        login_widget()
    
    with tab2:
        st.subheader("Daftar Akun Baru")
        if signup_widget():
            st.success("Pendaftaran berhasil! Silakan login dengan akun Anda.")
            st.rerun()
    
    st.stop()
```

---

### 3. Sidebar Info Update (Baris ~73+)

**❌ SEBELUM:**
```python
with st.sidebar:
    st.write(f"**Logged in as:** {st.session_state.user}")
    if st.button("Logout", key="logout_button"):
        logout()
        st.rerun()
```

**✅ SESUDAH:**
```python
with st.sidebar:
    st.write("---")
    st.write(f"**👤 Nama:** {st.session_state.user_name}")
    st.write(f"**📧 Email:** {st.session_state.user}")
    st.write(f"**👥 Role:** {st.session_state.user_role}")
    st.write("---")
    
    if st.button("🚪 Logout", use_container_width=True, key="logout_button"):
        logout()
        st.rerun()
```

---

## 📋 Complete Code Example

Berikut adalah skeleton code yang bisa Anda gunakan:

```python
import streamlit as st
import pandas as pd
import numpy as np

# ===== IMPORTS =====
from Signup import signup_widget
from Login import login_widget, logout, require_auth

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Prediksi Obesitas", layout="wide")

# ===== INIT SESSION STATE =====
if "user_authenticated" not in st.session_state:
    st.session_state.user_authenticated = False
    st.session_state.user = None
    st.session_state.user_id = None
    st.session_state.user_name = None
    st.session_state.user_role = None

# ===== AUTHENTICATION GATE =====
if not st.session_state.user_authenticated:
    st.title("🔐 Sistem Prediksi Obesitas")
    
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
    
    with tab1:
        login_widget()
    
    with tab2:
        st.subheader("Daftar Akun Baru")
        if signup_widget():
            st.success("Pendaftaran berhasil! Silakan login.")
            st.rerun()
    
    st.stop()

# ===== MAIN APPLICATION (User Sudah Login) =====

# Sidebar
with st.sidebar:
    st.write("---")
    st.write(f"👤 **{st.session_state.user_name}**")
    st.write(f"📧 {st.session_state.user}")
    st.write(f"👥 Role: **{st.session_state.user_role}**")
    st.write("---")
    
    if st.button("🚪 Logout", use_container_width=True):
        logout()
        st.rerun()

# Main Content
st.title("🏥 Prediksi Tingkat Obesitas & Rekomendasi Perubahan")
st.markdown(f"Selamat datang, **{st.session_state.user_name}**! 👋")

# ... Rest of your app ...
```

---

## 🔄 Session State Variables

Setelah login, gunakan variables ini:

```python
# User Identity
st.session_state.user_authenticated  # bool - Apakah user sudah login?
st.session_state.user                # str - Email user
st.session_state.user_id             # str - UUID dari tabel User
st.session_state.user_name           # str - Nama lengkap user
st.session_state.user_role           # str - Role user ('User' atau 'Admin')

# Contoh penggunaan:
st.write(f"Hello {st.session_state.user_name}!")

if st.session_state.user_role == "Admin":
    st.write("You have admin access!")
```

---

## ✅ Perbedaan Login Methods

### Method 1: Supabase Auth (Signup.py)
- Keuntungan: Secure, managed password
- Kekurangan: Memerlukan email verification
- Gunakan untuk: Public app dengan email verification

### Method 2: Direct Database Query (Login.py)
- Keuntungan: Simple, cepat, langsung dari database
- Kekurangan: Password plain text di database
- Gunakan untuk: Internal app, development, simple workflow

**Aplikasi Anda:** Gunakan **Method 2 (Login.py)** karena password disimpan di tabel User

---

## 🎯 Implementation Steps

### Step 1: Update Imports
Ganti import di Web_Prediksi_Obesity.py:
```python
# Ganti:
from Signup import signup_widget, login_with_email, logout, require_auth

# Dengan:
from Signup import signup_widget
from Login import login_widget, logout, require_auth
```

### Step 2: Replace Authentication Gate
Ganti section authentication gate dengan code yang lebih simple (lihat di atas)

### Step 3: Update Sidebar
Ganti sidebar info section dengan code yang baru

### Step 4: Test
```bash
streamlit run Web_Prediksi_Obesity.py
```

---

## 📊 Architecture

```
Web_Prediksi_Obesity.py (Main App)
    │
    ├─→ Login.py (Login functions)
    │    │
    │    └─→ Supabase (Query tabel User)
    │
    └─→ Signup.py (Sign Up functions)
         │
         └─→ Supabase (Insert ke tabel User)
```

---

## ⚙️ Configuration

Pastikan `.env` file Anda sudah ada:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

---

## 🧪 Testing Checklist

- [ ] Login page muncul
- [ ] Tab Login dan Sign Up tersedia
- [ ] Login dengan email & password yang terdaftar
- [ ] ✅ Login berhasil, redirect ke main app
- [ ] ❌ Login dengan password salah → error
- [ ] ❌ Login dengan email tidak terdaftar → error
- [ ] Sidebar menampilkan user info (nama, email, role)
- [ ] Logout button berfungsi
- [ ] Sign up form berfungsi
- [ ] Data baru tersimpan ke tabel User

---

## 🐛 Troubleshooting

### Masalah: "Import Error - login_widget not found"
**Solusi**: Pastikan Login.py ada di folder yang sama dengan Web_Prediksi_Obesity.py

### Masalah: "SUPABASE_URL and SUPABASE_KEY must be set"
**Solusi**: Buat file `.env` di folder parent dengan nilai yang benar

### Masalah: "Email atau password salah" padahal benar
**Solusi**: 
- Cek case sensitivity
- Pastikan data ada di tabel User
- Lihat di Supabase Console apakah ada data

### Masalah: Login berhasil tapi user_name kosong
**Solusi**: Pastikan kolom "Nama" di tabel User terisi data

---

## 📈 Next Steps (Optional)

1. **Admin Panel** - Buat halaman khusus untuk role='Admin'
2. **User Management** - Buat halaman untuk manage users (edit, delete)
3. **Activity Log** - Simpan log login/logout ke tabel baru
4. **Password Reset** - Implement forgot password functionality
5. **Email Verification** - Integrate dengan Supabase Auth email verification

---

## 📚 Files Reference

- **Login.py** - Login functions (sudah ter-update)
- **Signup.py** - Signup functions (sudah ter-update)
- **Web_Prediksi_Obesity.py** - Main app (perlu update)
- **LOGIN_DOCUMENTATION.md** - Dokumentasi Login.py
- **DEBUGGING_DATA_NOT_SAVED.md** - Troubleshooting Sign Up
- **.env** - Konfigurasi Supabase

---

**Status**: ✅ **READY FOR INTEGRATION**

Silakan update Web_Prediksi_Obesity.py dengan code di atas!
