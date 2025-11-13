# 📖 Dokumentasi: Login.py - Integrasi dengan Supabase

## 📋 Update Summary

File `Login.py` telah diupdate dari menggunakan JSON lokal menjadi **terhubung langsung dengan tabel "User" di Supabase**.

---

## 🔄 Perubahan Utama

### ❌ SEBELUM (JSON Lokal)
```python
# Menggunakan file users.json lokal
def authenticate(username: str, password: str) -> bool:
    users = _load_users()  # Baca dari JSON file
    pw_hash = _hash_password(password)
    return users.get(username) == pw_hash
```

### ✅ SESUDAH (Supabase)
```python
# Query langsung dari tabel User di Supabase
def authenticate(email: str, password: str) -> bool:
    response = supabase.table("User").select("*").eq("Email", email).eq("Password", password).execute()
    
    if response.data and len(response.data) > 0:
        user_data = response.data[0]
        # Simpan info user ke session state
        st.session_state.user_authenticated = True
        st.session_state.user = email
        st.session_state.user_id = user_data["ID_User"]
        st.session_state.user_role = user_data["Role"]
        st.session_state.user_name = user_data["Nama"]
        return True
    return False
```

---

## 🎯 Fitur Baru

### 1. Authentikasi dari Supabase
```python
# Login menggunakan email & password dari tabel User
if authenticate(email, password):
    st.success("Login berhasil!")
```

### 2. Session State Lengkap
Setelah login, tersimpan di session state:
- `user_authenticated` - Status login (True/False)
- `user` - Email user
- `user_id` - UUID dari tabel User
- `user_role` - Role user (User/Admin)
- `user_name` - Nama lengkap user

### 3. UI yang Lebih Baik
- Menampilkan nama user saat login
- Menampilkan role user
- Logout button yang jelas

---

## 🚀 Cara Menggunakan

### Basic Usage

```python
from Login import login_widget

# Render login widget
if login_widget():
    st.write("User sudah login!")
    st.write(f"Nama: {st.session_state.user_name}")
    st.write(f"Email: {st.session_state.user}")
    st.write(f"Role: {st.session_state.user_role}")
```

### Require Authentication

```python
from Login import require_auth

# Block halaman jika user belum login
require_auth("Login dulu untuk akses halaman ini")

st.write("Halaman ini hanya untuk user yang sudah login!")
```

### Logout

```python
from Login import logout

if st.button("Logout"):
    logout()
    st.rerun()
```

---

## 📊 Flow Login

```
┌─────────────────────────────┐
│  User Input (Email, Password) │
└────────────┬────────────────┘
             │
             ▼
    ┌────────────────────┐
    │ Validasi Input     │
    │ (tidak kosong?)    │
    └────────┬───────────┘
             │ OK
             ▼
┌────────────────────────────────────┐
│ Query Tabel "User" Supabase:        │
│                                    │
│ SELECT * FROM "User"               │
│ WHERE Email = ? AND Password = ?   │
└────────────┬───────────────────────┘
             │
       ┌─────┴─────┐
       │           │
      YES         NO
       │           │
       ▼           ▼
  ✅ FOUND    ❌ NOT FOUND
  Login Ok    Login Gagal
       │           │
       ▼           ▼
  Simpan ke    Show Error
  Session      Message
  State
       │
       ▼
  ✅ LOGIN SUKSES!
```

---

## 🔐 Session State yang Tersimpan

Setelah login berhasil, data berikut disimpan di `st.session_state`:

```python
st.session_state = {
    "user_authenticated": True,
    "user": "john@example.com",
    "user_id": "550e8400-e29b-41d4...",
    "user_role": "User",  # atau "Admin"
    "user_name": "John Doe"
}
```

Gunakan di mana saja dalam aplikasi:
```python
st.write(f"Welcome {st.session_state.user_name}!")
```

---

## 📝 Fungsi yang Tersedia

### 1. `authenticate(email: str, password: str) -> bool`
**Kegunaan**: Verify email & password terhadap tabel User

**Contoh**:
```python
if authenticate("john@example.com", "mypassword"):
    st.success("Login OK!")
else:
    st.error("Login GAGAL!")
```

**Yang dilakukan**:
- Query tabel User dengan email & password
- Jika ditemukan → simpan ke session state → return True
- Jika tidak ditemukan → return False

---

### 2. `logout() -> None`
**Kegunaan**: Logout user dan clear session state

**Contoh**:
```python
if st.button("Logout"):
    logout()
    st.rerun()
```

**Yang dilakukan**:
- Clear semua session state (user, user_id, user_role, etc)
- Sign out dari Supabase Auth (jika ada)

---

### 3. `login_widget() -> bool`
**Kegunaan**: Render UI login form

**Contoh**:
```python
if login_widget():
    st.success("Anda sudah login!")
```

**Yang dilakukan**:
- Jika belum login → tampilkan form (email, password)
- Jika sudah login → tampilkan info user + logout button
- Return True jika user sudah authenticated

---

### 4. `require_auth(message: str) -> None`
**Kegunaan**: Block halaman jika user belum login

**Contoh**:
```python
require_auth("Silakan login dulu!")
st.write("Halaman yang protected!")
```

**Yang dilakukan**:
- Cek apakah user_authenticated = True
- Jika tidak → tampilkan warning & stop execution
- Jika ya → lanjut ke code berikutnya

---

## 🔌 Integrasi dengan Web_Prediksi_Obesity.py

Edit `Web_Prediksi_Obesity.py` untuk menggunakan Login.py:

```python
# Import Login functions
from Login import login_widget, require_auth, logout

# Di bagian authentication gate:
if not st.session_state.user_authenticated:
    # Tampilkan login widget
    login_widget()
    st.stop()
else:
    # User sudah login - tampilkan main app
    st.title("Prediksi Obesitas")
    
    # Sidebar logout button
    with st.sidebar:
        st.write(f"Logged in as: {st.session_state.user_name}")
        if st.button("Logout"):
            logout()
            st.rerun()
    
    # Main app content...
```

---

## 🧪 Testing

### Test 1: Login dengan Email & Password Benar
```
1. Buka aplikasi
2. Masukkan email dari tabel User
3. Masukkan password yang benar
4. Klik Login
5. ✅ Harus muncul: "✅ Login berhasil!"
6. ✅ Harus tersimpan: user_id, user_role, user_name
```

### Test 2: Login dengan Password Salah
```
1. Buka aplikasi
2. Masukkan email yang terdaftar
3. Masukkan password SALAH
4. Klik Login
5. ✅ Harus muncul: "❌ Email atau password salah"
```

### Test 3: Login dengan Email Tidak Terdaftar
```
1. Buka aplikasi
2. Masukkan email yang TIDAK terdaftar
3. Masukkan password apapun
4. Klik Login
5. ✅ Harus muncul: "❌ Email atau password salah"
```

### Test 4: Logout
```
1. Login dulu dengan email & password
2. Lihat button "🚪 Logout"
3. Klik Logout button
4. ✅ Harus logout dan kembali ke login form
```

---

## ⚠️ Penting!

### 1. Email dan Password
- **Email**: Harus sama persis dengan di tabel User (case-sensitive)
- **Password**: Harus sama persis dengan di tabel User

### 2. Koneksi Supabase
- Pastikan `.env` file punya SUPABASE_URL dan SUPABASE_KEY yang benar
- Pastikan tabel "User" sudah punya data

### 3. Data di Session State
- Data dalam `st.session_state` hanya tersedia di Streamlit
- Jangan expose password di UI

### 4. RLS Policy
- Jika RLS enabled di tabel "User", pastikan policy mengizinkan SELECT

---

## 🔄 Komparasi: JSON vs Supabase

| Aspek | JSON Lokal | Supabase |
|-------|-----------|----------|
| Storage | File lokal (users.json) | Cloud database |
| Skalabilitas | Terbatas | Unlimited |
| Security | Rentan (plain text) | Aman (managed) |
| Multi-device | Tidak | Ya |
| Backup | Manual | Otomatis |
| Access | Lokal saja | Dari mana saja |
| Performance | Cepat | Tergantung internet |

---

## 📚 Referensi

Fungsi-fungsi lain yang tersedia:
- `get_supabase_client()` - Dapatkan Supabase client instance

File yang terkait:
- `Signup.py` - Sign up baru
- `Web_Prediksi_Obesity.py` - Main app
- `.env` - Konfigurasi Supabase

---

## ✅ Checklist Implementasi

- [x] Update Login.py untuk use Supabase
- [x] Tambah session state fields (user_id, user_role, user_name)
- [x] Update authenticate() function
- [x] Update logout() function
- [ ] Update Web_Prediksi_Obesity.py untuk import dari Login.py
- [ ] Test login dengan Supabase
- [ ] Test logout
- [ ] Test dengan multiple users

---

**Status**: ✅ **READY TO USE**

**Next Step**: Update `Web_Prediksi_Obesity.py` untuk menggunakan Login.py ini!
