# 🔍 DEBUGGING: Login "Email atau Password Salah" Issue

## 🚨 Masalah

Anda memasukkan email dan password yang **benar**, tapi muncul error:
> ❌ Email atau password salah

---

## 🔧 Penyebab Kemungkinan

### 1. **Whitespace (Spasi) - PALING UMUM!**

**Masalah:**
```
Database:  "john@example.com"
Input:     "john@example.com " (ada spasi di akhir)
           ↑ TIDAK COCOK!
```

**Solusi:**
- Input akan di-trim otomatis (sudah ditambahkan di debug code)
- Atau manual: `.strip()` saat input

### 2. **Case Sensitivity**

**Masalah:**
```
Database:  "John@Example.COM"
Input:     "john@example.com"
           ↑ Case BERBEDA!
```

**Solusi:**
- Gunakan `.lower()` untuk normalize email

### 3. **Data Tidak Ada di Tabel**

**Masalah:**
```
Email yang diinput: test@example.com
Tapi di database tidak ada email ini!
```

**Solusi:**
- Verifikasi data ada di Supabase Console
- Atau sign up dulu

### 4. **RLS Policy Menghalangi Query**

**Masalah:**
```
RLS enabled + policy restrict access
Query gagal / return empty
```

**Solusi:**
- Disable RLS untuk debug
- Atau buat policy yang tepat

### 5. **Connection Error**

**Masalah:**
```
Tidak bisa connect ke Supabase
- SUPABASE_URL salah
- SUPABASE_KEY salah
- Internet mati
```

**Solusi:**
- Cek `.env` file
- Cek internet connection

---

## 🧪 Debugging Steps

### Step 1: Lihat Debug Info di Browser

Setelah input email & password yang benar:

1. Refresh page
2. Login lagi
3. Lihat section "🔍 DEBUG INFO" (akan muncul jika gagal)
4. Expand dan lihat:
   - Email yang dicari
   - Password yang dicari
   - **Email yang tersedia di database**
   - Semua data users (dalam tabel)

### Step 2: Lihat Console Output

Buka terminal Streamlit Anda (tempat jalankan `streamlit run`):

Cari baris yang berisi:
```
DEBUG: Searching for Email='...', Password='...'
DEBUG: All users in table: [...]
DEBUG: Query response: [...]
DEBUG: No match found. Available emails: [...]
```

### Step 3: Cek Data di Supabase Console

1. Buka https://supabase.com/dashboard
2. Pilih project Anda
3. Buka Table Editor → Tabel "User"
4. Lihat data:
   - Ada berapa rows?
   - Email kolom berisi apa?
   - Password kolom berisi apa?

---

## 🛠️ Common Fixes

### Fix 1: Strip Whitespace

Update login form:
```python
email = st.text_input("Email").strip()
password = st.text_input("Password", type="password").strip()
```

### Fix 2: Normalize Email (lowercase)

Update authenticate function:
```python
def authenticate(email: str, password: str) -> bool:
    email = email.lower().strip()  # ← ADD THIS
    # ... rest of code
```

### Fix 3: Check Email Exists

Tambah query tanpa password dulu:
```python
# Cek apakah email ada
email_query = supabase.table("User").select("*").eq("Email", email.lower()).execute()

if email_query.data:
    # Email ada, cek password
    full_query = supabase.table("User").select("*").eq("Email", email.lower()).eq("Password", password).execute()
    if full_query.data:
        # Password cocok
        return True
    else:
        st.error("❌ Password salah")
        return False
else:
    st.error("❌ Email tidak terdaftar")
    return False
```

### Fix 4: Disable RLS untuk Debug

Di Supabase Console SQL Editor:
```sql
-- Disable RLS temporarily
ALTER TABLE public."User" DISABLE ROW LEVEL SECURITY;

-- Test login
-- If works → RLS policy masalah
-- If still not works → bukan RLS

-- Re-enable RLS
ALTER TABLE public."User" ENABLE ROW LEVEL SECURITY;
```

---

## 🎯 Debugging Workflow

```
Login gagal?
    │
    ├─→ Lihat DEBUG INFO di UI
    │   ├─ Email tersedia? ✓/✗
    │   ├─ Password match? ✓/✗
    │   └─ Data structure OK? ✓/✗
    │
    ├─→ Lihat console output
    │   ├─ Search email: ...
    │   ├─ Available emails: [...]
    │   └─ Error message: ...
    │
    ├─→ Cek Supabase Console
    │   ├─ Tabel User ada? ✓/✗
    │   ├─ Data ada? ✓/✗
    │   ├─ Email format? 
    │   └─ Password format?
    │
    └─→ Test RLS
        ├─ Disable RLS
        ├─ Try login
        ├─ Works? → RLS issue
        └─ Not works? → Data issue
```

---

## 📋 Checklist Debugging

```
[ ] DEBUG INFO di UI sudah dilihat?
    [ ] Email tersedia di list?
    [ ] Data format OK?

[ ] Console output sudah dicek?
    [ ] Ada error message?
    [ ] Email ada di available list?

[ ] Supabase Console sudah dicek?
    [ ] Tabel User ada?
    [ ] Data ada?
    [ ] Email format (lowercase/uppercase)?
    [ ] Password text betul?

[ ] Testing RLS?
    [ ] RLS disabled → bisa login?
    [ ] RLS enabled → ada policy?

[ ] Testing whitespace?
    [ ] Input sudah .strip()?
    [ ] Database punya spasi?

[ ] Testing case?
    [ ] Email lowercase?
    [ ] Password exact match?
```

---

## 🧬 Updated authenticate() Function

Sudah disertakan di Login.py dengan:
- ✅ Debug print statements
- ✅ DEBUG INFO expansion di UI
- ✅ Tampilkan semua users (untuk comparison)
- ✅ Error details expansion

**Cara menggunakan:**
1. Login dengan email & password
2. Jika gagal, lihat "🔍 DEBUG INFO" section
3. Expand dan cek available emails
4. Buka terminal dan lihat console output (DEBUG messages)
5. Compare dengan database di Supabase

---

## 🔄 Quick Fix Steps

1. **Test dengan data yang pasti benar:**
   ```
   Email: test@example.com
   Password: Test123
   ```

2. **Jika masih gagal, expand DEBUG INFO:**
   - Lihat available emails
   - Apakah email Anda ada?
   - Format apa?

3. **Jika email tidak ada:**
   - Sign up dengan email baru
   - Verify data tersimpan
   - Try login

4. **Jika email ada tapi password tidak cocok:**
   - Cek password di database (Supabase Console)
   - Apakah exact match?

5. **Jika masih error:**
   - Disable RLS di Supabase
   - Try login
   - Jika berhasil → RLS policy problem
   - Jika tetap gagal → connection problem

---

## 📝 Sample Output

Jika berhasil login:
```
✅ Login berhasil!
```

Jika gagal, akan muncul:
```
❌ Email atau password salah

🔍 DEBUG INFO (expandable)
├─ Email yang dicari: john@example.com
├─ Password yang dicari: MyPassword123
├─ Email yang tersedia di database: 
│  - test@example.com
│  - admin@example.com
│  - john@example.com    ← YOUR EMAIL!
└─ Semua data users:
   [TABLE dengan ID_User, Email, Password, Role, Nama]
```

---

## 💡 Pro Tips

1. **Copy-paste email dari Supabase Console**
   - Pastikan format exact
   
2. **Tidak ada typo di password**
   - Case sensitive!
   - Spasi juga dihitung!

3. **Check terminal output**
   - DEBUG messages ada di sini
   - Lihat Available emails

4. **Validate RLS**
   - Try disable untuk test
   - Jika login works → RLS issue
   - Jika still fails → data issue

---

## 🆘 Jika Masih Tidak Berhasil

Share informasi ini:
1. Screenshot dari "🔍 DEBUG INFO" section
2. Console output (dari terminal Streamlit)
3. Data di Supabase Console (screenshot tabel User)
4. Error message lengkap

Maka saya bisa bantu lebih spesifik! 🚀

---

**Status**: ✅ Debug mode sudah ditambahkan ke Login.py

**Next Action**: 
1. Login dengan email & password Anda
2. Expand "🔍 DEBUG INFO" jika gagal
3. Share hasilnya!
