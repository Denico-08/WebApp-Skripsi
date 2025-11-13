# 🔍 DEBUGGING: Mengapa Data Tidak Tersimpan di Tabel "User"

## Struktur Tabel "User" (Sudah Dikonfirmasi)

```
Kolom        | Tipe Data | Keterangan
─────────────┼───────────┼─────────────────────
ID_User      | UUID      | Primary Key
Email        | VARCHAR   | Unique
Password     | VARCHAR   | Password user
Role         | ENUM      | 'User' atau 'Admin'
Nama         | TEXT      | Nama lengkap user
```

---

## 🚨 Kemungkinan Penyebab Data Tidak Tersimpan

### 1. ✅ RLS (Row Level Security) Menghalangi INSERT

**Tanda-tanda:**
- Sign up berhasil (pesan ✅ muncul)
- Tapi data tidak ada di tabel
- Tidak ada error message

**Solusi:**
1. Buka Supabase Dashboard → Tabel "User"
2. Klik tombol "RLS" (di sudut kanan)
3. Jika **RLS is enabled**, coba disable untuk testing:
   ```
   [ Disable ] ← Klik ini
   ```
4. Test sign up lagi
5. Jika data tersimpan → RLS memang masalahnya!

**Jika RLS yang masalah, jalankan SQL ini:**
```sql
-- Disable RLS untuk testing
ALTER TABLE public."User" DISABLE ROW LEVEL SECURITY;

-- Test sign up

-- Kemudian enable kembali dengan policy yang benar
ALTER TABLE public."User" ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Enable insert for authenticated users" ON public."User"
FOR INSERT WITH CHECK (true);

CREATE POLICY "Enable select for all users" ON public."User"
FOR SELECT USING (true);
```

---

### 2. ✅ Error Tapi Pesan Tidak Ditampilkan

**Solusi:**
Update `Signup.py` dengan debugging yang lebih detail:

```python
try:
    insert_data = {
        "ID_User": response.user.id,
        "Email": email,
        "Password": password,
        "Nama": full_name if full_name else email.split("@")[0],
        "Role": "User"
    }
    
    print(f"DEBUG: Inserting data: {insert_data}")  # Log ke console
    
    result = supabase.table("User").insert(insert_data).execute()
    print(f"DEBUG: Response: {result}")  # Log response
    
except Exception as e:
    print(f"DEBUG ERROR: {type(e).__name__}: {str(e)}")  # Log error
    st.error(f"Error: {str(e)}")
```

---

### 3. ✅ Data Type Mismatch

**Cek yang perlu dilakukan:**

| Kolom | Expected | Yang Dikirim | Status |
|-------|----------|-------------|--------|
| ID_User | UUID string | response.user.id | ✅ OK |
| Email | VARCHAR | email input | ✅ OK |
| Password | VARCHAR | password input | ✅ OK |
| Nama | TEXT | full_name atau email | ✅ OK |
| Role | ENUM ('User'/'Admin') | 'User' | ✅ OK |

---

### 4. ✅ Email Duplicate

**Tanda-tanda:**
- Mencoba sign up dengan email yang sama 2x
- Error: "duplicate key value violates unique constraint"

**Solusi:**
Gunakan email baru setiap kali test

---

## 🧪 Testing Step-by-Step

### Step 1: Cek RLS Status
```
1. Buka Supabase Dashboard
2. Pilih Tabel "User"
3. Lihat button "RLS" di sebelah kanan
4. Apakah RLS is ENABLED atau DISABLED?
```

### Step 2: Test Insert Manual di Supabase

Buka SQL Editor dan jalankan:

```sql
INSERT INTO public."User" (
  ID_User,
  Email,
  Password,
  Nama,
  Role
) VALUES (
  gen_random_uuid(),
  'test_manual@example.com',
  'TestPassword123',
  'Test User',
  'User'
);
```

**Hasil yang diharapkan:**
- ✅ SUCCESS: Data tersimpan
- ❌ ERROR: Lihat pesan error-nya

---

### Step 3: Test via Application

1. Restart Streamlit app
2. Sign up dengan email baru: `test@example.com`
3. Tunggu pesan "✅ Profil berhasil disimpan"
4. Buka Supabase Table Editor
5. Cek apakah ada row baru dengan email `test@example.com`

---

## 📊 Checklist Debugging

```
[ ] Cek RLS status (enabled/disabled?)
[ ] Test manual insert di SQL Editor
[ ] Lihat error message di console
[ ] Restart Streamlit app
[ ] Test sign up dengan email baru
[ ] Verifikasi di Supabase Table Editor
[ ] Cek kolom Role (apakah ada default value?)
```

---

## 🔧 Update Code yang Sudah Dilakukan

✅ **Baris 87**: Menambahkan `"Role": "User"` ke insert data

Data yang akan dikirim sekarang:
```python
{
    "ID_User": "550e8400-e29b-41d4...",
    "Email": "user@example.com",
    "Password": "user_password_123",
    "Nama": "User Name",
    "Role": "User"  ← ADDED
}
```

---

## 📝 Rekomendasi

### Jika RLS Adalah Masalahnya:

**Disable RLS untuk development:**
```sql
ALTER TABLE public."User" DISABLE ROW LEVEL SECURITY;
```

**Enable kembali dengan policy yang proper:**
```sql
ALTER TABLE public."User" ENABLE ROW LEVEL SECURITY;

-- Allow everyone to insert (untuk sign up)
CREATE POLICY "Allow insert for signup" ON public."User"
FOR INSERT WITH CHECK (true);

-- Allow select own data
CREATE POLICY "Allow users to view own profile" ON public."User"
FOR SELECT USING (auth.uid() = ID_User::uuid OR true);

-- Allow update own data
CREATE POLICY "Allow users to update own profile" ON public."User"
FOR UPDATE USING (auth.uid() = ID_User::uuid);
```

---

## 🎯 Next Actions

**Langkah 1 (PRIORITAS):**
Cek apakah RLS enabled atau disabled:
- Buka Supabase Dashboard
- Pilih tabel "User"
- Lihat tombol "RLS"

**Langkah 2:**
Jika RLS enabled, disable untuk testing

**Langkah 3:**
Test sign up lagi dan lihat apakah data tersimpan

**Langkah 4:**
Report hasilnya ke saya dengan:
- RLS status (enabled/disabled)
- Ada error message atau tidak
- Data tersimpan atau tidak

---

## 💡 Pro Tips

1. **Buka console browser** saat testing:
   - F12 → Console
   - Lihat ada error atau tidak

2. **Buka terminal Streamlit** saat testing:
   - Lihat ada debug message atau tidak
   - Pesan dari `print()` akan muncul di sini

3. **Cek Network tab** di browser Developer Tools:
   - Lihat request ke Supabase API
   - Status 200 (OK) atau error?

---

**Silakan lakukan debugging ini dan report hasilnya! Saya siap membantu.** 🚀
