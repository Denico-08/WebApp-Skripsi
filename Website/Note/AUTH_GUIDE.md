# Authentication System Guide

## Overview
The system now includes a complete authentication flow using Supabase Auth with signup, login, and logout functionality.

## Files

### 1. `Signup.py`
- **Supabase-backed signup and login**
- Functions:
  - `signup_widget()` — Render signup form (email, password, name)
  - `login_with_email(email, password)` — Authenticate user via Supabase
  - `logout()` — Sign out user
  - `require_auth(message)` — Guard function for protected pages

### 2. `Login.py`
- **Local JSON-backed login (fallback/alternative)**
- Functions:
  - `login_widget(allow_register=False)` — Render login form
  - `authenticate(username, password)` — Check credentials
  - `register_user(username, password)` — Create new user
  - `logout()` — Sign out user

### 3. `Web_Prediksi_Obesity.py`
- **Main app now requires authentication**
- Auth gate added at the top of the app
- If not logged in: shows login + signup forms side-by-side
- If logged in: shows "Logged in as: {email}" with logout button in sidebar
- All prediction/XAI features gated behind login

## How to Use

### First Time Setup
1. Create account via "Daftar Akun Baru" form
2. Enter email and password (password must be ≥6 characters)
3. Supabase will send a verification email
4. Click the verification link in email
5. Return to app and login with email + password

### Login
1. Enter email and password
2. Click "Login" button
3. App will authenticate with Supabase and reload

### Logout
1. Click "Logout" button in sidebar
2. You'll be redirected to login page

## Environment Variables

Required in `.env` file (already set):
```
SUPABASE_URL=https://wtmbkdyhzsdlnbumnmbx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## Running the App

```bash
cd "c:\Users\LENOVO\Documents\DENICO\Skripsi\Python\Website"
streamlit run Web_Prediksi_Obesity.py
```

## Default Test Credentials (if using Local Login)

If you want to use the local `Login.py` instead of Supabase:
- Username: `admin`
- Password: `admin123`

User data stored in `users.json` (auto-created)

## Security Notes

- Passwords are hashed with SHA-256 (local)
- Supabase Auth uses industry-standard OAuth/JWT
- Session data stored in Streamlit's session state (cleared on browser close)
- Never commit `.env` file or `users.json` to version control

## Troubleshooting

### "Signup fails with already registered error"
- Email is already registered in Supabase
- Try a different email or use "Lupa Password?" if Supabase adds password reset

### "Login fails with timeout"
- Check internet connection
- Verify `SUPABASE_URL` and `SUPABASE_KEY` in `.env`
- Ensure Supabase project is active in Supabase dashboard

### "Module not found: Signup/Login"
- Ensure `Signup.py` and `Login.py` are in same directory as `Web_Prediksi_Obesity.py`
- Verify `supabase` and `python-dotenv` packages are installed:
  ```bash
  pip install supabase python-dotenv
  ```

## Next Steps

- Optional: Add password reset via email
- Optional: Add user profile page (name, upload photo, etc.)
- Optional: Switch to different auth provider (Google, GitHub OAuth)
