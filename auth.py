# auth.py
import streamlit as st
import sqlite3, os, hashlib
from datetime import datetime

# Users DB akan disimpan di folder 'projects' sebagai 'users.db'
BASE_PROJECTS_DIR = "projects"
USERS_DB = os.path.join(BASE_PROJECTS_DIR, "users.db")
os.makedirs(BASE_PROJECTS_DIR, exist_ok=True)

def init_users_db():
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            pw_hash TEXT,
            salt TEXT,
            created_at TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def _hash_password(password: str, salt_hex: str) -> str:
    # PBKDF2-HMAC-SHA256
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200000)
    return dk.hex()

def create_user(username: str, password: str) -> (bool, str):
    """Create user. Returns (ok, message)."""
    init_users_db()
    username = username.strip()
    if not username or not password:
        return False, "Username/password tidak boleh kosong."
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    if c.fetchone():
        conn.close()
        return False, "Username sudah terdaftar."
    salt = os.urandom(16).hex()
    pw_hash = _hash_password(password, salt)
    c.execute("INSERT INTO users (username, pw_hash, salt, created_at) VALUES (?, ?, ?, ?)",
              (username, pw_hash, salt, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return True, "Akun berhasil dibuat."

def verify_user(username: str, password: str) -> bool:
    init_users_db()
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute("SELECT pw_hash, salt FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if not row:
        return False
    pw_hash_db, salt = row
    return _hash_password(password, salt) == pw_hash_db

def login_page():
    """Render login / register UI."""
    init_users_db()
    st.title("🔐 Login / Register")

    # Initialize session state to control form visibility
    if "show_login_form" not in st.session_state:
        st.session_state["show_login_form"] = True  # Default is showing login

    # Style buttons to look more modern
    login_button_style = """
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            padding: 10px 20px;
            border: none;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
    """
    st.markdown(login_button_style, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])  # Add columns for button layout
    with col1:
        if st.button("Login"):
            st.session_state["show_login_form"] = True  # Show login form
    with col2:
        if st.button("Create New Account"):
            st.session_state["show_login_form"] = False  # Show register form

    if st.session_state["show_login_form"]:
        # Show login form
        st.subheader("Login")
        with st.form("login_form"):
            login_user = st.text_input("Username", key="login_user")
            login_pw = st.text_input("Password", type="password", key="login_pw")
            submitted = st.form_submit_button("Login")
            if submitted:
                if verify_user(login_user, login_pw):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = login_user.strip()
                    st.success(f"Welcome back, {login_user}!")
                    st.session_state["user_folder"] = os.path.join(BASE_PROJECTS_DIR, login_user)
                    os.makedirs(st.session_state["user_folder"], exist_ok=True)

                    st.switch_page("pages/Ask_Kelly.py")

                else:
                    st.error("Incorrect username or password.")

    else:
        # Show register form
        st.subheader("Create a New Account")
        with st.form("register_form"):
            reg_user = st.text_input("New Username", key="reg_user")
            reg_pw = st.text_input("New Password", type="password", key="reg_pw")
            reg_submit = st.form_submit_button("Create Account")
            if reg_submit:
                ok, msg = create_user(reg_user, reg_pw)
                if ok:
                    st.success("Account created successfully. Please log in.")
                else:
                    st.error(msg)


def logout_and_rerun():
    for k in ["logged_in", "username"]:
        if k in st.session_state:
            st.session_state.pop(k)
    # segarkan UI
    st.rerun()
