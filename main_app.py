import os
import streamlit as st
from auth import login_page, logout_and_rerun  

BASE_PROJECTS_DIR = "projects"
os.makedirs(BASE_PROJECTS_DIR, exist_ok=True)

# --- Force login ---
if "logged_in" not in st.session_state or not st.session_state.get("logged_in"):
    login_page()   # tampilkan form login/register
    st.stop()      # stop main app sampai user login

# user sudah login di sini
USERNAME = st.session_state.get("username", "guest")
PROJECTS_DIR = os.path.join(BASE_PROJECTS_DIR, USERNAME)
os.makedirs(PROJECTS_DIR, exist_ok=True)

# contoh fungsi safe_name dan project_path (pastikan tidak membuat double-nesting)
import re
def safe_name(name: str, max_len: int = 120) -> str:
    s = re.sub(r"[^0-9a-zA-Z_\-\.]", "_", (name or "untitled"))
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:max_len] or "project").lower()

def project_path(proj_name: str) -> str:
    p = os.path.join(PROJECTS_DIR, safe_name(proj_name))
    os.makedirs(p, exist_ok=True)
    return p

def project_db_path(proj_name: str) -> str:
    return os.path.join(project_path(proj_name), "project.db")

# di sidebar: tombol logout
with st.sidebar:
    st.write(f"👋 Logged in as: **{USERNAME}**")
    if st.button("🚪 Logout"):
        logout_and_rerun()
