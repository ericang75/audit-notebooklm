"""
NotebookLM-style Audit Tool with Templates (enhanced)
- Multi-project with file management (create, delete, remove files)
- Upload multi-file (CSV/XLSX/PDF) with persistence
- Auto reload documents from project folder
- Select which files to analyze
- Automatic audit findings templates with compact summary view
- Dual-table key auto-detection and manual mapping with join analysis
- Natural language narrative generation via OpenAI (optional)
- Export findings to JSON/Excel/PDF
- Improved UI with file management table
- Conversation/logging persisted per project (SQLite)
"""

import streamlit as st
import os, io, re, json, sqlite3, time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import base64

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# OpenAI (optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
except Exception:
    SimpleDocTemplate = None

# ---------------- Configuration ----------------
PROJECTS_DIR = "projects"
DB_FILENAME = "project.db"
os.makedirs(PROJECTS_DIR, exist_ok=True)

st.set_page_config(page_title="NotebookLM — Audit Templates", page_icon="📘", layout="wide")
st.title("📘 NotebookLM — Audit (Templates Enhanced)")

# ---------------- Session defaults ----------------
if "active_project" not in st.session_state:
    st.session_state.active_project = None
if "dfs" not in st.session_state:
    st.session_state.dfs = {}

if "show_log" not in st.session_state:
    st.session_state.show_log = False
if "show_details" not in st.session_state:
    st.session_state.show_details = {}
if "last_findings" not in st.session_state:
    st.session_state.last_findings = {}
if "last_narrative" not in st.session_state:
    st.session_state.last_narrative = ""

# ---------------- Utilities ----------------
def safe_name(name: str, max_len: int = 120) -> str:
    s = re.sub(r"[^0-9a-zA-Z_\-\.]", "_", (name or "untitled"))
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:max_len] or "project").lower()

def project_path(name: str) -> str:
    p = os.path.join(PROJECTS_DIR, safe_name(name))
    os.makedirs(p, exist_ok=True)
    return p

def project_db_path(name: str) -> str:
    return os.path.join(project_path(name), DB_FILENAME)

def list_projects() -> List[str]:
    return sorted([d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))])

# ---------------- Database (per-project) ----------------
def init_project_db(proj: str):
    dbp = project_db_path(proj)
    conn = sqlite3.connect(dbp)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_name TEXT PRIMARY KEY,
            file_type TEXT,
            original_filename TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            role TEXT,
            message TEXT
        )
    """)
    conn.commit()
    conn.close()

def register_document(proj: str, doc_name: str, file_type: str, original_filename: str):
    dbp = project_db_path(proj)
    conn = sqlite3.connect(dbp)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO documents (doc_name, file_type, original_filename, uploaded_at) VALUES (?, ?, ?, ?)",
              (doc_name, file_type, original_filename, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def remove_document(proj: str, doc_name: str):
    """Remove document from database and memory"""
    dbp = project_db_path(proj)
    conn = sqlite3.connect(dbp)
    c = conn.cursor()
    c.execute("DELETE FROM documents WHERE doc_name = ?", (doc_name,))
    conn.commit()
    conn.close()
    
    # Remove from session state
    if proj in st.session_state.dfs and doc_name in st.session_state.dfs[proj]:
        del st.session_state.dfs[proj][doc_name]

def list_documents(proj: str) -> List[Tuple[str, str, str, str]]:
    dbp = project_db_path(proj)
    conn = sqlite3.connect(dbp)
    c = conn.cursor()
    try:
        c.execute("SELECT doc_name, file_type, original_filename, uploaded_at FROM documents ORDER BY uploaded_at DESC")
        rows = c.fetchall()
    except Exception:
        rows = []
    conn.close()
    return rows

def log_conversation(proj: str, role: str, message: str):
    dbp = project_db_path(proj)
    conn = sqlite3.connect(dbp)
    c = conn.cursor()
    c.execute("INSERT INTO conversations (role, message) VALUES (?, ?)", (role, message))
    conn.commit()
    conn.close()

def get_conversations(proj: str, limit: int = 200) -> List[Tuple[int, str, str, str]]:
    dbp = project_db_path(proj)
    conn = sqlite3.connect(dbp)
    c = conn.cursor()
    try:
        c.execute("SELECT id, ts, role, message FROM conversations ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall()
    except Exception:
        rows = []
    conn.close()
    return rows

# ---------------- Export Functions ----------------
def export_to_json(findings: Dict, narrative: str = "") -> str:
    """Export findings and narrative to JSON"""
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "findings": findings,
        "narrative": narrative
    }
    return json.dumps(export_data, indent=2, default=str)

def export_to_excel(findings: Dict, narrative: str = "") -> bytes:
    """Export findings to Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Summary sheet
        summary_data = []
        for table_name, table_findings in findings.items():
            summary_data.append({
                "Table": table_name,
                "Rows": table_findings.get("row_count", 0),
                "Columns": table_findings.get("col_count", 0),
                "Missing Values": len(table_findings.get("missing_values", {})),
                "Duplicates": sum(len(v.get("sample", [])) for v in table_findings.get("duplicates", {}).values()),
                "Numeric Anomalies": len(table_findings.get("numeric_anomalies", {}))
            })
        if summary_data:
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Narrative sheet
        if narrative:
            pd.DataFrame([{"Narrative": narrative}]).to_excel(writer, sheet_name='Narrative', index=False)
        
        # Detailed findings
        for table_name, table_findings in findings.items():
            sheet_name = table_name[:31]  # Excel sheet name limit
            pd.DataFrame([table_findings]).to_excel(writer, sheet_name=sheet_name, index=False)
    
    return output.getvalue()

def generate_pdf_report(findings: Dict, narrative: str = "", project_name: str = "") -> bytes:
    """Generate PDF report"""
    if SimpleDocTemplate is None:
        return b"PDF generation not available. Install reportlab."
    
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=24)
    story.append(Paragraph(f"Audit Report - {project_name}", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    for table_name, table_findings in findings.items():
        story.append(Paragraph(f"<b>{table_name}</b>", styles['Normal']))
        summary_text = f"""
        • Rows: {table_findings.get('row_count', 0)}<br/>
        • Columns: {table_findings.get('col_count', 0)}<br/>
        • Missing Values: {len(table_findings.get('missing_values', {}))}<br/>
        • Duplicates Found: {sum(len(v.get('sample', [])) for v in table_findings.get('duplicates', {}).values())}<br/>
        • Numeric Anomalies: {len(table_findings.get('numeric_anomalies', {}))}
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Narrative
    if narrative:
        story.append(Paragraph("Analysis Narrative", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(narrative.replace('\n', '<br/>'), styles['Normal']))
    
    doc.build(story)
    return output.getvalue()

# ---------------- Join Analysis ----------------
def detect_join_keys(df1: pd.DataFrame, df2: pd.DataFrame) -> List[str]:
    """Auto-detect potential join keys between two dataframes"""
    potential_keys = []
    
    # Common key patterns
    common_patterns = ['id', 'no', 'code', 'key', 'ref']
    
    for col1 in df1.columns:
        for col2 in df2.columns:
            # Check exact match
            if col1.lower() == col2.lower():
                potential_keys.append((col1, col2))
                continue
            
            # Check pattern match
            for pattern in common_patterns:
                if pattern in col1.lower() and pattern in col2.lower():
                    potential_keys.append((col1, col2))
                    break
    
    return potential_keys

def perform_join_analysis(df1: pd.DataFrame, df2: pd.DataFrame, key1: str, key2: str) -> Dict[str, Any]:
    """Perform join analysis between two tables"""
    analysis = {}
    
    try:
        # Convert keys to string for consistent comparison
        df1_copy = df1.copy()
        df2_copy = df2.copy()
        df1_copy[key1] = df1_copy[key1].astype(str)
        df2_copy[key2] = df2_copy[key2].astype(str)
        
        # Find matches and mismatches
        keys1 = set(df1_copy[key1].dropna())
        keys2 = set(df2_copy[key2].dropna())
        
        matched = keys1.intersection(keys2)
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        
        analysis["total_rows_table1"] = len(df1)
        analysis["total_rows_table2"] = len(df2)
        analysis["matched_keys"] = len(matched)
        analysis["unmatched_in_table1"] = len(only_in_1)
        analysis["unmatched_in_table2"] = len(only_in_2)
        
        # Sample mismatches
        if only_in_1:
            sample_1 = df1_copy[df1_copy[key1].isin(list(only_in_1)[:5])]
            analysis["sample_unmatched_table1"] = sample_1.head(5).to_dict(orient='records')
        
        if only_in_2:
            sample_2 = df2_copy[df2_copy[key2].isin(list(only_in_2)[:5])]
            analysis["sample_unmatched_table2"] = sample_2.head(5).to_dict(orient='records')
        
        # Check for duplicates in join keys
        dup1 = df1_copy[df1_copy.duplicated(subset=[key1], keep=False)]
        dup2 = df2_copy[df2_copy.duplicated(subset=[key2], keep=False)]
        
        if not dup1.empty:
            analysis["duplicates_in_table1_key"] = len(dup1)
        if not dup2.empty:
            analysis["duplicates_in_table2_key"] = len(dup2)
            
    except Exception as e:
        analysis["error"] = str(e)
    
    return analysis

# ---------------- OpenAI client ----------------
@st.cache_resource
def get_openai_client():
    # Lebih aman pakai .get() agar tidak error kalau key belum ada
    api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return None

    if OpenAI is None:
        st.warning("⚠️ openai package not installed; LLM features disabled.")
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.warning(f"⚠️ Failed to initialize OpenAI client: {e}")
        return None

if "openai_client" not in st.session_state:
    st.session_state.openai_client = get_openai_client()

if st.session_state.openai_client is None:
    st.warning("⚠️ OpenAI API Key not loaded. Please check your Streamlit Secrets.")
else:
    st.success("✅ OpenAI client initialized.")

# ---------------- File readers ----------------
def read_csv_bytes(b: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(b), dtype=str, low_memory=False)
    except Exception:
        s = b.decode("utf-8", errors="ignore")
        return pd.read_csv(io.StringIO(s), dtype=str, low_memory=False)

def read_excel_bytes(b: bytes) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(b), dtype=str)
    except Exception:
        return pd.DataFrame()

def extract_pdf_tables_or_text(b: bytes, max_pages: int = 50) -> Tuple[List[pd.DataFrame], str]:
    """
    Extract PDF menggunakan pdfplumber.
    Mengembalikan (list_of_tables, plain_text).
    """
    dfs = []  # kalau mau tetap support Camelot, bisa taruh di sini
    text = ""

    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= max_pages:
                    break
                t = page.extract_text() or ""
                if t.strip():
                    parts.append(f"[page {i+1}]\n{t}")
        text = "\n\n".join(parts)
    except Exception as e:
        text = f"[pdfplumber error: {e}]"

    return dfs, text


def clean_table_with_ai_from_text(text: str, client, max_chars: int = 4000) -> pd.DataFrame:
    """Gunakan AI untuk membersihkan teks PDF menjadi tabel (CSV)."""
    if client is None or not text.strip():
        return pd.DataFrame()

    try:
        snippet = text[:max_chars]

        prompt = f"""
        Berikut adalah isi PDF (teks hasil ekstraksi):

        {snippet}

        Tolong ekstrak tabel ini menjadi format CSV yang rapi:
        - Tentukan header yang sesuai
        - Gabungkan baris tabel yang pecah
        - Hilangkan noise (nomor halaman, header/footer)
        - Jangan tambahkan penjelasan, hanya keluarkan CSV valid
        """

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a PDF-to-table converter. Output only clean CSV."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=2000
        )

        cleaned_csv = resp.choices[0].message.content.strip()
        df_clean = pd.read_csv(io.StringIO(cleaned_csv))
        return df_clean

    except Exception as e:
        st.warning(f"AI table extraction failed: {e}")
        return pd.DataFrame()


# ---------------- Persistence helpers ----------------
def save_uploaded_file(proj: str, file, raw: bytes):
    """Save & parse uploaded file into session_state.dfs[proj]"""
    safe = safe_name(os.path.splitext(file.name)[0])
    lower = file.name.lower()
    proj_dir = project_path(proj)

    target_path = os.path.join(proj_dir, file.name)
    try:
        with open(target_path, "wb") as fo:
            fo.write(raw)
    except Exception:
        pass

    if lower.endswith(".csv"):
        df = read_csv_bytes(raw)
        st.session_state.dfs[proj][safe] = df
        register_document(proj, safe, "csv", file.name)
        st.success(f"✅ Loaded CSV `{file.name}` → table `{safe}` ({len(df)} rows).")

    elif lower.endswith((".xls", ".xlsx")):
        df = read_excel_bytes(raw)
        st.session_state.dfs[proj][safe] = df
        register_document(proj, safe, "excel", file.name)
        st.success(f"✅ Loaded Excel `{file.name}` → table `{safe}` ({len(df)} rows).")

    elif lower.endswith(".pdf"):
        dfs, txt = extract_pdf_tables_or_text(raw)

        if txt:
            parsed = []
            for seg in re.finditer(r"\[page\s*(\d+)\]\s*(.*?)\n(?=\[page|\Z)", txt, flags=re.S):
                pg = seg.group(1)
                content = seg.group(2).strip()
                parsed.append({"page": int(pg), "text": content})

            # fallback kalau regex gagal
            if not parsed:
                raw_pages = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
                for i, t in enumerate(raw_pages):
                    parsed.append({"page": i+1, "text": t})

            df_pdf = pd.DataFrame(parsed)
            st.session_state.dfs[proj][safe] = df_pdf
            register_document(proj, safe, "pdf_text", file.name)
            st.success(f"✅ Loaded PDF `{file.name}` → stored as text ({len(df_pdf)} chunks).")
        else:
            st.warning(f"⚠️ Failed to parse PDF `{file.name}` (no table/text found).")


def reload_project_files(proj: str):
    """Load saved CSV/XLSX files from project folder back into session_state.dfs."""
    proj_dir = project_path(proj)
    if proj not in st.session_state.dfs:
        st.session_state.dfs[proj] = {}
    for f in os.listdir(proj_dir):
        path = os.path.join(proj_dir, f)
        safe = safe_name(os.path.splitext(f)[0])
        try:
            if f.lower().endswith(".csv"):
                st.session_state.dfs[proj][safe] = pd.read_csv(path, dtype=str, low_memory=False)
            elif f.lower().endswith((".xls", ".xlsx")):
                st.session_state.dfs[proj][safe] = pd.read_excel(path, dtype=str)
        except Exception:
            continue

# ---------------- Audit templates ----------------
def run_single_table_templates(df: pd.DataFrame, table_name: str, file_type: str = "") -> Dict[str, Any]:
    """Run single-table templates"""
    findings: Dict[str, Any] = {"table": table_name, "row_count": int(len(df)), "col_count": int(len(df.columns))}

    if file_type == "pdf_text":
        findings["note"] = "This document was extracted as plain text, not a structured table."
        hidden_sample = df.head(5).to_dict(orient="records")
        findings["_hidden_sample_text"] = hidden_sample
        return findings

    # Structured-data analysis
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if not miss.empty:
        findings["missing_values"] = miss.astype(int).to_dict()

    # Duplicates
    dup_report = {}
    candidate_keys = ["Employee_No", "EmployeeNo", "Tax_ID", "TaxID"]
    candidate_keys += [df.columns[0]] if len(df.columns) > 0 else []
    for key in candidate_keys:
        if key in df.columns:
            try:
                dups = df[df.duplicated(subset=[key], keep=False)]
                if not dups.empty:
                    dup_report[key] = {"count": int(dups.shape[0]), "sample": dups.head(5).to_dict(orient="records")}
            except Exception:
                continue
    if dup_report:
        findings["duplicates"] = dup_report

    # Numeric anomalies
    anomalies = {}
    numeric_cols = []
    for c in df.columns:
        try:
            coer = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
            if coer.notna().sum() > 0:
                numeric_cols.append(c)
        except Exception:
            continue

    for c in numeric_cols:
        col = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
        neg = df[col < 0] if not col.empty else pd.DataFrame()
        if not neg.empty:
            anomalies[c] = {"negative_count": int(len(neg)), "sample": neg.head(3).to_dict(orient="records")}
        zeros = df[col == 0] if not col.empty else pd.DataFrame()
        if not zeros.empty and len(df) > 0 and (len(zeros) / len(df)) > 0.01:
            anomalies[c] = anomalies.get(c, {})
            anomalies[c]["zero_count"] = int(len(zeros))
    if anomalies:
        findings["numeric_anomalies"] = anomalies

    return findings

def truncate_dict(d: dict, max_chars: int = 4000) -> str:
    """
    Convert dict to JSON string and truncate if too long
    (to avoid hitting OpenAI context limit).
    """
    s = json.dumps(d, default=str, indent=2)
    if len(s) > max_chars:
        s = s[:max_chars] + "\n...[TRUNCATED]..."
    return s

def generate_narrative(question: str,
                       findings: Dict[str, Any],
                       context_samples: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate AI narrative combining rule-based findings + small data samples.
    findings → full rule-based results (compact already).
    context_samples → dict of {table_name: list of sample rows}.
    """
    client = st.session_state.openai_client
    if client is None:
        return json.dumps(findings, indent=2)[:800]

    # Compact findings
    findings_str = truncate_dict(findings, max_chars=5000)

    # Add samples if available
    sample_str = ""
    if context_samples:
        sample_str = truncate_dict(context_samples, max_chars=3000)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an experienced audit analyst. "
                        "Your task is to explain the findings clearly, "
                        "highlight risks, anomalies, and give recommendations. "
                        "Use the provided rule-based findings as ground truth. "
                        "Use data samples only as illustrative examples, not for statistics."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\n"
                               f"Rule-based Findings:\n{findings_str}\n\n"
                               f"Sample Records (for illustration only):\n{sample_str}"
                }
            ],
            temperature=0.2,
            max_tokens=800
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Narrative generation failed: {e}]\n\nFallback Findings:\n{findings_str}"

# ---------------- UI helpers ----------------
def sidebar_panel():
    st.sidebar.header("📁 Projects")
    projects = list_projects()
    if st.session_state.active_project in projects:
        default_index = projects.index(st.session_state.active_project) + 1
    else:
        default_index = 0
    choice = st.sidebar.selectbox("Active project", ["-- new project --"] + projects, index=default_index)
    if choice == "-- new project --":
        new_name = st.sidebar.text_input("Project name")
        if st.sidebar.button("➕ Create project"):
            if new_name.strip():
                safe = safe_name(new_name.strip())
                st.session_state.active_project = safe
                init_project_db(safe)
                reload_project_files(safe)
                st.sidebar.success(f"Created project: {safe}")
    else:
        st.session_state.active_project = choice
        st.sidebar.markdown("---")
        st.sidebar.markdown("⚠️ **Delete Project**")
        delete_confirm = st.sidebar.text_input("Type DELETE to confirm")
        if st.sidebar.button("🗑️ Delete project", type="secondary"):
            if delete_confirm == "DELETE":
                import shutil
                shutil.rmtree(project_path(choice), ignore_errors=True)
                st.session_state.active_project = None
                st.sidebar.success(f"Deleted project: {choice}")

def render_summary(findings: Dict[str, Any], table_name: str):
    """Render compact summary with expandable details"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", findings.get("row_count", 0))
        st.metric("Total Columns", findings.get("col_count", 0))
    
    with col2:
        missing_count = len(findings.get("missing_values", {}))
        st.metric("Missing Values", missing_count)
        
        dup_count = sum(v.get("count", 0) for v in findings.get("duplicates", {}).values())
        st.metric("Duplicate Rows", dup_count)
    
    with col3:
        anomaly_count = len(findings.get("numeric_anomalies", {}))
        st.metric("Numeric Anomalies", anomaly_count)
    
    # Show details button
    details_key = f"details_{table_name}"
    if st.button(f"📊 Show Details for {table_name}", key=f"btn_{details_key}"):
        st.session_state.show_details[details_key] = not st.session_state.show_details.get(details_key, False)
    
    if st.session_state.show_details.get(details_key, False):
        st.markdown("### 🔍 Detailed Findings")

        # Missing values
        if "missing_values" in findings:
            st.markdown("#### ⚠️ Missing Values")
            st.json(findings["missing_values"])

        # Duplicates
        if "duplicates" in findings:
            st.markdown("#### 🔁 Duplicate Records")
            for key, dup_info in findings["duplicates"].items():
                st.write(f"Duplicates detected on **{key}**: {dup_info.get('count', 0)} rows")
                st.dataframe(pd.DataFrame(dup_info.get("sample", [])))

        # Numeric anomalies
        if "numeric_anomalies" in findings:
            st.markdown("#### 📉 Numeric Anomalies")
            for col, issues in findings["numeric_anomalies"].items():
                st.write(f"Column **{col}** anomalies:")
                st.json(issues)

        # Full JSON fallback
        with st.expander("🗂 Full JSON Findings"):
            st.json(findings)

# ---------------- Main UI ----------------
def main_ui():
    st.session_state.openai_client = get_openai_client()
    sidebar_panel()

    if not st.session_state.active_project:
        st.info("👈 Create or select a project in the sidebar to begin.")
        st.stop()

    proj = st.session_state.active_project
    init_project_db(proj)
    reload_project_files(proj)

    if proj not in st.session_state.dfs:
        st.session_state.dfs[proj] = {}

    st.header(f"📋 Project: {proj}")

    # Upload area
    with st.expander("📤 Upload Files", expanded=True):
        uploaded = st.file_uploader("Choose files (CSV / XLSX / PDF)", type=["csv", "xls", "xlsx", "pdf"], accept_multiple_files=True)
        if uploaded:
            for f in uploaded:
                try:
                    raw = f.read()
                    save_uploaded_file(proj, f, raw)
                except Exception as e:
                    st.error(f"Failed to read {f.name}: {e}")

    # File Management Table
    st.markdown("---")
    st.subheader("📂 File Management")
    docs = list_documents(proj)

    if docs:
        # Create DataFrame for display
        file_data = []
        for doc in docs:
            doc_name, file_type, original_filename, uploaded_at = doc
            file_data.append({
                "Name": doc_name,
                "Type": file_type,
                "Original File": original_filename,
                "Uploaded At": uploaded_at
            })
        
        df_files = pd.DataFrame(file_data)
        
        # Display with multi-selection
        selected_rows = st.dataframe(
            df_files,
            use_container_width=True,
            hide_index=True,
            selection_mode="multi-row",  # 🔹 ubah ke multi
            on_select="rerun"
        )
        
        if selected_rows and selected_rows.selection.rows:
            selected_idx_list = selected_rows.selection.rows
            selected_docs = [docs[i][0] for i in selected_idx_list]
            st.info(f"Selected: {', '.join(selected_docs)}")
            
            if st.button("🗑️ Remove Selected Files", type="secondary"):
                for i in selected_idx_list:
                    selected_doc = docs[i][0]
                    remove_document(proj, selected_doc)
                    try:
                        file_path = os.path.join(project_path(proj), docs[i][2])
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except:
                        pass
                st.success(f"Removed {len(selected_idx_list)} file(s)")
                st.rerun()
    else:
        st.info("No files uploaded yet. Use the upload section above.")


    # Table selection for analysis
    st.markdown("---")
    st.subheader("🔍 Analysis")
    table_names = list(st.session_state.dfs.get(proj, {}).keys())
    
    if not table_names:
        st.warning("No tables available. Please upload some files first.")
        st.stop()
    
    chosen = st.multiselect(
        "Select files to analyze (max 2 for join analysis)",
        table_names,
        max_selections=2,
        help="Select 1 file for single analysis, 2 files for join analysis"
    )
    
    if not chosen:
        st.info("Select up to 2 tables/documents to run analysis.")
        st.stop()

    # Build tables dict
    docs_map = {d[0]: d[1] for d in docs}
    tables: Dict[str, pd.DataFrame] = {}
    for n in chosen:
        if n in st.session_state.dfs[proj]:
            tables[n] = st.session_state.dfs[proj][n]

    # Preview section
    with st.expander("📊 Data Preview", expanded=False):
        for name, df in tables.items():
            file_type = docs_map.get(name, "")
            if file_type == "pdf_text" and set(df.columns) == {"page", "text"}:
                st.info(f"`{name}` is a PDF text document. Preview hidden for privacy.")
            else:
                st.markdown(f"**{name}** — {len(df)} rows × {len(df.columns)} cols")
                st.dataframe(df.head(20), use_container_width=True)


    if len(chosen) == 1:
        # --- Single File Mode ---
        name = chosen[0]
        df = tables[name]
        ft = docs_map.get(name, "")

        if st.button("🔎 Run Analysis", use_container_width=True):
            fnd = run_single_table_templates(df, name, file_type=ft)
            render_summary(fnd, name)

            # 🔹 AI insight
            if st.session_state.openai_client:
                ai_text = generate_narrative(
                    f"Analyze single table {name} for audit findings",
                    fnd,
                    context_samples={name: df.head(10).to_dict(orient="records")}
                )
                st.markdown("#### 🤖 AI Audit Findings")
                st.write(ai_text)
                st.session_state.last_narrative = ai_text

            st.session_state.last_findings = {name: fnd}

    elif len(chosen) == 2:
        st.markdown("### 🔗 Join Analysis")
        df1, df2 = tables[chosen[0]], tables[chosen[1]]
        
        # Auto-detect keys
        potential_keys = detect_join_keys(df1, df2)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if potential_keys:
                auto_key1, auto_key2 = potential_keys[0] if potential_keys else (None, None)
            else:
                st.info("No automatic keys detected. Please select manually.")
                auto_key1, auto_key2 = None, None
            
            key1 = st.selectbox(
                f"Join key from {chosen[0]}",
                df1.columns,
                index=list(df1.columns).index(auto_key1) if auto_key1 and auto_key1 in df1.columns else 0
            )
        
        with col_b:
            key2 = st.selectbox(
                f"Join key from {chosen[1]}",
                df2.columns,
                index=list(df2.columns).index(auto_key2) if auto_key2 and auto_key2 in df2.columns else 0
            )
        
        if st.button("🔍 Analyze Join", use_container_width=True):
            dup1 = df1[df1.duplicated([key1], keep=False)]
            dup2 = df2[df2.duplicated([key2], keep=False)]

            join_analysis = perform_join_analysis(df1, df2, key1, key2)
            join_analysis["duplicates_table1"] = dup1.to_dict(orient="records") if not dup1.empty else []
            join_analysis["duplicates_table2"] = dup2.to_dict(orient="records") if not dup2.empty else []
            
            # Display join results
            st.markdown("#### Join Analysis Results")
            
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.metric("Matched Records", join_analysis.get("matched_keys", 0))
            with col_2:
                st.metric(f"Only in {chosen[0]}", join_analysis.get("unmatched_in_table1", 0))
            with col_3:
                st.metric(f"Only in {chosen[1]}", join_analysis.get("unmatched_in_table2", 0))
            
            # Show mismatches
            if join_analysis.get("unmatched_in_table1", 0) > 0:
                with st.expander(f"Sample records only in {chosen[0]}"):
                    sample = join_analysis.get("sample_unmatched_table1", [])
                    if sample:
                        st.dataframe(pd.DataFrame(sample))
                    else:
                        st.write("No sample available")

            if join_analysis.get("unmatched_in_table2", 0) > 0:
                with st.expander(f"Sample records only in {chosen[1]}"):
                    sample = join_analysis.get("sample_unmatched_table2", [])
                    if sample:
                        st.dataframe(pd.DataFrame(sample))
                    else:
                        st.write("No sample available")

            if join_analysis["duplicates_table1"]:
                with st.expander(f"🔁 Duplicates in {chosen[0]} on {key1}"):
                    st.dataframe(dup1)

            if join_analysis["duplicates_table2"]:
                with st.expander(f"🔁 Duplicates in {chosen[1]} on {key2}"):
                    st.dataframe(dup2)
            
            # Save join analysis
            st.session_state.last_findings["join_analysis"] = join_analysis
            
            # 🔹 AI Narrative tambahan
            ai_text = generate_narrative(
                f"Analyze and provide audit insights for the join between {chosen[0]} and {chosen[1]} on keys {key1} and {key2}",
                join_analysis,
                context_samples={
                    chosen[0]: df1.head(10).to_dict(orient="records"),
                    chosen[1]: df2.head(10).to_dict(orient="records")
                }
            )
            st.markdown("#### 🤖 AI Insights on Join")
            st.write(ai_text)
            st.session_state.last_narrative = ai_text

    elif len(chosen) > 2:
        st.markdown("### 🔍 Multi-File Analysis")
        # 🔹 AI-driven relationship finding
        if st.session_state.openai_client:
            schema_parts = []
            context_samples = {}
            for name, df in tables.items():
                schema_parts.append(f"{name}: {len(df)} rows, {len(df.columns)} cols")
                context_samples[name] = df.head(5).to_dict(orient="records")
            
            ai_text = generate_narrative(
                f"Analyze relationships and potential joins across {len(chosen)} tables",
                {"schemas": schema_parts},
                context_samples=context_samples
            )
            st.markdown("#### 🤖 AI Multi-Table Insights")
            st.write(ai_text)
            st.session_state.last_narrative = ai_text



    # Export section
    if st.session_state.last_findings:
        st.markdown("---")
        st.subheader("📥 Export Results")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            json_data = export_to_json(st.session_state.last_findings, st.session_state.last_narrative)
            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name=f"{proj}_findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col_exp2:
            excel_data = export_to_excel(st.session_state.last_findings, st.session_state.last_narrative)
            st.download_button(
                label="📊 Download Excel",
                data=excel_data,
                file_name=f"{proj}_findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col_exp3:
            pdf_data = generate_pdf_report(st.session_state.last_findings, st.session_state.last_narrative, proj)
            st.download_button(
                label="📑 Download PDF Report",
                data=pdf_data,
                file_name=f"{proj}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    # Ask AI section
    # Ask AI section
    st.markdown("---")
    st.subheader("💬 Ask AI")
    with st.expander("💡 Suggested Questions", expanded=False):
        suggested_questions = [
            "What are the main data quality issues in these tables?",
            "Identify the top 5 risks based on the anomalies found",
            "Are there any suspicious patterns in the duplicate records?",
            "What additional checks would you recommend for this audit?",
            "Summarize the critical findings that need immediate attention",
            "Compare the data between the two tables and highlight discrepancies",
            "What are the potential financial impacts of these anomalies?",
            "Generate an executive summary for senior management"
        ]
        
        col_sug1, col_sug2 = st.columns(2)
        for i, suggestion in enumerate(suggested_questions):
            col = col_sug1 if i % 2 == 0 else col_sug2
            with col:
                if st.button(f"📌 {suggestion[:40]}...", key=f"sug_{i}", use_container_width=True):
                    st.session_state[f"ai_input_{proj}"] = suggestion
                    st.rerun()

    # Text area terhubung langsung ke session_state
    q = st.text_area(
        "Ask a custom question about the selected tables/documents",
        height=100,
        placeholder="e.g., What are the main risks in this data? Are there any patterns in the anomalies?",
        key=f"ai_input_{proj}"
    )

    # Button untuk kirim pertanyaan ke AI
    if st.button("🤖 Ask Kelly", use_container_width=True):
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            # Build context
            schema_parts = []
            context_samples = {}
            for name, df in tables.items():
                cols = list(df.columns)
                schema_parts.append(f"{name}: columns = {cols}, rows = {len(df)}")
                ft = docs_map.get(name, "")
                if ft == "pdf_text":
                    context_samples[name] = df.head(5).to_dict(orient="records")
                else:
                    try:
                        context_samples[name] = df.head(10).to_dict(orient="records")
                    except Exception:
                        context_samples[name] = str(df.head(5))

            prompt_meta = "\n".join(schema_parts)
            client = st.session_state.openai_client

            if client is None:
                st.info("OpenAI not configured. Showing context preview:")
                st.code(prompt_meta)
                log_conversation(proj, "user", q)
                log_conversation(proj, "assistant", "OpenAI not configured")
            else:
                log_conversation(proj, "user", q)
                findings_stub = {"selected_docs": list(tables.keys()), "schema": schema_parts}
                with st.spinner("Thinking..."):
                    narrative = generate_narrative(
                        f"User question: {q}",
                        findings_stub,
                        context_samples=context_samples
                    )
                st.markdown("### 🤖 AI Answer")
                st.write(narrative)
                log_conversation(proj, "assistant", narrative)


    # Conversation log
    st.markdown("---")
    st.subheader("📜 Conversation Log")
    
    col_log1, col_log2 = st.columns([1, 4])
    with col_log1:
        if st.button("🔄 Toggle Log", use_container_width=True):
            st.session_state.show_log = not st.session_state.show_log
    
    if st.session_state.show_log:
        convs = get_conversations(proj, limit=50)
        if convs:
            for cid, ts, role, message in convs[::-1]:
                with st.container():
                    if role == "user":
                        st.markdown(f"**👤 User** - {ts}")
                    else:
                        st.markdown(f"**🤖 {role}** - {ts}")
                    st.write(message[:500] + "..." if len(message) > 500 else message)
                    st.markdown("---")
        else:
            st.info("No conversation history yet.")
    else:
        st.info("📌 Log is hidden. Click 'Toggle Log' to view conversation history.")

# ---------------- Run app ----------------
if __name__ == "__main__":
    main_ui()