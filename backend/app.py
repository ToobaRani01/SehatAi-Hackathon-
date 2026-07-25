from flask import Flask, render_template, request, jsonify, send_from_directory, abort, session
from flask_cors import CORS
import joblib
import pandas as pd
import tensorflow as tf
try:
    from tensorflow.keras.preprocessing import image
except ImportError:
    try:
        from keras.preprocessing import image
    except ImportError:
        image = None # Will be handled in get_model/predict
import numpy as np
import io
from PIL import Image
import os
import sqlite3
import hashlib
import secrets
import uuid
import json
from datetime import datetime

app = Flask(__name__, 
            static_url_path='', 
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend'),
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend', 'templates'))
app.secret_key = secrets.token_hex(32)
CORS(app, supports_credentials=True)

# Setup absolute base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base_dir, 'sehatai.db')

# Uploads directory for avatars
uploads_dir = os.path.join(base_dir, 'uploads', 'avatars')
os.makedirs(uploads_dir, exist_ok=True)
chat_lab_dir = os.path.join(base_dir, 'uploads', 'chat_lab')
os.makedirs(chat_lab_dir, exist_ok=True)

# ─── DATABASE SETUP ────────────────────────────────────────────────────────────

def get_db():
    """Get a database connection with row_factory for dict-like access."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initialize database tables if they don't exist."""
    conn = get_db()
    cursor = conn.cursor()

    # Table 1: user_record — manages sign-in / login
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_record (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            avatar_path TEXT DEFAULT NULL,
            mobile_number TEXT DEFAULT '',
            specialization TEXT DEFAULT '',
            clinic_name TEXT DEFAULT '',
            license_number TEXT DEFAULT '',
            theme TEXT DEFAULT 'light',
            created_at TEXT DEFAULT (datetime('now'))
        )
    ''')

    # Table 2: patient_info — stores patient information
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patient_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL UNIQUE,
            full_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            contact TEXT,
            medical_history TEXT,
            created_by INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (created_by) REFERENCES user_record(id)
        )
    ''')

    # Table 3: diagnosis_history — stores all diagnostic results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnosis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_db_id INTEGER NOT NULL,
            patient_name TEXT NOT NULL,
            patient_pid TEXT NOT NULL,
            diagnosis_type TEXT NOT NULL DEFAULT 'pneumonia',
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            doctor_notes TEXT DEFAULT '',
            performed_by INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            chat_user_query TEXT DEFAULT '',
            chat_ai_response TEXT DEFAULT '',
            chat_image_path TEXT DEFAULT NULL,
            referenced_history_ids TEXT DEFAULT '',
            FOREIGN KEY (patient_db_id) REFERENCES patient_info(id),
            FOREIGN KEY (performed_by) REFERENCES user_record(id)
        )
    ''')

    # Table 4: medical_chat_report — saved assistant outputs (doctor-edited, printable)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medical_chat_report (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER NOT NULL,
            mode TEXT NOT NULL DEFAULT 'normal',
            patient_ids_json TEXT NOT NULL DEFAULT '[]',
            patient_names TEXT NOT NULL DEFAULT '',
            ai_raw TEXT NOT NULL DEFAULT '',
            doctor_final TEXT NOT NULL,
            structured_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (doctor_id) REFERENCES user_record(id)
        )
    ''')

    # Migration: add columns if they don't exist yet (for existing DBs)
    try:
        cursor.execute("ALTER TABLE user_record ADD COLUMN avatar_path TEXT DEFAULT NULL")
    except: pass
    try:
        cursor.execute("ALTER TABLE user_record ADD COLUMN mobile_number TEXT DEFAULT ''")
    except: pass
    try:
        cursor.execute("ALTER TABLE user_record ADD COLUMN specialization TEXT DEFAULT ''")
    except: pass
    try:
        cursor.execute("ALTER TABLE user_record ADD COLUMN clinic_name TEXT DEFAULT ''")
    except: pass
    try:
        cursor.execute("ALTER TABLE user_record ADD COLUMN license_number TEXT DEFAULT ''")
    except: pass
    try:
        cursor.execute("ALTER TABLE user_record ADD COLUMN theme TEXT DEFAULT 'light'")
    except: pass
    try:
        cursor.execute("ALTER TABLE diagnosis_history ADD COLUMN doctor_notes TEXT DEFAULT ''")
    except: pass
    try:
        cursor.execute("ALTER TABLE diagnosis_history ADD COLUMN chat_user_query TEXT DEFAULT ''")
    except: pass
    try:
        cursor.execute("ALTER TABLE diagnosis_history ADD COLUMN chat_ai_response TEXT DEFAULT ''")
    except: pass
    try:
        cursor.execute("ALTER TABLE diagnosis_history ADD COLUMN chat_image_path TEXT DEFAULT NULL")
    except: pass
    try:
        cursor.execute("ALTER TABLE diagnosis_history ADD COLUMN referenced_history_ids TEXT DEFAULT ''")
    except: pass
    try:
        cursor.execute("ALTER TABLE diagnosis_history ADD COLUMN medical_report_id INTEGER DEFAULT NULL")
    except: pass
    try:
        cursor.execute("ALTER TABLE medical_chat_report ADD COLUMN user_query TEXT DEFAULT ''")
    except: pass
    try:
        cursor.execute("ALTER TABLE medical_chat_report ADD COLUMN referenced_history_ids TEXT DEFAULT ''")
    except: pass
    try:
        cursor.execute("ALTER TABLE medical_chat_report ADD COLUMN image_path TEXT DEFAULT NULL")
    except: pass

    conn.commit()
    conn.close()


def hash_password(password):
    """Hash a password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{pwd_hash}"


def verify_password(stored_hash, password):
    """Verify a password against a stored hash."""
    salt, pwd_hash = stored_hash.split(':')
    return hashlib.sha256((salt + password).encode()).hexdigest() == pwd_hash


def generate_patient_id():
    """Generate a unique patient ID like P-00001."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM patient_info")
    count = cursor.fetchone()['count']
    conn.close()
    return f"P-{str(count + 1).zfill(5)}"


def build_patient_demographics_only(patient_db_ids):
    """Patient profile + on-file medical history only (no specific past test rows)."""
    if not patient_db_ids:
        return ""
    conn = get_db()
    cursor = conn.cursor()
    blocks = []
    for pid in patient_db_ids:
        cursor.execute("SELECT * FROM patient_info WHERE id = ?", (pid,))
        p = cursor.fetchone()
        if not p:
            continue
        blocks.append(
            f"PATIENT: {p['full_name']} — ID {p['patient_id']}, age {p['age']}, {p['gender']}\n"
            f"Contact: {p['contact'] or '—'}\n"
            f"Medical history on file: {p['medical_history'] or '—'}\n"
            f"Selected past tests / labs: none (doctor chose profile-only; use demographics and history above only)."
        )
    conn.close()
    if not blocks:
        return ""
    return (
        "LINKED PATIENT CONTEXT (integrate with the doctor's question and any attached image):\n\n"
        + "\n\n—\n\n".join(blocks)
    )


def _parse_id_list(raw):
    """Parse comma-separated integer IDs from form/query strings."""
    if not raw:
        return []
    out = []
    for x in str(raw).split(","):
        x = x.strip()
        if x.isdigit():
            out.append(int(x))
    return out


def _report_link_hints(cursor, referenced_history_ids="", medical_report_id=None):
    """Describe links between reports and prior history rows."""
    hints = []
    ref_raw = (referenced_history_ids or "").strip()
    if ref_raw:
        ref_ids = _parse_id_list(ref_raw.replace(" ", ","))
        if ref_ids:
            qh = ",".join("?" * len(ref_ids))
            cursor.execute(
                f"""SELECT id, diagnosis_type, result, created_at, medical_report_id
                    FROM diagnosis_history WHERE id IN ({qh})""",
                ref_ids,
            )
            for row in cursor.fetchall():
                r = dict(row)
                link = f"history record #{r['id']} ({r['diagnosis_type']} → {r['result']}, {r['created_at']})"
                if r.get("medical_report_id"):
                    link += f" [linked prescription report #{r['medical_report_id']}]"
                hints.append(link)
    if medical_report_id:
        cursor.execute(
            """SELECT id, referenced_history_ids FROM medical_chat_report WHERE id = ?""",
            (medical_report_id,),
        )
        rep = cursor.fetchone()
        if rep and (rep["referenced_history_ids"] or "").strip():
            hints.append(
                f"prescription report #{medical_report_id} references prior context IDs: {rep['referenced_history_ids']}"
            )
    if not hints:
        return ""
    return "\n    Report links: " + "; ".join(hints)


def build_lab_context_from_selected_history(patient_db_ids, history_ids):
    """Only the diagnosis_history rows the doctor ticked (must belong to selected patients)."""
    if not patient_db_ids:
        return ""
    if not history_ids:
        return ""

    conn = get_db()
    cursor = conn.cursor()
    qh = ",".join("?" * len(history_ids))
    qp = ",".join("?" * len(patient_db_ids))
    cursor.execute(
        f"""SELECT * FROM diagnosis_history
            WHERE id IN ({qh}) AND patient_db_id IN ({qp})
            ORDER BY datetime(created_at) DESC""",
        [*history_ids, *patient_db_ids],
    )
    rows = [dict(r) for r in cursor.fetchall()]
    by_patient = {}
    for r in rows:
        by_patient.setdefault(r["patient_db_id"], []).append(r)

    blocks = []
    for pid in patient_db_ids:
        cursor.execute("SELECT * FROM patient_info WHERE id = ?", (pid,))
        p = cursor.fetchone()
        if not p:
            continue
        pr_rows = by_patient.get(pid, [])
        hist_lines = []
        for r in pr_rows:
            conf_pct = float(r["confidence"] or 0) * 100.0
            note = (r["doctor_notes"] or "").strip()
            if len(note) > 400:
                note = note[:400] + "…"
            link_hint = _report_link_hints(
                cursor,
                r.get("referenced_history_ids"),
                r.get("medical_report_id"),
            )
            hist_lines.append(
                f"  • [Diagnosis / test record #{r['id']}] {r['created_at']}: {r['diagnosis_type']} → {r['result']} ({conf_pct:.1f}% conf)"
                + (f"\n    Clinical notes: {note}" if note else "")
                + link_hint
            )
            if r.get("diagnosis_type") == "ai_chat":
                q = (r.get("chat_user_query") or "").strip()
                if q:
                    hist_lines[-1] += f"\n    Doctor query: {q[:300]}"
                ai = (r.get("chat_ai_response") or "").strip()
                if ai:
                    hist_lines[-1] += f"\n    AI response excerpt: {ai[:800]}"
        sel_block = "\n".join(hist_lines) if hist_lines else (
            "  (No matching records for this patient among your selection — check selections.)"
        )
        blocks.append(
            f"PATIENT: {p['full_name']} — ID {p['patient_id']}, age {p['age']}, {p['gender']}\n"
            f"Contact: {p['contact'] or '—'}\n"
            f"Medical history on file: {p['medical_history'] or '—'}\n"
            f"SELECTED diagnoses / tests (use for this consult):\n{sel_block}"
        )
    conn.close()
    if not blocks:
        return ""
    return (
        "SELECTED DIAGNOSIS / TEST CONTEXT (integrate with the doctor's question and any attached image):\n\n"
        + "\n\n—\n\n".join(blocks)
    )


def build_lab_context_from_selected_prescriptions(patient_db_ids, prescription_ids, doctor_id):
    """Saved medical_chat_report rows the doctor selected (must belong to selected patients)."""
    if not patient_db_ids or not prescription_ids:
        return ""

    conn = get_db()
    cursor = conn.cursor()
    qp = ",".join("?" * len(prescription_ids))
    cursor.execute(
        f"""SELECT * FROM medical_chat_report
            WHERE doctor_id = ? AND id IN ({qp})
            ORDER BY datetime(created_at) DESC""",
        [doctor_id, *prescription_ids],
    )
    reports = [dict(r) for r in cursor.fetchall()]
    by_patient = {pid: [] for pid in patient_db_ids}

    for rep in reports:
        try:
            pids = json.loads(rep.get("patient_ids_json") or "[]")
        except (TypeError, ValueError):
            pids = []
        for pid in patient_db_ids:
            if pid in pids:
                by_patient.setdefault(pid, []).append(rep)

    blocks = []
    for pid in patient_db_ids:
        cursor.execute("SELECT * FROM patient_info WHERE id = ?", (pid,))
        p = cursor.fetchone()
        if not p:
            continue
        pr_reps = by_patient.get(pid, [])
        presc_lines = []
        for rep in pr_reps:
            final_text = (rep.get("doctor_final") or "").strip()
            if len(final_text) > 1500:
                final_text = final_text[:1500] + "…"
            link_hint = _report_link_hints(
                cursor, rep.get("referenced_history_ids"), rep.get("id")
            )
            user_q = (rep.get("user_query") or "").strip()
            presc_lines.append(
                f"  • [Prescription report #{rep['id']}] {rep['created_at']}"
                + (f"\n    Original doctor query: {user_q[:300]}" if user_q else "")
                + link_hint
                + f"\n    Prescription / report body:\n{final_text}"
            )
        sel_block = "\n".join(presc_lines) if presc_lines else (
            "  (No matching prescriptions for this patient among your selection.)"
        )
        blocks.append(
            f"PATIENT: {p['full_name']} — ID {p['patient_id']}, age {p['age']}, {p['gender']}\n"
            f"Contact: {p['contact'] or '—'}\n"
            f"Medical history on file: {p['medical_history'] or '—'}\n"
            f"SELECTED prescriptions (use for this consult):\n{sel_block}"
        )
    conn.close()
    if not blocks:
        return ""
    return (
        "SELECTED PRESCRIPTION CONTEXT (integrate with the doctor's question and any attached image):\n\n"
        + "\n\n—\n\n".join(blocks)
    )


def build_lab_context_mixed(patient_db_ids, history_ids, prescription_ids, doctor_id):
    """Combine selected diagnosis/test rows and saved prescriptions."""
    parts = []
    if history_ids:
        hist = build_lab_context_from_selected_history(patient_db_ids, history_ids)
        if hist:
            parts.append(hist)
    if prescription_ids:
        presc = build_lab_context_from_selected_prescriptions(
            patient_db_ids, prescription_ids, doctor_id
        )
        if presc:
            parts.append(presc)
    if not parts:
        return build_patient_demographics_only(patient_db_ids)
    return (
        "LINKED PATIENT CONTEXT — use ALL selected diagnoses/tests AND prescriptions below "
        "together with the doctor's current question:\n\n"
        + "\n\n".join(parts)
    )


def build_lab_context_last_prescription(patient_db_ids, doctor_id):
    """Latest saved AI prescription per linked patient."""
    if not patient_db_ids:
        return ""
    import json
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT * FROM medical_chat_report WHERE doctor_id = ?
           ORDER BY datetime(created_at) DESC""",
        (doctor_id,),
    )
    all_reports = [dict(r) for r in cursor.fetchall()]

    blocks = []
    for pid in patient_db_ids:
        cursor.execute("SELECT * FROM patient_info WHERE id = ?", (pid,))
        p = cursor.fetchone()
        if not p:
            continue
        latest = None
        for rep in all_reports:
            try:
                pids = json.loads(rep.get("patient_ids_json") or "[]")
            except (TypeError, ValueError):
                pids = []
            if pid in pids:
                latest = rep
                break
        if latest:
            final_text = (latest.get("doctor_final") or "").strip()
            if len(final_text) > 1500:
                final_text = final_text[:1500] + "…"
            presc_block = (
                f"LAST SAVED PRESCRIPTION (report #{latest['id']}, {latest['created_at']}):\n"
                f"{final_text}"
            )
        else:
            presc_block = "LAST SAVED PRESCRIPTION: none on file for this patient."
        blocks.append(
            f"PATIENT: {p['full_name']} — ID {p['patient_id']}, age {p['age']}, {p['gender']}\n"
            f"Contact: {p['contact'] or '—'}\n"
            f"Medical history on file: {p['medical_history'] or '—'}\n"
            f"CONTEXT MODE: Follow-up based on last prescription — tailor your reply to continue or adjust this plan.\n"
            f"{presc_block}"
        )
    conn.close()
    if not blocks:
        return ""
    return (
        "LINKED PATIENT CONTEXT (integrate with the doctor's question and any attached image):\n\n"
        + "\n\n—\n\n".join(blocks)
    )


def build_lab_context_last_visit(patient_db_ids):
    """Most recent diagnosis_history row per patient (any type)."""
    if not patient_db_ids:
        return ""
    conn = get_db()
    cursor = conn.cursor()
    blocks = []
    for pid in patient_db_ids:
        cursor.execute("SELECT * FROM patient_info WHERE id = ?", (pid,))
        p = cursor.fetchone()
        if not p:
            continue
        cursor.execute(
            """SELECT * FROM diagnosis_history WHERE patient_db_id = ?
               ORDER BY datetime(created_at) DESC LIMIT 1""",
            (pid,),
        )
        row = cursor.fetchone()
        if row:
            r = dict(row)
            conf_pct = float(r.get("confidence") or 0) * 100.0
            visit_block = (
                f"LAST VISIT ({r['created_at']}):\n"
                f"  Type: {r['diagnosis_type']}\n"
                f"  Result: {r['result']} ({conf_pct:.1f}% conf)"
            )
            if r.get("chat_user_query"):
                visit_block += f"\n  Doctor query: {(r['chat_user_query'] or '')[:400]}"
            if r.get("chat_ai_response"):
                ai_snip = (r["chat_ai_response"] or "")[:800]
                visit_block += f"\n  AI response excerpt: {ai_snip}"
            note = (r.get("doctor_notes") or "").strip()
            if note:
                visit_block += f"\n  Notes: {note[:400]}"
        else:
            visit_block = "LAST VISIT: no prior records for this patient."
        blocks.append(
            f"PATIENT: {p['full_name']} — ID {p['patient_id']}, age {p['age']}, {p['gender']}\n"
            f"Contact: {p['contact'] or '—'}\n"
            f"Medical history on file: {p['medical_history'] or '—'}\n"
            f"CONTEXT MODE: Follow-up based on last visit — relate your answer to this prior encounter.\n"
            f"{visit_block}"
        )
    conn.close()
    if not blocks:
        return ""
    return (
        "LINKED PATIENT CONTEXT (integrate with the doctor's question and any attached image):\n\n"
        + "\n\n—\n\n".join(blocks)
    )


# Initialize database on startup
init_db()

# ─── MODEL LOADING (LAZY LOADED) ────────────────────────────────────────────────
_PNEUMONIA_MODEL = None
_STROKE_MODEL = None
_SKIN_MODEL = None

def _load_keras_model(path):
    """Load a Keras model using TensorFlow or standalone Keras fallback."""
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        try:
            import keras
            return keras.models.load_model(path)
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            return None


def get_pneumonia_model():
    """Lazily load the pneumonia detection model."""
    global _PNEUMONIA_MODEL
    if _PNEUMONIA_MODEL is None:
        try:
            model_path = os.path.join(base_dir, 'model', 'pneumonia_classification_model.h5')
            if os.path.exists(model_path):
                _PNEUMONIA_MODEL = _load_keras_model(model_path)
        except Exception as e:
            print(f"Error loading pneumonia model: {e}")
    return _PNEUMONIA_MODEL

def stroke_prediction_load():
    """Lazily load the stroke prediction Gradient Boosting model."""
    global _STROKE_MODEL
    if _STROKE_MODEL is None:
        
        try:
            model_path = os.path.join(base_dir, 'model', 'Gradient_Boosting_stroke_model.pkl')
            if os.path.exists(model_path):
                _STROKE_MODEL = joblib.load(model_path)
            else:
                # Try fallback filename (e.g., the copy if primary is missing)
                alt_path = os.path.join(base_dir, 'model', 'Gradient_Boosting_stroke_model copy.pkl')
                if os.path.exists(alt_path):
                    _STROKE_MODEL = joblib.load(alt_path)

        except Exception as e:
            print(f"Error loading stroke model: {e}")
    return _STROKE_MODEL


def get_skin_model():
    """Lazily load the skin disease classification model."""
    global _SKIN_MODEL
    if _SKIN_MODEL is None:
        try:
            model_path = os.path.join(base_dir, 'model', 'skin_disease_final_model_2.h5')
            if os.path.exists(model_path):
                _SKIN_MODEL = _load_keras_model(model_path)
        except Exception as e:
            print(f"Error loading skin disease model: {e}")
    return _SKIN_MODEL


def preprocess_image(file_bytes):
    """Preprocess uploaded X-ray to match training pipeline from notebook."""
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode != 'L':
        img = img.convert('L')
    img = img.resize((150, 150))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = img_array.reshape(150, 150, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def preprocess_skin_image(file_bytes):
    """Preprocess skin lesion image for ResNet50-based skin disease model (224x224 RGB)."""
    from tensorflow.keras.applications.resnet50 import preprocess_input
    
    # Load image and convert to RGB
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to ResNet50 standard input size
    img = img.resize((224, 224))
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply ResNet50 preprocessing (mean subtraction + BGR conversion)
    img_array = preprocess_input(img_array)
    
    return img_array


# Skin disease class mapping (must match your training notebook)
SKIN_DISEASE_CLASSES = [
    "Atopic Dermatitis",
    "Basal Cell Carcinoma (BCC)",
    "Benign Keratosis-like Lesions (BKL)",
    "Eczema",
    "Fungal Infections (Tinea, Ringworm, Candidiasis)",
    "Melanocytic Nevi (NV)",
    "Melanoma",
    "Psoriasis and Related Diseases",
    "Seborrheic Keratoses and Benign Tumors",
    "Viral Infections (Warts, Molluscum)"
]


# ─── STATIC FILE ROUTES ────────────────────────────────────────────────────────

@app.route('/')
def home():
    return app.send_static_file('index.html')


@app.route('/favicon.ico')
def favicon():
    """Ensure favicon.ico is always served to avoid 404 errors."""
    try:
        return app.send_static_file('favicon.ico')
    except:
        # Serve a minimal transparent PNG if file doesn't exist
        from base64 import b64decode
        return b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='), 200, {'Content-Type': 'image/png', 'Cache-Control': 'public, max-age=86400'}


@app.after_request
def add_header(response):
    """Ensure root requests return fresh responses without 304 caching."""
    if request.path == '/':
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
    return response


@app.route('/signup.html')
def signup():
    return render_template('signup.html')

@app.route('/dashboard.html')
def dashboard():
    return render_template('dashboard.html')

@app.route('/history.html')
def history():
    return render_template('history.html')

@app.route('/pneumonia_prediction.html')
def route_pneumonia():
    return render_template('pneumonia_prediction.html')

@app.route('/stroke_prediction.html')
def route_stroke():
    return render_template('stroke_prediction.html')

@app.route('/skin_prediction.html')
def route_skin():
    return render_template('skin_prediction.html')

@app.route('/profile.html')
def profile():
    return render_template('profile.html')

@app.route('/diagnostics.html')
def diagnostics():
    return render_template('diagnostics.html')

@app.route('/add_patient.html')
def add_patient():
    return render_template('add_patient.html')

@app.route('/patients.html')
def patients_page():
    return render_template('patients.html')


@app.route('/chat_bot.html')
def chat_bot_page():
    return render_template('chat_bot.html')


@app.route('/medical_reports.html')
def medical_reports_page():
    return render_template('medical_reports.html')


@app.route('/medical_report_print.html')
def medical_report_print_page():
    return render_template('medical_report_print.html')


@app.route('/<path:path>')
def catch_all(path):
    if path.startswith('templates/'):
        path = path[len('templates/'):]
    frontend_path = os.path.join(app.static_folder, path)
    if os.path.exists(frontend_path) and os.path.isfile(frontend_path):
        return send_from_directory(app.static_folder, path)
    template_path = os.path.join(app.static_folder, 'templates', path)
    if os.path.exists(template_path) and os.path.isfile(template_path):
        return send_from_directory(os.path.join(app.static_folder, 'templates'), path)
    return abort(404)


# ─── AUTH API ENDPOINTS ─────────────────────────────────────────────────────────

@app.route('/api/signup', methods=['POST'])
def api_signup():
    """Register a new user account."""
    data = request.get_json()
    full_name = data.get('full_name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not full_name or not email or not password:
        return jsonify({'error': 'All fields are required.'}), 400

    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters.'}), 400

    conn = get_db()
    cursor = conn.cursor()

    # Check if email already exists
    cursor.execute("SELECT id FROM user_record WHERE email = ?", (email,))
    if cursor.fetchone():
        conn.close()
        return jsonify({'error': 'An account with this email already exists.'}), 409

    pwd_hash = hash_password(password)
    cursor.execute(
        "INSERT INTO user_record (full_name, email, password_hash) VALUES (?, ?, ?)",
        (full_name, email, pwd_hash)
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()

    session['user_id'] = user_id
    session['user_name'] = full_name
    session['user_email'] = email

    return jsonify({
        'message': 'Account created successfully.',
        'user': {'id': user_id, 'full_name': full_name, 'email': email}
    }), 201


@app.route('/api/login', methods=['POST'])
def api_login():
    """Login an existing user."""
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({'error': 'Email and password are required.'}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_record WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if not user or not verify_password(user['password_hash'], password):
        return jsonify({'error': 'Invalid email or password.'}), 401

    session['user_id'] = user['id']
    session['user_name'] = user['full_name']
    session['user_email'] = user['email']

    return jsonify({
        'message': 'Login successful.',
        'user': {
            'id': user['id'],
            'full_name': user['full_name'],
            'email': user['email']
        }
    }), 200


@app.route('/api/logout', methods=['POST'])
def api_logout():
    """Logout the current user."""
    session.clear()
    return jsonify({'message': 'Logged out successfully.'}), 200


@app.route('/api/me', methods=['GET'])
def api_me():
    """Get current logged-in user info."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, full_name, email, avatar_path, mobile_number, specialization, clinic_name, license_number, theme, created_at FROM user_record WHERE id = ?", (session['user_id'],))
    user = cursor.fetchone()
    conn.close()

    if not user:
        session.clear()
        return jsonify({'error': 'User not found.'}), 404

    avatar_url = f"/api/me/avatar/{user['avatar_path']}" if user['avatar_path'] else None

    return jsonify({
        'user': {
            'id': user['id'],
            'full_name': user['full_name'],
            'email': user['email'],
            'mobile_number': user['mobile_number'] or '',
            'specialization': user['specialization'] or '',
            'clinic_name': user['clinic_name'] or '',
            'license_number': user['license_number'] or '',
            'avatar_url': avatar_url,
            'theme': user['theme'] or 'light',
            'created_at': user['created_at']
        }
    }), 200


@app.route('/api/me/update', methods=['PUT'])
def api_update_profile():
    """Update current user's profile name."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401

    data = request.get_json()
    full_name = data.get('full_name', '').strip()
    mobile_number = data.get('mobile_number', '').strip()
    specialization = data.get('specialization', '').strip()
    clinic_name = data.get('clinic_name', '').strip()
    license_number = data.get('license_number', '').strip()

    if not full_name:
        return jsonify({'error': 'Name cannot be empty.'}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE user_record 
        SET full_name = ?, mobile_number = ?, specialization = ?, clinic_name = ?, license_number = ?
        WHERE id = ?
    """, (full_name, mobile_number, specialization, clinic_name, license_number, session['user_id']))
    conn.commit()
    conn.close()

    session['user_name'] = full_name

    return jsonify({'message': 'Profile updated successfully.'}), 200


@app.route('/api/me/password', methods=['PUT'])
def api_update_password():
    """Update current user's password."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401

    data = request.get_json()
    current_password = data.get('current_password', '')
    new_password = data.get('new_password', '')

    if not current_password or not new_password:
        return jsonify({'error': 'Both current and new password are required.'}), 400

    if len(new_password) < 6:
        return jsonify({'error': 'New password must be at least 6 characters.'}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM user_record WHERE id = ?", (session['user_id'],))
    user = cursor.fetchone()

    if not user or not verify_password(user['password_hash'], current_password):
        conn.close()
        return jsonify({'error': 'Current password is incorrect.'}), 401

    new_hash = hash_password(new_password)
    cursor.execute("UPDATE user_record SET password_hash = ? WHERE id = ?", (new_hash, session['user_id']))
    conn.commit()
    conn.close()

    return jsonify({'message': 'Password updated successfully.'}), 200


@app.route('/api/me/deactivate', methods=['DELETE'])
def api_deactivate():
    """Deactivate (delete) the current user's account."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401

    user_id = session['user_id']
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM diagnosis_history WHERE performed_by = ?", (user_id,))
    cursor.execute("DELETE FROM user_record WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

    session.clear()
    return jsonify({'message': 'Account deactivated.'}), 200


@app.route('/api/history/<int:record_id>', methods=['DELETE'])
def api_delete_diagnosis(record_id):
    """Delete a diagnostic record; cascades linked AI chat report everywhere."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT medical_report_id FROM diagnosis_history WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Record not found.'}), 404

    report_id = row['medical_report_id']
    if report_id:
        cursor.execute("DELETE FROM diagnosis_history WHERE medical_report_id = ?", (report_id,))
        cursor.execute(
            "DELETE FROM medical_chat_report WHERE id = ? AND doctor_id = ?",
            (report_id, session['user_id']),
        )
    else:
        cursor.execute("DELETE FROM diagnosis_history WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Record deleted successfully.'}), 200


@app.route('/api/me/avatar', methods=['POST'])
def api_upload_avatar():
    """Upload a profile picture."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401

    if 'avatar' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    file = request.files['avatar']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    # Validate file type
    allowed = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in allowed:
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, GIF, or WebP.'}), 400

    # Save with unique name
    filename = f"{session['user_id']}_{uuid.uuid4().hex[:8]}.{ext}"
    filepath = os.path.join(uploads_dir, filename)

    # Delete old avatar if exists
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT avatar_path FROM user_record WHERE id = ?", (session['user_id'],))
    old = cursor.fetchone()
    if old and old['avatar_path']:
        old_path = os.path.join(uploads_dir, old['avatar_path'])
        if os.path.exists(old_path):
            os.remove(old_path)

    file.save(filepath)
    cursor.execute("UPDATE user_record SET avatar_path = ? WHERE id = ?", (filename, session['user_id']))
    conn.commit()
    conn.close()

    return jsonify({
        'message': 'Avatar uploaded successfully.',
        'avatar_url': f'/api/me/avatar/{filename}'
    }), 200


@app.route('/api/me/avatar', methods=['DELETE'])
def api_delete_avatar():
    """Remove current profile picture."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT avatar_path FROM user_record WHERE id = ?", (session['user_id'],))
    res = cursor.fetchone()

    if res and res['avatar_path']:
        filepath = os.path.join(uploads_dir, res['avatar_path'])
        if os.path.exists(filepath):
            os.remove(filepath)
        
        cursor.execute("UPDATE user_record SET avatar_path = NULL WHERE id = ?", (session['user_id'],))
        conn.commit()
    
    conn.close()
    return jsonify({'message': 'Avatar deleted successfully.'}), 200


@app.route('/api/me/avatar/<filename>')
def api_serve_avatar(filename):
    """Serve avatar image."""
    return send_from_directory(uploads_dir, filename)


@app.route('/api/me/theme', methods=['PUT'])
def api_update_theme():
    """Update user theme preference."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401

    data = request.get_json()
    theme = data.get('theme', 'light')
    if theme not in ('light', 'dark'):
        theme = 'light'

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE user_record SET theme = ? WHERE id = ?", (theme, session['user_id']))
    conn.commit()
    conn.close()

    return jsonify({'message': 'Theme updated.', 'theme': theme}), 200


# ─── DIAGNOSIS EDIT ENDPOINT ────────────────────────────────────────────────────

@app.route('/api/history/<int:record_id>', methods=['PUT'])
def api_edit_diagnosis(record_id):
    """Edit a diagnosis record (result, notes)."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401

    data = request.get_json()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT diagnosis_type FROM diagnosis_history WHERE id = ?", (record_id,))
    row0 = cursor.fetchone()
    is_ai_chat = row0 and row0['diagnosis_type'] == 'ai_chat'

    # Build dynamic update
    updates = []
    values = []

    if 'result' in data:
        result = data['result']
        if is_ai_chat and isinstance(result, str) and result.strip():
            updates.append('result = ?')
            values.append(result.strip()[:500])
        elif result in ('pneumonia', 'normal', 'high risk (stroke)', 'moderate risk (stroke)', 'normal (low risk) (stroke)') or result in SKIN_DISEASE_CLASSES:
            updates.append('result = ?')
            values.append(result)

    if 'doctor_notes' in data:
        updates.append('doctor_notes = ?')
        values.append(data['doctor_notes'])

    if not updates:
        conn.close()
        return jsonify({'error': 'No valid fields to update.'}), 400

    values.append(record_id)
    cursor.execute(f"UPDATE diagnosis_history SET {', '.join(updates)} WHERE id = ?", values)
    conn.commit()
    conn.close()

    return jsonify({'message': 'Record updated successfully.'}), 200


# ─── PATIENT API ENDPOINTS ──────────────────────────────────────────────────────

@app.route('/api/patients', methods=['GET'])
def api_get_patients():
    """Get all patients, optionally filtered by search query."""
    search = request.args.get('search', '').strip()
    conn = get_db()
    cursor = conn.cursor()

    if search:
        cursor.execute(
            """SELECT * FROM patient_info 
               WHERE full_name LIKE ? OR patient_id LIKE ? OR contact LIKE ?
               ORDER BY created_at DESC""",
            (f'%{search}%', f'%{search}%', f'%{search}%')
        )
    else:
        cursor.execute("SELECT * FROM patient_info ORDER BY created_at DESC")

    patients = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify({'patients': patients}), 200


@app.route('/api/patients/<int:patient_db_id>', methods=['GET'])
def api_get_patient(patient_db_id):
    """Get a single patient by database ID."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patient_info WHERE id = ?", (patient_db_id,))
    patient = cursor.fetchone()
    conn.close()

    if not patient:
        return jsonify({'error': 'Patient not found.'}), 404

    return jsonify({'patient': dict(patient)}), 200


@app.route('/api/patients', methods=['POST'])
def api_add_patient():
    """Add a new patient record."""
    data = request.get_json()
    full_name = data.get('full_name', '').strip()
    age = data.get('age')
    gender = data.get('gender', '').strip()
    contact = data.get('contact', '').strip()
    medical_history = data.get('medical_history', '').strip()

    if not full_name or age is None or not gender:
        return jsonify({'error': 'Name, age, and gender are required.'}), 400

    try:
        age = int(age)
        if age < 0 or age > 150:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({'error': 'Age must be a valid number between 0 and 150.'}), 400

    patient_id = generate_patient_id()
    created_by = session.get('user_id')

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO patient_info 
           (patient_id, full_name, age, gender, contact, medical_history, created_by)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (patient_id, full_name, age, gender, contact, medical_history, created_by)
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()

    return jsonify({
        'message': 'Patient registered successfully.',
        'patient': {
            'id': new_id,
            'patient_id': patient_id,
            'full_name': full_name,
            'age': age,
            'gender': gender,
            'contact': contact,
            'medical_history': medical_history
        }
    }), 201


@app.route('/api/patients/count', methods=['GET'])
def api_patient_count():
    """Get total number of patients."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM patient_info")
    count = cursor.fetchone()['count']
    conn.close()
    return jsonify({'count': count}), 200


@app.route('/api/patients/next-id', methods=['GET'])
def api_next_patient_id():
    """Get the next auto-generated patient ID."""
    return jsonify({'patient_id': generate_patient_id()}), 200


# ─── DIAGNOSIS HISTORY ENDPOINTS ──────────────────────────────────────────────

@app.route('/api/history', methods=['GET'])
def api_get_history():
    """Get diagnostic history, optionally filtered by search query."""
    search = request.args.get('search', '').strip()
    conn = get_db()
    cursor = conn.cursor()

    if search:
        cursor.execute(
            """SELECT * FROM diagnosis_history 
               WHERE patient_name LIKE ? OR patient_pid LIKE ? OR result LIKE ?
               ORDER BY created_at DESC""",
            (f'%{search}%', f'%{search}%', f'%{search}%')
        )
    else:
        cursor.execute("SELECT * FROM diagnosis_history ORDER BY created_at DESC")

    history = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify({'history': history}), 200


@app.route('/api/history/stats', methods=['GET'])
def api_history_stats():
    """Get diagnostic history statistics."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total FROM diagnosis_history")
    total = cursor.fetchone()['total']
    cursor.execute("SELECT COUNT(*) as count FROM diagnosis_history WHERE result = 'pneumonia'")
    pneumonia = cursor.fetchone()['count']
    cursor.execute("SELECT COUNT(*) as count FROM diagnosis_history WHERE result = 'normal'")
    normal = cursor.fetchone()['count']
    conn.close()

    return jsonify({'total': total, 'pneumonia': pneumonia, 'normal': normal}), 200


@app.route('/api/patients/prescriptions', methods=['GET'])
def api_patient_prescriptions():
    """List saved AI prescriptions (medical_chat_report) for selected patients."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401

    pids = _parse_id_list(request.args.get('patient_ids', ''))
    if not pids:
        return jsonify({'prescriptions': []}), 200

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT id, mode, patient_ids_json, patient_names, created_at,
                  user_query, referenced_history_ids, structured_json, doctor_final
           FROM medical_chat_report WHERE doctor_id = ?
           ORDER BY datetime(created_at) DESC, id DESC""",
        (session['user_id'],),
    )
    all_rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    out = []
    for rep in all_rows:
        try:
            rep_pids = json.loads(rep.get("patient_ids_json") or "[]")
        except (TypeError, ValueError):
            rep_pids = []
        if not any(pid in rep_pids for pid in pids):
            continue
        summary = ""
        try:
            structured = json.loads(rep.get("structured_json") or "{}")
            summary = (structured.get("primary_diagnosis") or "").strip()
        except (TypeError, ValueError):
            pass
        if not summary:
            summary = (rep.get("doctor_final") or "").strip().split("\n")[0][:120]
        out.append({
            "id": rep["id"],
            "created_at": rep["created_at"],
            "patient_names": rep.get("patient_names") or "",
            "patient_ids": rep_pids,
            "user_query": rep.get("user_query") or "",
            "referenced_history_ids": rep.get("referenced_history_ids") or "",
            "summary": summary,
        })
    return jsonify({'prescriptions': out}), 200


@app.route('/api/patients/<int:patient_db_id>/history', methods=['GET'])
def api_patient_history(patient_db_id):
    """Get all diagnosis records for a specific patient."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT * FROM diagnosis_history 
           WHERE patient_db_id = ?
           ORDER BY created_at DESC""",
        (patient_db_id,)
    )
    history = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify({'history': history}), 200


# ─── PREDICTION ENDPOINT ───────────────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    loaded_model = get_pneumonia_model()
    if loaded_model is None:
        return jsonify({'error': 'Model not loaded or model file missing'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Get patient info from form data (sent alongside file)
    patient_db_id = request.form.get('patient_db_id')
    patient_name = request.form.get('patient_name', 'Unknown')
    patient_pid = request.form.get('patient_pid', '')

    try:
        file_bytes = file.read()
        import uuid
        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else 'png'
        image_filename = f"scan_{uuid.uuid4().hex[:12]}.{ext}"
        with open(os.path.join(chat_lab_dir, image_filename), 'wb') as out:
            out.write(file_bytes)
            
        img_array = preprocess_image(file_bytes)
        prediction = loaded_model.predict(img_array)
        score = float(prediction[0][0])

        if score >= 0.5:
            result = 'normal'
            display_score = score
        else:
            result = 'pneumonia'
            display_score = 1.0 - score

        # Save to diagnosis_history
        if patient_db_id:
            try:
                conn = get_db()
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO diagnosis_history 
                       (patient_db_id, patient_name, patient_pid, diagnosis_type, result, confidence, performed_by, chat_image_path)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (int(patient_db_id), patient_name, patient_pid, 'pneumonia', result, display_score, session.get('user_id'), image_filename)
                )
                conn.commit()
                conn.close()
            except Exception as db_err:
                print(f"Warning: Failed to save history: {db_err}")

        return jsonify({
            'prediction': result,
            'score': display_score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/skin-predict', methods=['POST'])
def skin_predict_api():
    """Predict skin disease risk from an uploaded lesion image."""
    loaded_model = get_skin_model()
    if loaded_model is None:
        return jsonify({'error': 'Skin disease model not loaded or model file missing'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    patient_db_id = request.form.get('patient_db_id')
    patient_name = request.form.get('patient_name', 'Unknown')
    patient_pid = request.form.get('patient_pid', '')

    try:
        file_bytes = file.read()
        import uuid
        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else 'png'
        image_filename = f"scan_{uuid.uuid4().hex[:12]}.{ext}"
        with open(os.path.join(chat_lab_dir, image_filename), 'wb') as out:
            out.write(file_bytes)

        img_array = preprocess_skin_image(file_bytes)
        prediction_probs = loaded_model.predict(img_array)
        
        # Get the class with highest probability
        predicted_class_idx = int(np.argmax(prediction_probs[0]))
        confidence_score = float(np.max(prediction_probs[0]))
        
        # Map class index to disease name
        disease_name = SKIN_DISEASE_CLASSES[predicted_class_idx] if predicted_class_idx < len(SKIN_DISEASE_CLASSES) else "Unknown Disease"
        result = disease_name
        display_score = confidence_score

        if patient_db_id:
            try:
                conn = get_db()
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO diagnosis_history 
                       (patient_db_id, patient_name, patient_pid, diagnosis_type, result, confidence, performed_by, chat_image_path)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (int(patient_db_id), patient_name, patient_pid, 'skin', result, display_score, session.get('user_id'), image_filename)
                )
                conn.commit()
                conn.close()
            except Exception as db_err:
                print(f"Warning: Failed to save history: {db_err}")

        return jsonify({
            'prediction': result,
            'score': display_score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stroke-predict', methods=['POST'])
def stroke_predict_api():
    """Predict stroke risk with categorical thresholds (<35, 36-70, >70)."""
    model = stroke_prediction_load()
    if model is None:
        return jsonify({"error": "Stroke model offline"}), 500

    try:
        data = request.json
        
        # Preprocessing
        gender_map = {"Male": 0, "Female": 1, "Other": 2}
        gender = gender_map.get(data.get('gender', 'Female'), 1)
        age = (float(data.get('age', 40)) - 43.2266) / 22.6126
        hypertension = 1 if data.get('hypertension') in [1, '1', True, 'Yes'] else 0
        heart_disease = 1 if data.get('heart_disease') in [1, '1', True, 'Yes'] else 0
        married_map = {"Yes": 0, "No": 1}; ever_married = married_map.get(data.get('ever_married', 'No'), 1)
        work_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}
        work_type = work_map.get(data.get('work_type', 'Private'), 0)
        res_map = {"Urban": 0, "Rural": 1}; residence_type = res_map.get(data.get('Residence_type', 'Urban'), 0)
        glucose_raw = float(data.get('avg_glucose_level', 100))
        avg_glucose_level = (glucose_raw - 106.1477) / 45.2836
        bmi_raw = float(data.get('bmi', 25))
        bmi = (bmi_raw - 28.8932) / 7.8541
        smoke_map = {"formerly smoked": 0, "never smoked": 1, "smokes": 2, "Unknown": 3}
        smoking_status = smoke_map.get(data.get('smoking_status', 'never smoked'), 1)

        input_df = pd.DataFrame([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]], 
                                columns=['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status'])
        
        # Inference
        probabilities = model.predict_proba(input_df)[0]
        risk_score = float(probabilities[1]) * 100
        
        # Thresholds (<35: Normal, 36-70: Moderate, >70: High)
        if risk_score <= 35:
            result_label, history_result = "Normal (Low Risk)", "normal (low risk)"
        elif risk_score <= 70:
            result_label, history_result = "Moderate Risk", "moderate risk"
        else:
            result_label, history_result = "High Risk", "high risk"
        
        # Log to DB
        p_db_id = data.get('patient_db_id')
        if p_db_id:
            try:
                conn = get_db(); cursor = conn.cursor()
                cursor.execute("SELECT full_name, patient_id FROM patient_info WHERE id = ?", (p_db_id,))
                p = cursor.fetchone()
                if p:
                    cursor.execute(
                        """INSERT INTO diagnosis_history (patient_db_id, patient_name, patient_pid, diagnosis_type, result, confidence, doctor_notes, performed_by, created_at) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (int(p_db_id), p['full_name'], p['patient_id'], 'stroke', f"{history_result} (stroke)", risk_score / 100.0, 
                         f"Biometric Analysis: glucose={glucose_raw}, bmi={bmi_raw}, smoking={data.get('smoking_status')}", 
                         session.get('user_id'), datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    )
                    conn.commit()
                conn.close()
            except Exception as e: print(f"Log Error: {e}")

        return jsonify({"success": True, "prediction": result_label, "confidence": round(risk_score, 1)})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400


@app.route('/api/chatbot/status', methods=['GET'])
def api_chatbot_status():
    """Check whether Gemini API key is loaded (logged-in users only)."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401
    try:
        from chatbot.service import get_chatbot_config_public
        return jsonify(get_chatbot_config_public()), 200
    except Exception:
        return jsonify({'api_configured': False, 'model': 'unknown'}), 200


@app.route('/api/chatbot', methods=['POST'])
def api_chatbot():
    """Medical assistant: text only, or text + optional image (multipart)."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401

    message = (request.form.get('message') or '').strip()
    if not message:
        return jsonify({'error': 'Message is required.'}), 400

    image_file = request.files.get('image')
    image_bytes = None
    if image_file and image_file.filename:
        allowed = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
        ext = image_file.filename.rsplit('.', 1)[-1].lower() if '.' in image_file.filename else ''
        if ext not in allowed:
            return jsonify({'error': 'Image must be PNG, JPG, GIF, or WebP.'}), 400
        image_bytes = image_file.read()
        if len(image_bytes) > 8 * 1024 * 1024:
            return jsonify({'error': 'Image too large (max 8 MB).'}), 400

    lab_context = (request.form.get('lab_context') or '').strip().lower() in ('1', 'true', 'yes', 'on')
    patient_ids_raw = (request.form.get('patient_ids') or '').strip()
    context_prefix = None
    if lab_context and patient_ids_raw:
        try:
            pids = [int(x.strip()) for x in patient_ids_raw.split(',') if x.strip().isdigit()]
        except ValueError:
            pids = []
        if pids:
            history_raw = (request.form.get('history_ids') or '').strip()
            hid = _parse_id_list(history_raw)
            presc_raw = (request.form.get('prescription_ids') or '').strip()
            presc_ids = _parse_id_list(presc_raw)
            profile_only = (request.form.get('profile_only_lab') or '').strip().lower() in ('1', 'true', 'yes', 'on')
            context_mode = (request.form.get('context_mode') or '').strip().lower()
            if context_mode == 'mixed' or (hid and presc_ids):
                if not hid and not presc_ids:
                    return jsonify({'error': 'Select at least one diagnosis or prescription.'}), 400
                context_prefix = build_lab_context_mixed(
                    pids, hid, presc_ids, session['user_id']
                )
            elif context_mode == 'prescriptions':
                if not presc_ids:
                    return jsonify({'error': 'Select at least one saved prescription.'}), 400
                context_prefix = build_lab_context_from_selected_prescriptions(
                    pids, presc_ids, session['user_id']
                )
            elif context_mode == 'tests':
                if not hid:
                    return jsonify({'error': 'Select at least one past test from history.'}), 400
                context_prefix = build_lab_context_from_selected_history(pids, hid)
            elif context_mode == 'last_prescription':
                context_prefix = build_lab_context_last_prescription(pids, session['user_id'])
            elif context_mode == 'last_visit':
                context_prefix = build_lab_context_last_visit(pids)
            elif profile_only or context_mode == 'profile_only':
                context_prefix = build_patient_demographics_only(pids)
            else:
                return jsonify({
                    'error': 'Choose past tests, prescriptions, profile only, or last visit.',
                }), 400

    try:
        from chatbot.service import run_medical_chat
        reply = run_medical_chat(message, image_bytes=image_bytes, context_prefix=context_prefix)
        return jsonify({'reply': reply}), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        import traceback
        traceback.print_exc()
        err = str(e).strip() or 'Chat request failed.'
        if len(err) > 600:
            err = err[:597] + '...'
        return jsonify({'error': err}), 503


@app.route('/api/uploads/chat-lab/<path:filename>')
def serve_chat_lab_image(filename):
    if 'user_id' not in session:
        abort(401)
    if '..' in filename or filename.startswith(('/', '\\')):
        abort(404)
    fp = os.path.join(chat_lab_dir, os.path.basename(filename))
    if not os.path.isfile(fp) or filename in ('unknown', 'unknown.png', 'none', 'null'):
        placeholder_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" width="100%" height="100%">
            <defs>
                <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#f8fafc;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#e2e8f0;stop-opacity:1" />
                </linearGradient>
            </defs>
            <rect width="100%" height="100%" fill="url(#grad)" rx="16"/>
            <g transform="translate(160, 80)" stroke="#94a3b8" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
                <rect x="0" y="0" width="80" height="60" rx="8" />
                <path d="M25 60 L15 80 L65 80 L55 60" />
                <path d="M10 30 L25 30 L30 15 L38 45 L44 25 L48 35 L53 30 L70 30" stroke="#3b82f6" stroke-dasharray="1 1" />
            </g>
            <text x="50%" y="195" text-anchor="middle" font-family="'Inter', sans-serif" font-size="14" font-weight="700" fill="#475569">No Scan Image Available</text>
            <text x="50%" y="215" text-anchor="middle" font-family="'Inter', sans-serif" font-size="11" font-weight="500" fill="#94a3b8">Biometric / Text-only Analysis Record</text>
        </svg>"""
        return placeholder_svg, 200, {'Content-Type': 'image/svg+xml', 'Cache-Control': 'public, max-age=86400'}
    return send_from_directory(chat_lab_dir, os.path.basename(filename))


@app.route('/api/history/chat-log', methods=['POST'])
def api_history_chat_log():
    """Save assistant chat (user text + AI reply + optional image) into diagnosis_history."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401

    patient_raw = request.form.get('patient_db_id')
    user_query = (request.form.get('user_query') or '').strip()
    ai_response = (request.form.get('ai_response') or '').strip()
    ref_ids = (request.form.get('referenced_history_ids') or '').strip()

    if not patient_raw or not user_query or not ai_response:
        return jsonify({'error': 'patient_db_id, user_query, and ai_response are required.'}), 400
    try:
        patient_db_id = int(patient_raw)
    except ValueError:
        return jsonify({'error': 'Invalid patient.'}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patient_info WHERE id = ?", (patient_db_id,))
    p = cursor.fetchone()
    if not p:
        conn.close()
        return jsonify({'error': 'Patient not found.'}), 404

    image_filename = None
    img_file = request.files.get('image')
    if img_file and img_file.filename:
        allowed = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
        ext = img_file.filename.rsplit('.', 1)[-1].lower() if '.' in img_file.filename else ''
        if ext in allowed:
            raw = img_file.read()
            if len(raw) <= 8 * 1024 * 1024:
                image_filename = f"{session['user_id']}_{uuid.uuid4().hex[:12]}.{ext}"
                with open(os.path.join(chat_lab_dir, image_filename), 'wb') as out:
                    out.write(raw)

    from chatbot.response_parser import parse_medical_report_text

    parsed = parse_medical_report_text(ai_response)
    result_line = (parsed.get('primary_diagnosis') or 'AI assistant consultation').strip()
    if len(result_line) > 160:
        result_line = result_line[:157] + '...'

    notes_parts = [
        'Source: Medical assistant chat.',
        f"Referenced history IDs: {ref_ids or 'none'}.",
    ]
    if image_filename:
        notes_parts.append(f"Attachment: /api/uploads/chat-lab/{image_filename}")
    doctor_notes = '\n'.join(notes_parts)

    cursor.execute(
        """INSERT INTO diagnosis_history
           (patient_db_id, patient_name, patient_pid, diagnosis_type, result, confidence,
            doctor_notes, performed_by, chat_user_query, chat_ai_response, chat_image_path, referenced_history_ids)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            patient_db_id,
            p['full_name'],
            p['patient_id'],
            'ai_chat',
            result_line,
            0.0,
            doctor_notes,
            session['user_id'],
            user_query,
            ai_response,
            image_filename,
            ref_ids or '',
        ),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return jsonify({'message': 'Logged to patient history.', 'id': new_id}), 201


@app.route('/api/medical-reports', methods=['GET'])
def api_medical_reports_list():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT id, mode, patient_names, created_at, structured_json, doctor_final
           FROM medical_chat_report WHERE doctor_id = ?
           ORDER BY datetime(created_at) DESC, id DESC""",
        (session['user_id'],),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return jsonify({'reports': rows}), 200


@app.route('/api/medical-reports/<int:report_id>', methods=['GET'])
def api_medical_report_get(report_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT * FROM medical_chat_report WHERE id = ? AND doctor_id = ?""",
        (report_id, session['user_id']),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return jsonify({'error': 'Report not found.'}), 404
    report = dict(row)
    patients = []
    try:
        pids = json.loads(report.get("patient_ids_json") or "[]")
    except (TypeError, ValueError):
        pids = []
    if pids:
        conn = get_db()
        cursor = conn.cursor()
        for pid in pids:
            cursor.execute(
                "SELECT id, patient_id, full_name, age, gender FROM patient_info WHERE id = ?",
                (pid,),
            )
            pr = cursor.fetchone()
            if pr:
                patients.append(dict(pr))
        conn.close()
    report["patients"] = patients
    if patients:
        primary = patients[0]
        report["patient_id"] = primary.get("patient_id")
        report["patient_age"] = primary.get("age")
        report["patient_gender"] = primary.get("gender")
    return jsonify({'report': report}), 200


@app.route('/api/medical-reports/<int:report_id>', methods=['DELETE', 'PUT'])
def api_medical_report_modify(report_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401
    
    conn = get_db()
    cursor = conn.cursor()
    
    if request.method == 'DELETE':
        cursor.execute(
            "DELETE FROM diagnosis_history WHERE medical_report_id = ?",
            (report_id,),
        )
        cursor.execute(
            "DELETE FROM medical_chat_report WHERE id = ? AND doctor_id = ?",
            (report_id, session['user_id']),
        )
        conn.commit()
        deleted = cursor.rowcount
        conn.close()
        if not deleted:
            return jsonify({'error': 'Report not found.'}), 404
        return jsonify({'message': 'Deleted from all sections.'}), 200
        
    if request.method == 'PUT':
        data = request.get_json() or {}
        doctor_final = (data.get('doctor_final') or '').strip()
        if not doctor_final:
            return jsonify({'error': 'Report text cannot be empty.'}), 400
            
        from chatbot.service import extract_structured_with_ai
        structured = extract_structured_with_ai(doctor_final)
        if not structured:
            from chatbot.response_parser import parse_medical_report_text
            structured = parse_medical_report_text(doctor_final)
            
        import json
        structured_json = json.dumps(structured, ensure_ascii=False)
        
        cursor.execute(
            "UPDATE medical_chat_report SET doctor_final = ?, structured_json = ? WHERE id = ? AND doctor_id = ?",
            (doctor_final, structured_json, report_id, session['user_id'])
        )
        cursor.execute(
            """UPDATE diagnosis_history SET chat_ai_response = ?, result = ?
               WHERE medical_report_id = ?""",
            (
                doctor_final,
                (structured.get('primary_diagnosis') or 'AI assistant consultation')[:160],
                report_id,
            ),
        )
        conn.commit()
        updated = cursor.rowcount
        conn.close()
        if not updated:
            return jsonify({'error': 'Report not found.'}), 404
        return jsonify({'message': 'Report updated successfully.', 'structured': structured}), 200


@app.route('/api/medical-reports', methods=['POST'])
def api_medical_report_save():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401
    
    ai_raw = (request.form.get('ai_raw') or '').strip()
    doctor_final = (request.form.get('doctor_final') or '').strip()
    if not doctor_final:
        return jsonify({'error': 'Report text is required.'}), 400
    
    mode = request.form.get('mode') or 'normal'
    if mode not in ('normal', 'lab_context'):
        mode = 'normal'
        
    raw_ids_str = request.form.get('patient_ids') or ''
    patient_ids = []
    for x in raw_ids_str.split(','):
        if x.strip().isdigit():
            patient_ids.append(int(x.strip()))

    user_query = (request.form.get('user_query') or '').strip()
    referenced_history_ids = (request.form.get('referenced_history_ids') or '').strip()

    image_filename = None
    img_file = request.files.get('image')
    if img_file and img_file.filename:
        allowed = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
        ext = img_file.filename.rsplit('.', 1)[-1].lower() if '.' in img_file.filename else ''
        if ext in allowed:
            raw = img_file.read()
            if len(raw) <= 8 * 1024 * 1024:
                import uuid
                image_filename = f"{session['user_id']}_{uuid.uuid4().hex[:12]}.{ext}"
                with open(os.path.join(chat_lab_dir, image_filename), 'wb') as out:
                    out.write(raw)

    from chatbot.service import extract_structured_with_ai
    
    structured = extract_structured_with_ai(doctor_final)
    if not structured:
        from chatbot.response_parser import parse_medical_report_text
        structured = parse_medical_report_text(doctor_final)

    import json

    conn = get_db()
    cursor = conn.cursor()
    names = []
    patient_rows = []
    for pid in patient_ids:
        cursor.execute(
            "SELECT full_name, patient_id, age, gender FROM patient_info WHERE id = ?",
            (pid,),
        )
        r = cursor.fetchone()
        if r:
            names.append(r['full_name'])
            patient_rows.append((pid, r))

    if patient_rows:
        primary = patient_rows[0][1]
        structured["patient_id"] = primary["patient_id"]
        structured["patient_age"] = primary["age"]
        structured["patient_gender"] = primary["gender"]
    structured_json = json.dumps(structured, ensure_ascii=False)

    patient_names = " + ".join(names)
    cursor.execute(
        """INSERT INTO medical_chat_report
           (doctor_id, mode, patient_ids_json, patient_names, ai_raw, doctor_final, structured_json,
            user_query, referenced_history_ids, image_path)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session['user_id'],
            mode,
            json.dumps(patient_ids),
            patient_names,
            ai_raw,
            doctor_final,
            structured_json,
            user_query,
            referenced_history_ids,
            image_filename,
        ),
    )
    new_id = cursor.lastrowid

    result_line = (structured.get('primary_diagnosis') or 'AI assistant consultation').strip()
    if len(result_line) > 160:
        result_line = result_line[:157] + '...'
    doctor_notes = "Source: Medical assistant generated report.\n"
    if image_filename:
        doctor_notes += f"Attachment: /api/uploads/chat-lab/{image_filename}"
    if referenced_history_ids:
        doctor_notes += f"\nContext record IDs: {referenced_history_ids}"

    for pid, r in patient_rows:
        cursor.execute(
            """INSERT INTO diagnosis_history
               (patient_db_id, patient_name, patient_pid, diagnosis_type, result, confidence,
                doctor_notes, performed_by, chat_user_query, chat_ai_response, chat_image_path,
                referenced_history_ids, medical_report_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pid, r['full_name'], r['patient_id'], 'ai_chat', result_line, 0.0,
                doctor_notes, session['user_id'], user_query, doctor_final, image_filename,
                referenced_history_ids, new_id,
            ),
        )

    conn.commit()
    conn.close()
    return jsonify({'message': 'Saved.', 'id': new_id, 'structured': structured}), 201


if __name__ == '__main__':
    init_db()
    # Optimization: Use waitress for production-grade performance on Windows if available
    try:
        from waitress import serve
        print("Starting SehatAi server in optimized mode (Waitress)...")
        print("Listening on http://0.0.0.0:5000")
        serve(app, host='0.0.0.0', port=5000, threads=8)
    except ImportError:
        print("Waitress not found. Falling back to built-in Flask server...")
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
