
import streamlit as st
import joblib
import string
import pytesseract
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import base64
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# OCR setup (keep existing behavior)
# -----------------------------
# NOTE: keep the same hard-coded Windows path to avoid changing your working setup.
# If you want it configurable, we expose it in the sidebar while keeping default.
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


# -----------------------------
# Page config + global styles
# -----------------------------
st.set_page_config(page_title="📰 Fake News Detector Pro", layout="wide")

st.markdown(
    """
    <style>
      :root{
        --bg:#0b0f17;
        --panel:#111827;
        --panel2:#0f172a;
        --card: rgba(255,255,255,.06);
        --card2: rgba(255,255,255,.08);
        --text:#e5e7eb;
        --muted:#9ca3af;
        --border: rgba(255,255,255,.10);
        --good:#22c55e;
        --bad:#ef4444;
        --warn:#f59e0b;
        --brand:#7c3aed;
        --brand2:#06b6d4;
        --shadow: 0 12px 30px rgba(0,0,0,.35);
      }
      body{ background: radial-gradient(1200px 600px at 10% 0%, rgba(124,58,237,.20), transparent 55%),
                       radial-gradient(1000px 500px at 90% 10%, rgba(6,182,212,.15), transparent 50%),
                       var(--bg); color: var(--text); }
      .stApp{ background: transparent; }

      /* Card */
      .bb-card{ background: linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.04));
                 border: 1px solid var(--border);
                 border-radius: 16px;
                 box-shadow: var(--shadow);
                 padding: 18px; }

      /* Sidebar */
      section[data-testid="stSidebar"]{ background: rgba(17,24,39,.65); }

      /* Buttons */
      div.stButton > button{
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,.16) !important;
        background: linear-gradient(90deg, rgba(124,58,237,.95), rgba(6,182,212,.85)) !important;
        color: #fff !important;
        font-weight: 700 !important;
      }

      /* Text inputs */
      .stTextArea textarea, .stTextInput input{
        background-color: rgba(15,23,42,.75) !important;
        border: 1px solid rgba(255,255,255,.12) !important;
        color: #fff !important;
      }

      /* Select */
      .stSelectbox > div{ background: rgba(15,23,42,.75) !important; border: 1px solid rgba(255,255,255,.12) !important; }

      /* Progress bar */
      .stProgress > div > div{ background: linear-gradient(90deg, rgba(124,58,237,1), rgba(6,182,212,1)) !important; }

      footer{visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Constants
# -----------------------------
MODEL_FILENAME = "fake_news_model.pkl"
VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"

SUSPICIOUS_KEYWORDS = [
    "shocking",
    "unbelievable",
    "you won't believe",
    "click here",
    "must see",
    "urgent",
    "act now",
    "breaking",
    "rumor",
    "exclusive",
    "what they don't want you to know",
    "miracle",
    "scam",
]

REAL_FAKE_SAMPLES = {
    "real": "The government announced a new infrastructure plan today aimed at improving public transportation and reducing commute times across major cities.",
    "fake": "BREAKING: You won't believe what happened overnight—this miracle cure is guaranteed to make you immune in 24 hours. Act now! Click here.",
}


# -----------------------------
# Utilities: caching + paths
# -----------------------------

import os


def _project_path(filename: str) -> str:
    # Resolve relative to this file's directory to keep paths correct
    # regardless of where Streamlit is launched from.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, filename)


@st.cache_resource(show_spinner=False)
def load_model_resources():
    model = joblib.load(_project_path(MODEL_FILENAME))
    vectorizer = joblib.load(_project_path(VECTORIZER_FILENAME))
    return model, vectorizer


# -----------------------------
# Text cleaning + OCR
# -----------------------------

def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def extract_text_from_image(image: Image.Image) -> str:
    # Validate Tesseract availability before running OCR.
    # Important: pytesseract may have been configured with a Windows-only path that
    # doesn't exist in the current runtime. We validate availability explicitly.
    try:
        configured = getattr(pytesseract.pytesseract, "tesseract_cmd", "")
        if isinstance(configured, str) and configured.strip():
            # Only validate existence when it looks like a filesystem path.
            if ("\\" in configured or "/" in configured) and not os.path.exists(configured):
                raise RuntimeError(f"Configured Tesseract path does not exist: {configured}")

        _ = pytesseract.get_tesseract_version()
    except Exception as e:
        raise RuntimeError(
            "Tesseract OCR is not available. "
            "Install Tesseract and ensure it is on PATH, or set a valid Tesseract path in the sidebar. "
            f"Details: {type(e).__name__}: {e}"
        )


    img = np.array(image)

    # OpenCV expects BGR; PIL gives RGB.

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray)


def enhance_for_ocr(pil_img: Image.Image) -> Image.Image:
    """Best-effort OCR enhancement without breaking OCR support."""
    img = np.array(pil_img)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize up (helps small text)
    scale = 1.5
    gray = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)))

    # Contrast normalization
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Denoise a bit
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Otsu threshold for sharper text
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert back to PIL-friendly RGB
    th_rgb = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(th_rgb)


# -----------------------------
# Prediction (keep existing logic behavior)
# -----------------------------

def predict(text: str):
    model, vectorizer = load_model_resources()
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec).max()
    return pred, round(float(proba) * 100, 2)


# -----------------------------
# Credibility + keyword detection
# -----------------------------

def detect_suspicious_keywords(text: str):
    lowered = (text or "").lower()
    return [kw for kw in SUSPICIOUS_KEYWORDS if kw in lowered]


def compute_credibility(pred_label, confidence_percent: float) -> int:
    """Map model output to 0..100 credibility.

    If pred_label indicates REAL, credibility increases with confidence.
    If pred_label indicates FAKE, credibility decreases with confidence.

    This function is robust to different label encodings by treating
    strings containing 'fake' as fake.
    """
    conf = float(confidence_percent)

    label_str = str(pred_label).strip().lower()
    is_fake = "fake" in label_str
    is_real = "real" in label_str

    # Fallback heuristic: if it's not explicit, treat '0'/'1' as common encodings.
    if not is_fake and not is_real:
        # If label is numeric: assume 1 => fake, 0 => real (common in many datasets)
        # But we also handle the opposite by making it monotonic either way.
        try:
            num = int(float(label_str))
            is_fake = num == 1
            is_real = num == 0
        except Exception:
            is_fake = False
            is_real = True

    # Swap interpretation so REAL becomes FAKE (and vice-versa) for credibility scoring.
    if is_real and not is_fake:
        score = 100 - conf
    else:
        score = conf


    score = max(0, min(100, int(round(score))))
    return score


# -----------------------------
# Report generation
# -----------------------------

def build_report_text(record: dict) -> str:
    lines = []
    lines.append("FAKE NEWS DETECTOR PRO - RESULT REPORT")
    lines.append(f"Generated: {record.get('timestamp', '')}")
    lines.append("".strip())

    lines.append("[PREDICTION]")
    lines.append(f"Label: {record.get('pred_label', '')}")
    lines.append(f"Confidence: {record.get('confidence', '')}%")
    lines.append(f"Credibility (0-100): {record.get('credibility', '')}")

    lines.append("".strip())
    lines.append("[SUSPICIOUS KEYWORDS]")
    kws = record.get("suspicious_keywords", [])
    if kws:
        lines.append("Detected:")
        for kw in kws:
            lines.append(f"- {kw}")
    else:
        lines.append("None detected.")

    lines.append("".strip())
    lines.append("[RELATED NEWS MATCHES]")
    matches = record.get("related_matches", []) or []
    if matches:
        for m in matches:
            lines.append(f"- {m.get('headline','')} (similarity: {m.get('similarity','')}%)")
    else:
        lines.append("No matches available.")

    lines.append("".strip())
    lines.append("[INPUT]")
    text = record.get("input_text", "")
    lines.append(text)

    lines.append("".strip())
    lines.append("End of report.")
    return "\n".join(lines)


def _download_link_bytes(data: bytes, filename: str, mime: str):
    b64 = base64.b64encode(data).decode("utf-8")
    return (
        f"<a download=\"{filename}\" href=\"data:{mime};base64,{b64}\">"
        f"Download {filename}"
        f"</a>"
    )


def render_downloads(record: dict):
    report_text = build_report_text(record)
    txt_bytes = report_text.encode("utf-8")

    st.markdown(
        _download_link_bytes(txt_bytes, f"fake_news_report_{record.get('timestamp','').replace(':','-')}.txt", "text/plain"),
        unsafe_allow_html=True,
    )

    # PDF: keep dependency-free safe mode.
    st.info("PDF export is not enabled in this build (keeps requirements unchanged). TXT download is available.")


# -----------------------------
# Related / verification (offline placeholder)
# -----------------------------

def load_local_headlines():
    """Offline headline/content provider for *related-story similarity*.

    IMPORTANT:
    - This is NOT fact-checking.
    - It must use a consistent text field from `data/news.csv`.
    - Similarity results depend heavily on using the same preprocessing as the model.
    """
    fallback = [
        "Government unveils new public transportation initiative",
        "Scientists warn about spread of misinformation online",
        "Health officials debunk miracle cure claims",
        "Leaders discuss infrastructure funding and long-term plans",
        "Viral post alleges shocking event with no credible sources",
        "Breaking news headline turns out to be a hoax",
    ]

    try:
        df = pd.read_csv(_project_path(os.path.join("data", "news.csv")))


        # Prefer the same-ish field used in training.
        # Your training script uses `df['text']`, so we prioritize it strictly.
        preferred_cols = ["text", "content", "headline", "title", "news"]
        for col in preferred_cols:
            if col in df.columns:
                items = df[col].dropna().astype(str).tolist()
                items = [x for x in items if x and len(x.strip()) > 0]
                if items:
                    return items[:500]

        # Fallback: pick a text-like column with highest avg length.
        best_col = None
        best_avg_len = -1.0
        for c in df.columns:
            ser = df[c].dropna().astype(str)
            if len(ser) == 0:
                continue

            stripped = ser.str.strip()
            non_empty_ratio = float((stripped != "").mean())
            if non_empty_ratio < 0.5:
                continue

            avg_len = float(ser.str.len().mean())
            if avg_len > best_avg_len:
                best_avg_len = avg_len
                best_col = c

        if best_col is not None:
            items = df[best_col].dropna().astype(str).tolist()
            items = [x for x in items if x and len(x.strip()) > 0]
            if items:
                return items[:500]

    except Exception:
        pass

    return fallback


@st.cache_resource(show_spinner=False)
def get_headlines_cached():
    return load_local_headlines()


def similar_news_matches(query: str, headlines: list[str], top_k: int = 5):
    """Offline related-story similarity.

    Returns:
      [{ "headline": str, "similarity": float, "note": str }]
    """
    if not query or not query.strip():
        return []

    # IMPORTANT: use same cleaning as the model so similarity is meaningful.
    q_clean = clean_text(query)
    cleaned_headlines = []
    for h in headlines:
        if h is None:
            continue
        h_str = str(h)
        if not h_str.strip():
            continue
        cleaned_headlines.append(h_str)

    if not cleaned_headlines:
        return []

    vect = TfidfVectorizer(stop_words="english", max_features=6000)
    corpus = [q_clean] + [clean_text(h) for h in cleaned_headlines]
    mat = vect.fit_transform(corpus)

    sims = cosine_similarity(mat[0], mat[1:])[0]
    top_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_idx:
        original = cleaned_headlines[idx]
        sim = float(sims[idx])
        sim_pct = max(0.0, min(100.0, sim * 100))
        results.append({
            "headline": original,
            "similarity": round(sim_pct, 2),
            "note": "Offline related-story similarity (NOT a fact-check)."
        })

    return results


def rag_verify(query: str):
    """Offline related-story verification (similarity-only).

    Without API keys, we keep this as a local similarity cross-check against
    `data/news.csv` (or fallback headlines).
    """
    headlines = get_headlines_cached()
    matches = similar_news_matches(query, headlines, top_k=6)
    return {
        "matches": matches,
        "source": "Offline: local data/news.csv (related-story similarity only)"
    }


# -----------------------------
# Session state
# -----------------------------

def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "ocr_tesseract_cmd" not in st.session_state:
        st.session_state.ocr_tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
    if "enhance_ocr" not in st.session_state:
        st.session_state.enhance_ocr = True


# -----------------------------
# UI helpers
# -----------------------------

def result_card(pred_label, confidence, credibility, suspicious_keywords):
    pred_str = str(pred_label).upper()

    # Interpret label for styling
    label_lower = str(pred_label).lower()
    is_fake = "fake" in label_lower
    is_real = "real" in label_lower

    if not is_fake and not is_real:
        # numeric fallback
        try:
            is_fake = int(float(label_lower)) == 1
            is_real = not is_fake
        except Exception:
            is_real = True

    # Swap interpretation as requested: model's REAL -> show FAKE, model's FAKE -> show REAL
    status = "FAKE" if is_real else "REAL"
    accent = "var(--bad)" if status == "FAKE" else "var(--good)"
    icon = "🚨" if status == "FAKE" else "✅"


    st.markdown(
        f"""
        <div class='bb-card'>
          <div style='display:flex; align-items:center; justify-content:space-between; gap:14px;'>
            <div>
              <div style='font-size:14px; color: var(--muted); font-weight:600;'>DETECTION</div>
              <div style='font-size:34px; font-weight:900; color:{accent};'>{icon} {status}</div>
              <div style='margin-top:8px; color: var(--muted); font-weight:600;'>Model confidence: <span style='color: #fff;'>{confidence:.2f}%</span></div>
            </div>
            <div style='text-align:right;'>
              <div style='font-size:14px; color: var(--muted); font-weight:600;'>CREDIBILITY SCORE</div>
              <div style='font-size:26px; font-weight:900;'>{credibility}/100</div>
              <div style='color: var(--muted); font-weight:600;'>0 = suspicious · 100 = credible</div>
            </div>
          </div>
          <div style='margin-top:16px;'>
            <div style='color: var(--muted); font-weight:700; font-size:13px; margin-bottom:6px;'>CONFIDENCE</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(1.0, max(0.0, confidence / 100.0)))

    if suspicious_keywords:
        st.warning("⚠️ Suspicious keywords detected: " + ", ".join(suspicious_keywords))
    else:
        st.success("No suspicious keywords detected by heuristic checks.")


def show_history():
    history = st.session_state.history
    st.subheader("🕘 Prediction History")
    if not history:
        st.caption("No predictions yet. Run a check to populate history.")
        return

    cols = st.columns([2, 1, 1])
    cols[0].markdown("**Input (preview)**")
    cols[1].markdown("**Result**")
    cols[2].markdown("**Credibility**")

    for rec in reversed(history[-30:]):
        preview = (rec.get("input_text", "") or "").strip().replace("\n", " ")
        preview = (preview[:60] + "…") if len(preview) > 60 else preview
        label = str(rec.get("pred_label", "")).upper()
        cred = rec.get("credibility", "")

        c1, c2, c3 = st.columns([2, 1, 1])
        c1.write(preview)
        c2.write(label)
        c3.write(f"{cred}/100")

        # Optional: expand for actions
        with st.expander("Open details", expanded=False):
            st.caption(f"Timestamp: {rec.get('timestamp', '')}")
            st.write(f"Confidence: {rec.get('confidence', '')}%")
            if rec.get("suspicious_keywords"):
                st.write("Suspicious keywords: " + ", ".join(rec["suspicious_keywords"]))
            if rec.get("related_matches"):
                st.write("Related news matches:")
                for m in rec["related_matches"]:
                    st.write(f"- {m['headline']} ({m['similarity']}%)")
            render_downloads(rec)


def render_sample_text_buttons(tab_key: str = "main", _render_once: bool = False):

    st.subheader("✨ Try sample text")
    c1, c2 = st.columns(2)

    # This function is rendered in multiple tabs.
    # Ensure button keys are unique by scoping them with tab_key.
    with c1:
        if st.button(
            "Use REAL sample",
            use_container_width=True,
            type="primary",
            key=f"sample_real_btn_{tab_key}",
        ):
            st.session_state.sample_text = REAL_FAKE_SAMPLES["real"]
            st.write(f"DEBUG: SAMPLE REAL CLICKED (tab={tab_key})")

    with c2:
        if st.button(
            "Use FAKE sample",
            use_container_width=True,
            key=f"sample_fake_btn_{tab_key}",
        ):
            st.session_state.sample_text = REAL_FAKE_SAMPLES["fake"]



# -----------------------------
# Main app
# -----------------------------
init_session_state()

# Force tesseract cmd from sidebar config
try:
    pytesseract.pytesseract.tesseract_cmd = st.session_state.ocr_tesseract_cmd
except Exception:
    pass

with st.sidebar:
    st.markdown("## Fake News Detector Pro")
    st.caption("Production-style mentor-ready UI")

    st.markdown("---")
    st.markdown("### Tech Stack")
    st.write("- Streamlit")
    st.write("- scikit-learn (TF-IDF + classifier)")
    st.write("- pytesseract + OpenCV (OCR)")

    st.markdown("---")
    st.markdown("###  Usage")
    st.write("1) Paste text, upload a TXT, or upload a screenshot")
    st.write("2) Click **Check News**")
    st.write("3) Review results + credibility + related matches")

    st.markdown("---")
    st.markdown("### OCR Settings")
    cmd_default = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    default_cmd = st.session_state.get("ocr_tesseract_cmd") or cmd_default
    st.text_input("Tesseract path", value=default_cmd, key="ocr_tesseract_cmd")

    st.checkbox("Enhance image before OCR (recommended)", value=st.session_state.enhance_ocr, key="enhance_ocr")

    st.markdown("---")
    st.caption("Tip: Keep images clear; higher resolution improves OCR.")


# Hero
st.markdown(
    """
    <div class='bb-card' style='padding:22px; margin-bottom:16px;'>
      <div style='display:flex; align-items:center; justify-content:space-between; gap:18px;'>
        <div>
          <div style='font-size:14px; color: var(--muted); font-weight:700;'>FAKE NEWS • OCR • VERIFICATION</div>
          <div style='font-size:34px; font-weight:1000; line-height:1.1;'>📰 Fake News Detector Pro</div>
          <div style='margin-top:8px; color: var(--muted); font-weight:600;'>Mentor-ready, production-grade UI with confidence, credibility, history & related matches.</div>
        </div>
        <div style='text-align:right;'>
          <div style='display:inline-block; padding:10px 14px; border-radius:14px; border:1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.06);'>
            <div style='font-size:12px; color: var(--muted); font-weight:700;'>MODEL</div>
            <div style='font-size:18px; font-weight:900;'>TF-IDF Classifier</div>
            <div style='font-size:12px; color: var(--muted); font-weight:700;'>LOCAL OFFLINE</div>
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Tabs
text_tab, img_tab, txt_tab, verify_tab = st.tabs(["Text Input", "Screenshot Upload", "TXT Upload", "News Verification"])


def run_prediction_pipeline(input_text: str):
    input_text = (input_text or "").strip()
    if not input_text:
        st.error("No text found. Please paste text, upload a TXT file, or upload a screenshot.")
        return None

    if len(input_text) < 30:
        st.warning("Text looks very short. Prediction may be less reliable. Add more context for best results.")

    pred_label, confidence = predict(input_text)
    credibility = compute_credibility(pred_label, confidence)
    suspicious_keywords = detect_suspicious_keywords(input_text)

    # Verification / related matches
    verify = rag_verify(input_text)
    related_matches = verify.get("matches", [])

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_text": input_text,
        "pred_label": pred_label,
        "confidence": confidence,
        "credibility": credibility,
        "suspicious_keywords": suspicious_keywords,
        "related_matches": related_matches,
        "verification_source": verify.get("source", ""),
    }

    st.session_state.history.append(record)
    return record


with text_tab:
    render_sample_text_buttons(tab_key="text")


    user_text = st.text_area(
        "✍️ Paste your news text here",
        height=300,
        value=st.session_state.get("sample_text", ""),
        key="text_input_area",
    )

    # Keep placeholder for the fancy "Palantir-like" interaction feel
    cA, cB = st.columns([1, 1])
    with cA:
        st.caption("Tip: The model uses TF-IDF signals; removing punctuation is handled internally.")
    with cB:
        st.caption("Detection is offline and fast. No external requests.")

    if st.button("🔍 Check News", type="primary", use_container_width=True):
        record = run_prediction_pipeline(user_text)
        if record:
            st.markdown("## 🧾 Detection Result")
            result_card(
                pred_label=record.get("pred_label"),
                confidence=record.get("confidence", 0.0),
                credibility=record.get("credibility", 0),
                suspicious_keywords=record.get("suspicious_keywords", []),
            )

            with st.expander("🧠 Explainability (heuristic cues)", expanded=False):
                st.write({"suspicious_keywords_triggered": record.get("suspicious_keywords", [])})

            st.markdown("---")
            st.subheader("🧬 Related News Match")
            if record.get("related_matches"):
                for m in record["related_matches"]:
                    st.caption(f"Similarity: {m['similarity']}%")
                    st.code(m["headline"])
                    st.markdown("---")
            else:
                st.caption("No related matches available (offline placeholder).")

            st.markdown("---")
            with st.expander("📥 Download report", expanded=False):
                render_downloads(record)

            st.toast("Saved to session history ✅")

# -----------------------------
# Screenshot upload tab
# -----------------------------
with img_tab:
    render_sample_text_buttons(tab_key="image")


    uploaded_img = st.file_uploader(
        "📸 Upload a screenshot (PNG/JPG/JPEG)",
        type=["png", "jpg", "jpeg"],
    )

    enhance = st.checkbox("🔬 Enhance image before OCR", value=st.session_state.enhance_ocr)

    if uploaded_img:
        img = Image.open(uploaded_img)
        st.image(img, caption="Uploaded screenshot", use_column_width=True)

        if st.button("🔍 Extract text + Check News", type="primary", use_container_width=True):
            try:
                # OCR pipeline
                ocr_img = enhance_for_ocr(img) if enhance else img
                try:
                    extracted = extract_text_from_image(ocr_img)
                except Exception as e:
                    st.error(str(e))
                    return

                extracted_clean = (extracted or "").strip()

                if not extracted_clean or len(extracted_clean) < 10:
                    st.error("OCR did not extract enough text. Try a clearer image or enable enhancement.")
                else:
                    st.markdown("### 📝 Extracted Text (preview)")
                    st.code(extracted_clean[:1500] + ("…" if len(extracted_clean) > 1500 else ""))

                    record = run_prediction_pipeline(extracted_clean)
                    if record:
                        st.markdown("## 🧾 Detection Result")
                        result_card(
                            pred_label=record.get("pred_label"),
                            confidence=record.get("confidence", 0.0),
                            credibility=record.get("credibility", 0),
                            suspicious_keywords=record.get("suspicious_keywords", []),
                        )

                        st.markdown("---")
                        st.subheader("🧬 Related News Match")
                        if record.get("related_matches"):
                            for m in record["related_matches"]:
                                st.caption(f"Similarity: {m['similarity']}%")
                                st.code(m["headline"])
                                st.markdown("---")
                        else:
                            st.caption("No related matches available (offline placeholder).")

                        st.markdown("---")
                        with st.expander("📥 Download report", expanded=False):
                            render_downloads(record)

            except Exception as e:
                st.error(f"OCR/processing failed: {e}")

    else:
        st.info("Upload a screenshot to extract text and run detection.")

# -----------------------------
# TXT upload tab
# -----------------------------
with txt_tab:
    render_sample_text_buttons(tab_key="txt")


    uploaded_txt = st.file_uploader("📄 Upload a .txt file", type=["txt"])

    if uploaded_txt is not None:
        if st.button("🔍 Check News from TXT", type="primary", use_container_width=True):
            try:
                raw = uploaded_txt.read()
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw.decode("latin-1")

                text = (text or "").strip()
                if not text or len(text) < 10:
                    st.error("TXT upload contained too little text for prediction.")
                else:
                    st.markdown("### 📝 TXT content (preview)")
                    st.code(text[:2000] + ("…" if len(text) > 2000 else ""))

                    record = run_prediction_pipeline(text)
                    if record:
                        st.markdown("## 🧾 Detection Result")
                        result_card(
                            pred_label=record.get("pred_label"),
                            confidence=record.get("confidence", 0.0),
                            credibility=record.get("credibility", 0),
                            suspicious_keywords=record.get("suspicious_keywords", []),
                        )

                        st.markdown("---")
                        st.subheader("🧬 Related News Match")
                        if record.get("related_matches"):
                            for m in record["related_matches"]:
                                st.caption(f"Similarity: {m['similarity']}%")
                                st.code(m["headline"])
                                st.markdown("---")
                        else:
                            st.caption("No related matches available (offline placeholder).")

                        st.markdown("---")
                        with st.expander("📥 Download report", expanded=False):
                            render_downloads(record)

            except Exception as e:
                st.error(f"TXT processing failed: {e}")
    else:
        st.info("Upload a .txt file to run detection.")

# -----------------------------
# News verification tab (RSS-based real internet checking)
# -----------------------------
with verify_tab:
    st.subheader(" News Verification (Offline Related-News Similarity)")
    st.caption("This section shows *related story* similarity only (NOT fact-checking). External APIs/keys are not required.")

    q = st.text_area(
        "Enter the news text / headline to cross-check",
        height=200,
        placeholder="Example: Scientists warn about spread of misinformation online",
    )

    if st.button(" Find Related News Matches", type="primary", use_container_width=True):
        q2 = (q or "").strip()
        if not q2:
            st.error("Enter some text to verify.")
        else:
            try:
                verify = rag_verify(q2)
                st.markdown("### 🧬 Related News Match")
                st.caption(verify.get("source", ""))

                matches = verify.get("matches", []) or []
                if not matches:
                    st.caption("No matches found.")
                else:
                    for m in matches:
                        st.caption(f"Similarity: {m['similarity']}%")
                        st.code(m.get('headline', ''))
                        if m.get("note"):
                            st.caption(m["note"])
                        st.markdown("---")
            except Exception as e:
                st.error(f"Verification failed: {e}")

# -----------------------------
# Footer + History + session tools
# -----------------------------
st.markdown("---")
show_history()

st.markdown(
    """
    <div style='text-align:center; padding:18px; color: rgba(229,231,235,.65);'>
      <div style='font-weight:800;'>Fake News Detector Pro</div>
      <div style='font-size:12px;'>Mentor-ready UI • Offline TF-IDF detection • OCR + TXT support</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Clear history button
if st.button("🧹 Clear Prediction History", use_container_width=True):
    st.session_state.history = []
    st.toast("History cleared ✅")

