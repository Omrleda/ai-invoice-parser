import os
import requests
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="Invoice AI | Document Intelligence", page_icon="🧾", layout="wide")

API_URL = os.getenv("API_URL", "").strip().rstrip("/")
if not API_URL:
    try:
        API_URL = st.secrets.get("API_URL", "").strip().rstrip("/")
    except Exception:
        API_URL = ""

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f7fb; color: #172033; }
.block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 3rem; }
.hero { background: linear-gradient(120deg,#101a33 0%,#1c3159 65%,#295b75 100%); padding: 2rem 2.2rem; border-radius: 24px; color: white; margin-bottom: 1.4rem; }
.hero h1 { font-family: Manrope,sans-serif; font-size: 2.4rem; font-weight: 800; margin: 0; color: white; letter-spacing:-1px; }
.hero p { color:#d9e6f7; font-size:1rem; margin:.55rem 0 0 0; }
.pill { display:inline-block; border:1px solid #7b9fc7; color:#e5f0ff; border-radius:999px; padding:5px 11px; font-size:.75rem; margin-bottom:1rem; }
.metric { background:white; border:1px solid #e3e8f1; border-radius:16px; padding:1rem 1.1rem; }
.metric-label { color:#65728a; font-size:.8rem; }
.metric-value { font-size:1.25rem; font-weight:700; color:#15213a; margin-top:.2rem; }
.section-title { font-family:Manrope,sans-serif; font-size:1.15rem; font-weight:800; color:#18243c; margin:.3rem 0 .7rem; }
div[data-testid="stFileUploader"] { background:white; border:1px dashed #b7c5d9; border-radius:16px; padding:1rem; }
.stButton button, .stDownloadButton button { border-radius:10px; font-weight:700; min-height:2.7rem; }
small, .muted { color:#718096; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="pill">DOCUMENT INTELLIGENCE · PROTOTYPE</div>
  <h1>Invoice AI</h1>
  <p>Turn Arabic invoices into structured, reviewable business data.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="metric"><div class="metric-label">LAYOUT DETECTION</div><div class="metric-value">YOLOv8</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric"><div class="metric-label">TEXT RECOGNITION</div><div class="metric-value">TrOCR</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric"><div class="metric-label">OUTPUT</div><div class="metric-value">Review-ready JSON</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([0.92, 1.08], gap="large")
with left:
    st.markdown('<div class="section-title">01 / Upload invoice</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose a JPG or PNG invoice image", type=["jpg", "jpeg", "png"])
    st.caption("Prototype note: PDF support can be added by converting each page to an image.")
    if uploaded:
        st.image(uploaded, caption=uploaded.name, use_container_width=True)
    run = st.button("Analyze invoice  →", type="primary", use_container_width=True, disabled=not uploaded)

with right:
    st.markdown('<div class="section-title">02 / Extraction results</div>', unsafe_allow_html=True)
    if not API_URL:
        st.warning("Backend URL is not configured yet. Set API_URL in the app environment or Hugging Face Space secrets.")
        st.code("API_URL=https://your-fastapi-service.onrender.com", language="bash")
    if run and uploaded:
        if not API_URL:
            st.error("Configure API_URL before running inference.")
        else:
            with st.spinner("Detecting invoice regions and reading text…"):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")},
                        timeout=180,
                    )
                    if response.ok:
                        result = response.json()
                        st.success(f"Analysis complete · {len(result.get('detections', []))} regions detected")
                        st.caption(f"Processed on {result.get('device', 'unknown device')}")
                        rows = result.get("detections", [])
                        if rows:
                            st.dataframe(
                                [{"Region": r["class_name"], "Confidence": r["confidence"], "Extracted text": r["text"]} for r in rows],
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("No regions detected. Check the YOLO model and confidence threshold.")
                        st.download_button(
                            "Download results as JSON",
                            data=__import__("json").dumps(result, ensure_ascii=False, indent=2),
                            file_name="invoice_ai_results.json",
                            mime="application/json",
                            use_container_width=True,
                        )
                        with st.expander("Raw API response"):
                            st.json(result)
                    else:
                        try:
                            detail = response.json().get("detail", response.text)
                        except Exception:
                            detail = response.text
                        st.error(f"API returned {response.status_code}: {detail}")
                except requests.RequestException as exc:
                    st.error(f"Could not reach the API: {exc}")
    elif not run:
        st.info("Upload an invoice and select **Analyze invoice** to see detected regions and extracted text.")

st.divider()
with st.expander("System status & limitations"):
    if API_URL:
        try:
            h = requests.get(f"{API_URL}/health", timeout=8)
            st.write("Backend health endpoint")
            st.json(h.json())
        except requests.RequestException:
            st.warning("The backend health endpoint could not be reached.")
    st.markdown("""
- Results are a prototype and must be reviewed by a person.
- Accuracy targets are not performance claims until measured on held-out invoices.
- Do not upload confidential invoices to a public demo.
- This starter integrates YOLOv8 and the trained TrOCR model; any separate OCR/TOCR model needs its own adapter.
""")
