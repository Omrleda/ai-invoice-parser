
%%writefile /content/ai-invoice-parser/app.py
import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="INVOICE AI", page_icon="🧾", layout="wide")
st.title("🧾 INVOICE AI")
st.caption("Invoice layout detection and text recognition")

API_URL = st.text_input(
    "Backend API URL",
    value="https://sri-base-screensaver-bull.trycloudflare.com"
).strip().rstrip("/") + "/predict"

uploaded = st.file_uploader(
    "Upload invoice", type=["png", "jpg", "jpeg"]
)

if uploaded:
    data = uploaded.getvalue()
    image = Image.open(BytesIO(data)).convert("RGB")
    st.image(image, caption="Uploaded invoice", use_container_width=True)

    if st.button("Analyze Invoice", type="primary"):
        try:
            with st.spinner("Analyzing invoice..."):
                response = requests.post(
                    API_URL,
                    files={"file": (
                        uploaded.name, data,
                        uploaded.type or "image/jpeg"
                    )},
                    timeout=300
                )

            if response.ok:
                result = response.json()
                st.success("Analysis completed")
                st.json(result)

                def find_text(obj):
                    texts = []
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if k.lower() in ("text", "recognized_text", "ocr_text") and isinstance(v, str):
                                texts.append(v)
                            else:
                                texts.extend(find_text(v))
                    elif isinstance(obj, list):
                        for v in obj:
                            texts.extend(find_text(v))
                    return texts

                lines = find_text(result)
                if lines:
                    extracted = "\n".join(lines)
                    st.text_area("Recognized text", extracted, height=250)
                    st.download_button(
                        "Download text", extracted,
                        file_name="invoice_text.txt"
                    )
            else:
                st.error(f"Backend returned HTTP {response.status_code}")
                st.code(response.text[:1500])

        except Exception as e:
            st.error(f"Could not reach backend: {e}")
            st.info("Check that the backend and its Cloudflare tunnel are running.")
