
import os
import streamlit as st
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

st.set_page_config(
    page_title="INVOICE AI",
    page_icon="🧾",
    layout="wide"
)

st.title("🧾 INVOICE AI")
st.caption("Invoice layout detection and text recognition")

HF_REPO = "ledaaaa/invoice-ai-models"


@st.cache_resource(show_spinner="Downloading and loading AI models...")
def load_models():
    token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))

    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Add it in Streamlit app Secrets."
        )

    model_dir = snapshot_download(
        repo_id=HF_REPO,
        repo_type="model",
        token=token,
        allow_patterns=["yolo/best.pt", "trocr/*"]
    )

    yolo_path = os.path.join(model_dir, "yolo", "best.pt")
    trocr_path = os.path.join(model_dir, "trocr")

    if not os.path.exists(yolo_path):
        raise FileNotFoundError("Could not find yolo/best.pt in the model repo.")

    yolo = YOLO(yolo_path)
    processor = TrOCRProcessor.from_pretrained(trocr_path)
    trocr = VisionEncoderDecoderModel.from_pretrained(trocr_path)
    trocr.eval()

    return yolo, processor, trocr


def recognize_text(image, yolo, processor, trocr):
    detections = yolo.predict(image, conf=0.20, verbose=False)
    boxes = detections[0].boxes
    regions = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().tolist()
            )
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)

            if x2 > x1 and y2 > y1:
                regions.append((y1, x1, image.crop((x1, y1, x2, y2))))

    # If YOLO detects no regions, run OCR on the whole invoice.
    if not regions:
        regions = [(0, 0, image)]

    # Read regions approximately from top to bottom, then left to right.
    regions.sort(key=lambda item: (item[0], item[1]))

    output = []
    for _, _, crop in regions:
        crop = crop.convert("RGB")
        pixel_values = processor(
            images=crop,
            return_tensors="pt"
        ).pixel_values

        with torch.inference_mode():
            generated_ids = trocr.generate(
                pixel_values,
                max_new_tokens=128
            )

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        if text:
            output.append(text)

    return output, len(regions)


st.markdown("### Upload an invoice")
uploaded_file = st.file_uploader(
    "Choose an invoice image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.image(image, caption="Uploaded invoice", use_container_width=True)

    with right:
        st.markdown("#### Recognized text")

        if st.button("Analyze invoice", type="primary", use_container_width=True):
            try:
                with st.spinner(
                    "Loading models and analyzing the invoice. "
                    "The first run may take several minutes..."
                ):
                    yolo, processor, trocr = load_models()
                    lines, region_count = recognize_text(
                        image, yolo, processor, trocr
                    )

                if lines:
                    result = "\n".join(lines)
                    st.success("OCR finished.")
                    st.text_area(
                        "Extracted text",
                        value=result,
                        height=350
                    )
                    st.download_button(
                        "Download extracted text",
                        data=result,
                        file_name="invoice_text.txt",
                        mime="text/plain"
                    )
                    st.caption(f"Processed {region_count} image region(s).")
                else:
                    st.warning(
                        "No text was recognized. Try a clearer invoice image."
                    )

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.info(
                    "If the error mentions memory, the free hosting "
                    "environment may not have enough RAM for these models."
                )
