# Invoice AI — FastAPI + Streamlit starter

A clean starter for a shareable Arabic invoice demo. It connects a Streamlit frontend to a FastAPI backend and uses:
- YOLOv8 `best.pt` for invoice layout detection
- A locally saved Hugging Face TrOCR model directory for OCR
- JSON output for review and download

> This is a starter scaffold, not a deployed service. You must add your trained model artifacts and deploy both services to receive public URLs.

## 1) Prepare model files

Expected backend structure:

```text
backend/
  main.py
  requirements.txt
  weights/
    best.pt
    trocr/
      config.json
      model.safetensors (or pytorch_model.bin)
      preprocessor_config.json
      tokenizer files...
```

In the Colab notebook after training TrOCR, save the model and processor:

```python
trainer.save_model("/content/arabic_trocr_model")
processor.save_pretrained("/content/arabic_trocr_model")
```

Download `/content/arabic_trocr_model` and the YOLO `best.pt`, then place them in the paths above. Do not upload real customer invoices or private data to a public repository.

## 2) Run locally

Terminal 1:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd frontend
pip install -r requirements.txt
API_URL=http://127.0.0.1:8000 streamlit run app.py
```

API docs: http://127.0.0.1:8000/docs

If model files are absent, the API still starts but `/health` reports `models_missing` and `/predict` returns 503.

## 3) Deploy the backend to Render

1. Push this project to a Git repository.
2. Create a Render Web Service from that repository.
3. Set **Root Directory** to `backend`.
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add model files to the service securely or download them at startup from a private model repository using a secret token.
7. Check `https://YOUR-SERVICE.onrender.com/health` and `/docs`.

Render free services can sleep after inactivity; model loading and inference may be slow. For heavy weights or GPU inference, select suitable paid/GPU hosting after testing memory and latency.

## 4) Deploy the Streamlit frontend to Hugging Face Spaces

1. Create a new Space and choose the **Docker** SDK (the built-in Streamlit SDK is deprecated for new Spaces).
2. Upload the contents of `frontend/` to that Space.
3. In Space Settings → Variables and secrets, add `API_URL` with the deployed backend URL, e.g. `https://YOUR-SERVICE.onrender.com`.
4. Restart/rebuild the Space and test a sample image.

The public Space URL can be shared with your reviewer. Keep the API URL configured as a secret/variable; never put access tokens in code.

## 5) Important before demo day

- Test with a separate validation set; the training snippet currently uses the same YOLO image directory for train and val.
- The OCR training notebook starts from `microsoft/trocr-base-stage1`; evaluate Arabic character/word accuracy on held-out invoices.
- This API currently uses detected YOLO crops as TrOCR input. It does not implement table-cell reconstruction, arithmetic validation, or Excel/ERP export yet.
- If "TOCR" means a separate model beyond TrOCR, add its model name/files and inference code before claiming it is integrated.
- Avoid processing confidential invoices in a public demo without access controls and an approved data-handling plan.
