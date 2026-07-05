# TODO - Fake News Detector Pro

- [x] (1) Make OCR/Tesseract setup robust in `backend/app.py`
  - [x] Validate Tesseract via `pytesseract.get_tesseract_version()` before OCR
  - [x] Fallback behavior handled by `get_tesseract_version()` (PATH/cmd)
  - [x] Catch/avoid hard crash: screenshot tab now shows actionable error message
  - [ ] Update screenshot flow to skip OCR gracefully when Tesseract isn’t available


- [ ] (2) Test:
  - [ ] Run Streamlit and upload an image to confirm graceful failure
  - [ ] Run Streamlit with valid Tesseract installed to confirm OCR still works

