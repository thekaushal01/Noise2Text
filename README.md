# Noise2Text:- Robust OCR for Distorted Text Recognition
Noise2Text is an experimental offline OCR designed to recognize text from distorted or noisy images. The project explores how layered image preprocessing techniques combined with a deep-learning OCR model can improve recognition accuracy on challenging inputs such as synthetic CAPTCHA-style text.

## Project Structure

```
Noise2Text/
├── venv/               # Virtual environment (not included in repo)
├── app.py              # Gradio UI — entry point
├── preprocessing.py    # Image preprocessing pipeline
├── ocr_engine.py       # EasyOCR wrapper
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/thekaushal01/Noise2Text.git
cd Noise2Text
```

### 2. Confirm Python version

Python 3.10 or newer is required.

```bash
python --version
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows (PowerShell)**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD)**
```cmd
venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
source venv/bin/activate
```

Your prompt should show `(venv)` once active.

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

> EasyOCR will download its language model (~100 MB) the first time `ocr_engine.py` imports it. After that the model is cached and the system works fully offline.

### 6. Run the app

```bash
python app.py
```

Open the URL printed in the terminal (default `http://127.0.0.1:7860`), upload an image, and click **Recognize Text**.

### 7. Deactivate when done

```bash
deactivate
```

## Preprocessing Pipeline

| Step | Method |
|---|---|
| Grayscale conversion | OpenCV color space conversion |
| 3x upscale | `cv2.resize` with cubic interpolation |
| Binarisation | Otsu thresholding with binary inverse |
| Morphological opening | 2x2 kernel to remove noise |
| Noise removal | Median blur |

## Tech Stack

- Python 3.10+
- OpenCV
- EasyOCR
- NumPy
- Gradio
- Pillow
- PyTorch

## Potential Future Improvements

- Multiple preprocessing presets with result voting
- Bounding box visualization overlay
- Multi-language support
- Batch image processing
- Custom CRNN OCR model
