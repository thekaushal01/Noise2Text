import warnings
import easyocr
from PIL import Image

from preprocessing import preprocess_image

#this warning shows up even with gpu=False, just noise
warnings.filterwarnings(
    "ignore",
    message=".*pin_memory.*no accelerator.*",
    category=UserWarning,
)

#load the model once at startup, not on every call
reader = easyocr.Reader(['en'], gpu=False)


def recognize_text(image: Image.Image):
    if image is None:
        return None, "", "N/A"

    processed = preprocess_image(image)

    results = reader.readtext(
        processed,
        allowlist='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    )

    texts, confidences = [], []
    for (_bbox, text, conf) in results:
        stripped = text.strip()
        if stripped:
            texts.append(stripped)
            confidences.append(conf)

    recognized_text = " ".join(texts) if texts else "(no text detected)"
    avg_confidence = (
        f"{sum(confidences) / len(confidences) * 100:.1f}%"
        if confidences
        else "N/A"
    )

    return Image.fromarray(processed), recognized_text, avg_confidence
