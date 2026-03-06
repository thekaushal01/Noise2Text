import cv2
import easyocr
from PIL import Image

from preprocessing import preprocess_image

# Load once — EasyOCR model initialisation is expensive
reader = easyocr.Reader(['en'], gpu=False)


def recognize_text(image: Image.Image):
    if image is None:
        return None, "", "N/A"

    processed = preprocess_image(image)

    # EasyOCR works best with a 3-channel image even if the content is grayscale
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

    results = reader.readtext(processed_rgb)

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
