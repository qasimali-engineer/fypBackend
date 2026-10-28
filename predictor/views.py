import json
import re
import threading
from pathlib import Path

import torch
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path(settings.BASE_DIR) / "fyp-xlmr-model"
_model = None
_tokenizer = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_lock = threading.Lock()
MODEL_SUICIDE_THRESHOLD = 0.60
EXPLICIT_SELF_HARM_PATTERNS = (
    r"\bkill myself\b",
    r"\bhurt myself\b",
    r"\bharm myself\b",
    r"\bend my life\b",
    r"\bkhud\s*k[oо]\s*n[uو]qs?an\b",
    r"\bkhudkushi\b",
    r"خود\s*کشی",
    r"خود\s*کو\s*نقصان",
    r"اپنے\s*آپ\s*کو\s*مار",
)


def _load_model():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        with _model_lock:
            if _model is None or _tokenizer is None:
                if not MODEL_DIR.is_dir():
                    raise FileNotFoundError(
                        f"Trained model directory was not found: {MODEL_DIR}"
                    )
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
                _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
                _model.to(_device)
                _model.eval()
    return _model, _tokenizer


def _contains_explicit_self_harm_phrase(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return any(re.search(pattern, normalized) for pattern in EXPLICIT_SELF_HARM_PATTERNS)


@csrf_exempt
def classify_self_harm(request):
    if request.method != "POST":
        return JsonResponse(
            {"detail": "Only POST requests are allowed."},
            status=405,
            headers={"Allow": "POST"},
        )

    try:
        payload = request.body.decode("utf-8")
        data = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return JsonResponse({"detail": "Request body must be valid JSON."}, status=400)

    text = data.get("text") if isinstance(data, dict) else None
    if not isinstance(text, str) or not text.strip():
        return JsonResponse(
            {"detail": 'A non-empty string field named "text" is required.'},
            status=400,
        )

    try:
        model, tokenizer = _load_model()
    except FileNotFoundError as error:
        return JsonResponse({"detail": str(error)}, status=503)
    except (OSError, ValueError) as error:
        return JsonResponse({"detail": f"Unable to load trained model: {error}"}, status=503)

    inputs = tokenizer(
        text.strip(),
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(_device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]

    suicide_probability = float(probabilities[1].item())
    explicit_phrase = _contains_explicit_self_harm_phrase(text)
    is_self_harm = explicit_phrase or suicide_probability >= MODEL_SUICIDE_THRESHOLD
    prediction = 1 if is_self_harm else 0
    label = model.config.id2label.get(prediction, str(prediction))
    confidence = (
        1.0 if explicit_phrase else
        suicide_probability if is_self_harm else float(probabilities[0].item())
    )
    return JsonResponse(
        {
            "self_harm": "Yes" if is_self_harm else "No",
            "class": label,
            "confidence": round(confidence, 4),
            "suicide_probability": round(suicide_probability, 4),
            "detection_method": "explicit_phrase" if explicit_phrase else "model",
        }
    )
