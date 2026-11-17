
import re
import easyocr
import cv2
import numpy as np

from src.enhancement import enhance_plate
from src.utils import ROOT
from pathlib import Path


CHAR_TO_DIGIT = {
    'O': '0', 'D': '0', 'Q': '0', 'U': '0',
    'I': '1', 'L': '1', 'T': '1',
    'Z': '2',
    'E': '3',
    'A': '4',
    'S': '5',
    'G': '6',
    'B': '8'
}
DIGIT_TO_CHAR = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '3': 'E',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '8': 'B'
}


def correct_plate_syntax(raw):
    """
    Applies ALPR character ambiguity resolution using regional plate grammar:
    1. Standard 10-character format: [State: 2 chars][RTO: 2 digits][Series: 2 chars][Number: 4 digits]
       e.g. BROIPC3938 -> BR01PC3938
    2. 9-character format: [State: 2 chars][RTO: 2 digits][Series: 1 char][Number: 4 digits]
       e.g. DL1CAB1234 -> DL01AB1234
    """
    cleaned = re.sub(r'[^A-Z0-9]', '', raw.upper())
    chars = list(cleaned)
    n = len(chars)

    if n == 10:
        # State code (chars 0, 1) -> Must be letters
        chars[0] = DIGIT_TO_CHAR.get(chars[0], chars[0])
        chars[1] = DIGIT_TO_CHAR.get(chars[1], chars[1])
        # District / RTO code (chars 2, 3) -> Must be digits
        chars[2] = CHAR_TO_DIGIT.get(chars[2], chars[2])
        chars[3] = CHAR_TO_DIGIT.get(chars[3], chars[3])
        # Series code (chars 4, 5) -> Must be letters
        chars[4] = DIGIT_TO_CHAR.get(chars[4], chars[4])
        chars[5] = DIGIT_TO_CHAR.get(chars[5], chars[5])
        # Vehicle sequence (chars 6-9) -> Must be digits
        for i in range(6, 10):
            chars[i] = CHAR_TO_DIGIT.get(chars[i], chars[i])
        return "".join(chars)

    elif n == 9:
        # State code (chars 0, 1) -> Must be letters
        chars[0] = DIGIT_TO_CHAR.get(chars[0], chars[0])
        chars[1] = DIGIT_TO_CHAR.get(chars[1], chars[1])
        # District / RTO code (chars 2, 3) -> Must be digits
        chars[2] = CHAR_TO_DIGIT.get(chars[2], chars[2])
        chars[3] = CHAR_TO_DIGIT.get(chars[3], chars[3])
        # Series code (char 4) -> Must be letter
        chars[4] = DIGIT_TO_CHAR.get(chars[4], chars[4])
        # Vehicle sequence (chars 5-8) -> Must be digits
        for i in range(5, 9):
            chars[i] = CHAR_TO_DIGIT.get(chars[i], chars[i])
        return "".join(chars)

    return cleaned


class PlateOCR:
    def __init__(self, languages=None, gpu=False):
        model_dir = ROOT / "models" / "easyocr"
        model_dir.mkdir(parents=True, exist_ok=True)
        self.reader = easyocr.Reader(
            languages or ["en"],
            gpu=gpu,
            model_storage_directory=str(model_dir),
            user_network_directory=str(model_dir)
        )

    def read_preenhanced(self, gray_image):
        """
        Run OCR on an already-enhanced grayscale image without applying
        any internal enhancement pipeline. Use this in the Deblur Lab to
        avoid double-enhancement artifacts.
        """
        if gray_image is None or gray_image.size == 0:
            return {"text": "", "confidence": 0.0, "needs_review": True}

        # Ensure grayscale
        if len(gray_image.shape) == 3:
            gray = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray_image.copy()

        h, w = gray.shape[:2]
        if h < 10 or w < 20:
            return {"text": "", "confidence": 0.0, "needs_review": True}

        # Upscale only if too small for OCR
        if h < 60:
            scale = max(2, min(4, 80 // max(1, h)))
            gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        # Also try CLAHE and threshold variants for better coverage
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        _, gray_thresh = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        candidates = []
        for img in (gray, gray_clahe, gray_thresh):
            results = self.reader.readtext(
                img,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
            if not results:
                continue
            results = sorted(results, key=lambda r: min(p[0] for p in r[0]))
            raw_text = "".join(r[1] for r in results)
            normalized = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
            if not normalized:
                continue
            confidence = float(np.mean([float(r[2]) for r in results]))
            candidates.append({"text": normalized, "confidence": confidence})

        if not candidates:
            return {"text": "", "confidence": 0.0, "needs_review": True}

        best = max(candidates, key=lambda c: c["confidence"])
        text = correct_plate_syntax(best["text"])
        confidence = best["confidence"]
        needs_review = confidence < 0.70 or len(text) < 5 or len(text) > 12
        return {"text": text, "confidence": round(confidence, 4), "needs_review": needs_review}

    def read(self, plate_image):
        if plate_image is None or plate_image.size == 0:
            return {
                "text": "",
                "confidence": 0.0,
                "needs_review": True,
            }

        # Enlarge small plate crops.
        height, width = plate_image.shape[:2]

        if height < 20 or width < 40:
            return {
                "text": "",
                "confidence": 0.0,
                "needs_review": True,
            }

        scale = max(2, min(4, 120 // height))

        enlarged = cv2.resize(
            plate_image,
            (width * scale, height * scale),
            interpolation=cv2.INTER_CUBIC,
        )

        # Apply your existing enhancement function.
        enhanced = enhance_plate(enlarged)

        # Support either grayscale or color output.
        if len(enhanced.shape) == 3:
            gray = cv2.cvtColor(
                enhanced, cv2.COLOR_BGR2GRAY
            )
        else:
            gray = enhanced

        # Improve local contrast.
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        enhanced_gray = clahe.apply(gray)

        # Try multiple image versions.
        _, thresholded = cv2.threshold(
            enhanced_gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # Try multiple image versions (raw gray, enhanced CLAHE, and thresholded).
        raw_gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY) if len(enlarged.shape) == 3 else enlarged.copy()
        candidates = []

        for image in (raw_gray, enhanced_gray, thresholded):
            results = self.reader.readtext(
                image,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )

            if not results:
                continue

            # Combine detected text fragments from left to right.
            results = sorted(
                results,
                key=lambda item: min(
                    point[0] for point in item[0]
                ),
            )

            raw_text = "".join(
                item[1] for item in results
            )

            normalized = re.sub(
                r"[^A-Z0-9]",
                "",
                raw_text.upper(),
            )

            if not normalized:
                continue

            confidence = float(
                np.mean([float(item[2]) for item in results])
            )

            candidates.append({
                "text": normalized,
                "confidence": confidence,
            })

        if not candidates:
            return {
                "text": "",
                "confidence": 0.0,
                "needs_review": True,
            }

        # Prefer the candidate with the highest confidence.
        best = max(
            candidates,
            key=lambda item: item["confidence"],
        )

        text = correct_plate_syntax(best["text"])
        confidence = best["confidence"]

        # Basic quality checks, not country-specific validation.
        needs_review = (
            confidence < 0.70
            or len(text) < 5
            or len(text) > 12
        )

        return {
            "text": text,
            "confidence": round(confidence, 4),
            "needs_review": needs_review,
        }
