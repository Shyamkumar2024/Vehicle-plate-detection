import cv2
import numpy as np
from skimage.restoration import richardson_lucy


def enhance_low_light(image):
    """Enhance low-light or unevenly lit scenes using CLAHE in LAB color space."""
    if image is None or image.size == 0:
        raise ValueError("Empty image")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def unsharp_mask(image, sigma=1.2, strength=1.8):
    """High-boost unsharp masking filter to restore crisp edges in blurred plates."""
    if image is None or image.size == 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def create_motion_psf(length=9, angle_deg=0):
    """Generate a Point Spread Function (PSF) representing directional vehicle motion blur."""
    length = max(3, int(length))
    if length % 2 == 0:
        length += 1
    psf = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    rad = np.deg2rad(angle_deg)
    dx = np.cos(rad)
    dy = np.sin(rad)
    for i in range(-center, center + 1):
        x = int(round(center + i * dx))
        y = int(round(center + i * dy))
        if 0 <= x < length and 0 <= y < length:
            psf[y, x] = 1.0
    s = psf.sum()
    if s > 0:
        psf /= s
    else:
        psf[center, center] = 1.0
    return psf


def deblur_motion(image, kernel_size=9, angle=0, iterations=10):
    """Mathematical Richardson-Lucy deconvolution to reverse directional motion blur."""
    if image is None or image.size == 0:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    psf = create_motion_psf(length=kernel_size, angle_deg=angle)
    norm = gray.astype(np.float32) / 255.0
    deblurred = richardson_lucy(norm, psf, num_iter=max(5, int(iterations)))
    return (np.clip(deblurred, 0.0, 1.0) * 255.0).astype(np.uint8)


def denoise_bilateral(image, d=7, sigma_color=50, sigma_space=50):
    """Edge-preserving smoothing to eliminate high-frequency noise without blurring text."""
    if image is None or image.size == 0:
        return image
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def morphological_contrast(gray_image):
    """Gentle morphological leveling to eliminate shadows and glare without eroding character strokes."""
    if gray_image is None or gray_image.size == 0:
        return gray_image
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    tophat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, rect_kernel)
    # Add localized bright detail to balance background without subtracting dark character strokes
    enhanced = cv2.addWeighted(gray_image, 1.0, tophat, 0.5, 0)
    return enhanced


def deskew_plate(image, min_angle_deg=1.5):
    """
    Detect tilt angle of the license plate and rotate it to a horizontal baseline.
    Only corrects if the detected skew exceeds `min_angle_deg` to avoid unnecessary
    warping of already-level plates (which degrades OCR).
    """
    if image is None or image.size == 0:
        return image, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return image, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle
    # Constrain to realistic plate tilt; ignore if trivially small
    if abs(angle) > 20.0 or abs(angle) < min_angle_deg:
        return image, 0.0
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def smart_enhance_plate(plate_image, deskew=True):
    """
    Intelligent, non-destructive ALPR enhancement:
    - Upscales small plate crops to optimal OCR resolution (~100-140px height)
    - Straightens tilted perspective
    - Bilateral edge-preserving noise smoothing
    - Controlled CLAHE contrast expansion (leaves solid characters intact)
    - Mild edge sharpness boost
    """
    if plate_image is None or plate_image.size == 0:
        raise ValueError("Empty plate image")

    h, w = plate_image.shape[:2]
    scale = max(1.5, min(4.0, 120.0 / max(1, h)))
    enlarged = cv2.resize(plate_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    clean = enlarged
    tilt_angle = 0.0
    if deskew:
        clean, tilt_angle = deskew_plate(clean)

    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY) if len(clean.shape) == 3 else clean.copy()
    denoised = cv2.bilateralFilter(gray, 7, 40, 40)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    blurred = cv2.GaussianBlur(contrast, (0, 0), 1.0)
    sharpened = np.clip(cv2.addWeighted(contrast, 1.35, blurred, -0.35, 0), 0, 255).astype(np.uint8)
    return sharpened


def super_clean_plate(plate_image, deblur=False, deskew=True, sharpen=True, motion_angle=0):
    """
    Configurable multi-stage ALPR restoration pipeline.
    Note: Deconvolution (deblur=True) is only applied if explicit motion smear is selected.
    """
    if plate_image is None or plate_image.size == 0:
        raise ValueError("Empty plate image")

    h, w = plate_image.shape[:2]
    scale = max(1.5, min(4.0, 120.0 / max(1, h)))
    enlarged = cv2.resize(plate_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    clean = enlarged
    tilt_angle = 0.0
    if deskew:
        clean, tilt_angle = deskew_plate(clean)

    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY) if len(clean.shape) == 3 else clean.copy()
    denoised = denoise_bilateral(gray, d=7, sigma_color=40, sigma_space=40)

    if deblur:
        deblurred = deblur_motion(denoised, kernel_size=9, angle=motion_angle, iterations=8)
    else:
        deblurred = denoised

    if sharpen:
        sharpened = unsharp_mask(deblurred, sigma=1.0, strength=0.6)
    else:
        sharpened = deblurred

    morph = morphological_contrast(sharpened)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    final_enhanced = clahe.apply(morph)
    return final_enhanced


def enhance_plate(plate_image):
    """Non-destructive default enhancement for OCR."""
    return smart_enhance_plate(plate_image, deskew=True)

