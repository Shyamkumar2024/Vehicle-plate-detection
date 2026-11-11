import cv2

def assess_image(image):
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("Invalid or empty image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return {
        "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "is_dark": bool(gray.mean() < 50),
        "is_potentially_blurry": bool(cv2.Laplacian(gray, cv2.CV_64F).var() < 60),
    }
