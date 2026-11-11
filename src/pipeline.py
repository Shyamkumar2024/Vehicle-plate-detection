
from pathlib import Path

import cv2
from src.ocr import PlateOCR


def analyze_image(image, confidence=0.35):
    warnings = []
    vehicles = []
    annotated = image.copy()

    # 1. Find the vehicle model.
    candidates = [
        Path("models/vehicle_detector.pt"),
        Path("yolo26n.pt"),
    ]
    vehicle_path = next(
        (p for p in candidates if p.exists()), None
    )

    if vehicle_path is None:
        return {
            "vehicles": [],
            "annotated": annotated,
            "warnings": ["Vehicle model not found. Check yolo26n.pt."],
        }

    try:
        from ultralytics import YOLO

        vehicle_model = YOLO(str(vehicle_path))
        result = vehicle_model.predict(
            source=image,
            conf=confidence,
            verbose=False,
        )[0]

        names = vehicle_model.names

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                label = str(names[class_id]).lower()

                if label not in {
                    "car", "motorcycle", "bus", "truck"
                }:
                    continue

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )
                score = float(box.conf.item())

                vehicles.append({
                    "vehicle_type": label,
                    "confidence": round(score, 4),
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "plate_text": "",
                    "plate_confidence": None,
                    "ocr_confidence": 0.0,
                    "needs_review": True,
                })

                cv2.rectangle(
                    annotated, (x1, y1), (x2, y2),
                    (40, 200, 40), 2
                )
                cv2.putText(
                    annotated, f"{label} {score:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (40, 200, 40), 2
                )

    except Exception as exc:
        warnings.append(f"Vehicle detection failed: {exc}")

    # 2. Load the plate detector and OCR.
    plate_path = Path("models/plate_detector.pt")

    if not plate_path.exists():
        warnings.append(
            "Plate detector missing: models/plate_detector.pt"
        )
    elif vehicles:
        try:
            from ultralytics import YOLO

            plate_model = YOLO(str(plate_path))
            ocr = PlateOCR(gpu=False)

            # Detect all plates in the original image.
            plate_result = plate_model.predict(
                source=image,
                conf=0.20,
                imgsz=640,
                verbose=False,
            )[0]

            plates = []

            if plate_result.boxes is not None:
                for box in plate_result.boxes:
                    px1, py1, px2, py2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    px1 = max(0, px1)
                    py1 = max(0, py1)
                    px2 = min(image.shape[1], px2)
                    py2 = min(image.shape[0], py2)

                    if px2 <= px1 or py2 <= py1:
                        continue

                    plate_score = float(box.conf.item())
                    plate_crop = image[py1:py2, px1:px2]

                    # 3. Recognize characters from the plate crop.
                    ocr_result = ocr.read(plate_crop)

                    plate_text = ocr_result["text"]
                    ocr_score = float(
                        ocr_result["confidence"]
                    )

                    # Associate the plate with a vehicle whose
                    # bounding box contains the plate center.
                    cx = (px1 + px2) // 2
                    cy = (py1 + py2) // 2

                    matching_vehicle = None
                    smallest_area = float("inf")

                    for vehicle in vehicles:
                        inside = (
                            vehicle["x1"] <= cx <= vehicle["x2"]
                            and vehicle["y1"] <= cy <= vehicle["y2"]
                        )

                        if inside:
                            area = (
                                (vehicle["x2"] - vehicle["x1"])
                                * (vehicle["y2"] - vehicle["y1"])
                            )

                            if area < smallest_area:
                                matching_vehicle = vehicle
                                smallest_area = area

                    # Keep results for plates associated with vehicles.
                    if matching_vehicle is None:
                        continue

                    needs_review = bool(
                        ocr_result.get("needs_review", True)
                        or plate_score < 0.40
                        or ocr_score < 0.70
                        or len(plate_text) < 5
                        or len(plate_text) > 12
                    )

                    matching_vehicle["plate_text"] = plate_text
                    matching_vehicle["plate_confidence"] = round(
                        plate_score, 4
                    )
                    matching_vehicle["ocr_confidence"] = round(
                        ocr_score, 4
                    )
                    matching_vehicle["needs_review"] = needs_review

                    # Draw the plate box in blue.
                    cv2.rectangle(
                        annotated,
                        (px1, py1), (px2, py2),
                        (255, 100, 0), 2
                    )

                    display_text = plate_text or "Unreadable plate"
                    cv2.putText(
                        annotated,
                        display_text,
                        (px1, max(20, py1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 100, 0), 2
                    )

                    plates.append({
                        "text": plate_text,
                        "confidence": plate_score,
                        "ocr_confidence": ocr_score,
                    })

            if not plates:
                warnings.append(
                    "No license plates detected. Try a clearer image "
                    "or verify the plate detector model."
                )

        except Exception as exc:
            warnings.append(
                f"Plate detection/OCR failed: {exc}"
            )

    if not vehicles:
        warnings.append(
            "No supported vehicles detected. Try a clearer image "
            "or lower the confidence threshold."
        )

    return {
        "vehicles": vehicles,
        "annotated": annotated,
        "warnings": warnings,
    }
