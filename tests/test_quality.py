import numpy as np
import pytest
from src.quality import assess_image

def test_quality_fields():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = assess_image(image)
    assert {"blur_score", "brightness", "is_dark"}.issubset(result)
    assert result["is_dark"] is True

def test_invalid_image():
    with pytest.raises(ValueError):
        assess_image(None)
