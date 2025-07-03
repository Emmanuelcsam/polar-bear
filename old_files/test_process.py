import os
import tempfile
import numpy as np
import cv2
import sys
import pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root / 'current-process'))
sys.path.append(str(root))
import process


def test_reimagine_image_tmp():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpimg = os.path.join(tmpdir, 'img.png')
        cv2.imwrite(tmpimg, img)
        result = process.reimagine_image(tmpimg, output_folder=tmpdir)
        assert isinstance(result, dict)
        assert result  # should contain transformations
