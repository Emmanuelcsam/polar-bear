import pathlib
import sys
import tempfile
import numpy as np
import cv2

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root / 'current-process'))
sys.path.append(str(root))

from data_acquisition import DefectAggregator


def _create_dummy_image(path: pathlib.Path):
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_validate_defect_data_and_orientation():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)
        img_path = tmpdir_path / 'img.png'
        _create_dummy_image(img_path)
        aggregator = DefectAggregator(tmpdir_path, img_path)

        assert aggregator._validate_defect_data({'location_xy': (1, 2)})
        assert not aggregator._validate_defect_data({'location_xy': 'bad'})
        assert not aggregator._validate_defect_data({})

        assert aggregator.orientation_to_direction(0) == 'Horizontal'
        assert aggregator.orientation_to_direction(90) == 'Vertical'
        assert aggregator.orientation_to_direction(45) == 'Diagonal-NE'
        assert aggregator.orientation_to_direction(135) == 'Diagonal-NW'


def test_integrate_with_pipeline_invalid_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = pathlib.Path(tmpdir) / 'img.png'
        _create_dummy_image(img_path)
        non_existing = pathlib.Path(tmpdir) / 'missing'
        try:
            DefectAggregator(non_existing, img_path)
        except ValueError as e:
            assert 'Results directory does not exist' in str(e)
        else:
            raise AssertionError('Expected ValueError')
