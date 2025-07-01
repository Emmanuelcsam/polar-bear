import os
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import debug_utils


def test_setup_logging_creates_file(tmp_path):
    os.environ['DEBUG_MODE'] = '1'
    log_file = tmp_path / 'log.txt'
    os.environ['DEBUG_LOG_FILE'] = str(log_file)
    logger = debug_utils.setup_logging('test')
    logger.debug('test message')
    assert log_file.exists()
    with open(log_file) as f:
        assert 'test message' in f.read()
    os.environ.pop('DEBUG_MODE', None)
    os.environ.pop('DEBUG_LOG_FILE', None)
