import logging
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from indeca.utils.logging_config import (
    get_module_logger,
    log_dir,
    set_package_log_level,
)


class TestLoggingConfig:
    def test_basic_setup(self, temp_data_dir):
        """Test basic logging setup."""
        pass

    def test_log_file_creation(self, temp_data_dir):
        """Test log file creation and permissions."""
        # Set up a test logger
        logger = get_module_logger("test_file_creation")
        assert log_dir.exists()
        log_file = log_dir / "indeca.log"
        assert log_file.exists()

    @pytest.mark.skip(reason="Log formatting needs to be investigated")
    def test_log_formatting(self):
        """Test log message formatting."""
        logger = get_module_logger("test_formatting")
        logger.info("Test message")  # Write a test message
        with open(log_dir / "indeca.log", "r") as f:
            last_line = f.readlines()[-1]
            # Check basic log format components
            assert "test_formatting" in last_line
            assert "INFO" in last_line
            assert "Test message" in last_line

    def test_log_levels(self):
        """Test different logging levels."""
        pass

    def test_multiple_handlers(self):
        """Test multiple handler configuration."""
        pass

    def test_error_handling(self):
        """Test error handling in logging setup."""
        pass

    @pytest.mark.integration
    def test_logging_in_multiprocess(self):
        """Test logging in multiprocess environment."""
        pass


def test_set_package_log_level():
    """Test setting package log level."""
    # Test with string level
    set_package_log_level("DEBUG")
    logger = logging.getLogger("indeca")
    assert logger.level == logging.DEBUG

    # Test with integer level
    set_package_log_level(logging.INFO)
    assert logger.level == logging.INFO

    # Test invalid level
    with pytest.raises(ValueError):
        set_package_log_level("INVALID_LEVEL")


def test_get_module_logger():
    """Test getting module logger."""
    logger = get_module_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "indeca.test_module"
