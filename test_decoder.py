"""
Unit tests for the JPEG decoder.

This module contains tests to validate the JPEG decoder implementation.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jpeg_decoder import JPEGDecoder, decode_jpeg


def test_decoder_initialization():
    """Test that the decoder initializes correctly."""
    decoder = JPEGDecoder()

    assert decoder.width == 0
    assert decoder.height == 0
    assert len(decoder.components) == 0
    assert len(decoder.quantization_tables) == 0
    assert len(decoder.huffman_dc_tables) == 0
    assert len(decoder.huffman_ac_tables) == 0


def test_decode_file_exists():
    """Test that decode_file handles missing files correctly."""
    decoder = JPEGDecoder()

    with pytest.raises(FileNotFoundError):
        decoder.decode_file("nonexistent_file.jpg")


def test_decode_checkerboard():
    """Test decoding a simple checkerboard pattern."""
    test_image_path = "test_images/checkerboard.jpg"

    if not Path(test_image_path).exists():
        pytest.skip("Test image not found")

    # Decode the image
    decoder = JPEGDecoder()
    decoded_image = decoder.decode_file(test_image_path)

    # Verify basic properties
    assert decoded_image.shape[2] == 3  # RGB image
    assert decoded_image.dtype == np.uint8
    assert decoder.width > 0
    assert decoder.height > 0
    assert decoded_image.shape[0] == decoder.height
    assert decoded_image.shape[1] == decoder.width


def test_decode_gradient():
    """Test decoding a gradient image."""
    test_image_path = "test_images/gradient.jpg"

    if not Path(test_image_path).exists():
        pytest.skip("Test image not found")

    # Decode the image
    decoded_image = decode_jpeg(test_image_path)

    # Verify basic properties
    assert decoded_image.shape[2] == 3  # RGB image
    assert decoded_image.dtype == np.uint8
    assert np.min(decoded_image) >= 0
    assert np.max(decoded_image) <= 255


def test_decode_grayscale():
    """Test decoding a grayscale image."""
    test_image_path = "test_images/grayscale.jpg"

    if not Path(test_image_path).exists():
        pytest.skip("Test image not found")

    # Decode the image
    decoded_image = decode_jpeg(test_image_path)

    # Verify basic properties
    assert decoded_image.shape[2] == 3  # Still returns RGB
    assert decoded_image.dtype == np.uint8


def test_components_parsing():
    """Test that image components are correctly parsed."""
    test_image_path = "test_images/color_blocks.jpg"

    if not Path(test_image_path).exists():
        pytest.skip("Test image not found")

    decoder = JPEGDecoder()
    decoder.decode_file(test_image_path)

    # Check components
    assert len(decoder.components) > 0
    for comp in decoder.components:
        assert "id" in comp
        assert "h_sampling" in comp
        assert "v_sampling" in comp
        assert "qt_id" in comp


def test_quantization_tables():
    """Test that quantization tables are loaded."""
    test_image_path = "test_images/color_blocks.jpg"

    if not Path(test_image_path).exists():
        pytest.skip("Test image not found")

    decoder = JPEGDecoder()
    decoder.decode_file(test_image_path)

    # Check quantization tables
    assert len(decoder.quantization_tables) > 0
    for qt_id, qt in decoder.quantization_tables.items():
        assert qt.shape == (8, 8)
        assert np.all(qt > 0)  # All values should be positive


def test_huffman_tables():
    """Test that Huffman tables are loaded."""
    test_image_path = "test_images/color_blocks.jpg"

    if not Path(test_image_path).exists():
        pytest.skip("Test image not found")

    decoder = JPEGDecoder()
    decoder.decode_file(test_image_path)

    # Check Huffman tables
    assert len(decoder.huffman_dc_tables) > 0
    assert len(decoder.huffman_ac_tables) > 0


def test_image_dimensions():
    """Test that decoded image has correct dimensions."""
    test_image_path = "test_images/checkerboard.jpg"

    if not Path(test_image_path).exists():
        pytest.skip("Test image not found")

    # Get original dimensions
    original_image = Image.open(test_image_path)
    original_width, original_height = original_image.size

    # Decode and check
    decoded_image = decode_jpeg(test_image_path)

    assert decoded_image.shape[0] == original_height
    assert decoded_image.shape[1] == original_width


def test_pixel_value_range():
    """Test that pixel values are in valid range [0, 255]."""
    test_image_path = "test_images/gradient.jpg"

    if not Path(test_image_path).exists():
        pytest.skip("Test image not found")

    decoded_image = decode_jpeg(test_image_path)

    assert np.all(decoded_image >= 0)
    assert np.all(decoded_image <= 255)


def test_idct_table_initialization():
    """Test that IDCT table is properly initialized."""
    decoder = JPEGDecoder()

    # Check IDCT table shape
    assert hasattr(decoder, "idct_table")
    assert decoder.idct_table.shape == (8, 8, 8, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
