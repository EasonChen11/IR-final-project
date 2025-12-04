"""
Demo script to showcase the JPEG decoder functionality.

This script demonstrates various features of the JPEG decoder including
decoding, comparison, and information display.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from jpeg_decoder import JPEGDecoder


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def demo_basic_decoding():
    """Demonstrate basic JPEG decoding."""
    print_header("Demo 1: Basic JPEG Decoding")

    test_image = "test_images/gradient.jpg"
    if not Path(test_image).exists():
        print(f"Error: {test_image} not found. Run create_test_images.py first.")
        return

    print(f"\nDecoding: {test_image}")

    decoder = JPEGDecoder()
    image = decoder.decode_file(test_image)

    print("✓ Successfully decoded!")
    print(f"  Image dimensions: {decoder.width} × {decoder.height}")
    print(f"  Output shape: {image.shape}")
    print(f"  Data type: {image.dtype}")
    print(f"  Pixel range: [{np.min(image)}, {np.max(image)}]")


def demo_image_info():
    """Demonstrate extracting detailed image information."""
    print_header("Demo 2: Detailed Image Information")

    test_image = "test_images/color_blocks.jpg"
    if not Path(test_image).exists():
        print(f"Error: {test_image} not found.")
        return

    print(f"\nAnalyzing: {test_image}")

    decoder = JPEGDecoder()
    decoder.decode_file(test_image)

    print("\n📊 Image Metadata:")
    print(f"  Dimensions: {decoder.width} × {decoder.height}")
    print(f"  Components: {len(decoder.components)}")

    print("\n🔢 Compression Tables:")
    print(f"  Quantization tables: {len(decoder.quantization_tables)}")
    print(f"  DC Huffman tables: {len(decoder.huffman_dc_tables)}")
    print(f"  AC Huffman tables: {len(decoder.huffman_ac_tables)}")

    print("\n🎨 Color Components:")
    for i, comp in enumerate(decoder.components):
        print(f"  Component {i} (ID={comp['id']}):")
        print(f"    - Sampling: {comp['h_sampling']}×{comp['v_sampling']}")
        print(f"    - Quantization table: {comp['qt_id']}")
        print(
            f"    - Huffman tables: DC={comp['dc_table_id']}, AC={comp['ac_table_id']}"
        )


def demo_comparison():
    """Demonstrate comparison with PIL."""
    print_header("Demo 3: Decoder Validation")

    test_image = "test_images/checkerboard.jpg"
    if not Path(test_image).exists():
        print(f"Error: {test_image} not found.")
        return

    print(f"\nComparing decoder output with PIL for: {test_image}")

    # Decode with custom decoder
    decoder = JPEGDecoder()
    custom_decoded = decoder.decode_file(test_image)

    # Decode with PIL
    pil_decoded = np.array(Image.open(test_image))

    print("\n📐 Dimension Check:")
    print(f"  Custom decoder: {custom_decoded.shape}")
    print(f"  PIL decoder: {pil_decoded.shape}")

    if custom_decoded.shape == pil_decoded.shape:
        print("  ✓ Dimensions match!")

        # Calculate differences
        diff = np.abs(
            custom_decoded.astype(np.float32) - pil_decoded.astype(np.float32)
        )
        mae = np.mean(diff)
        max_diff = np.max(diff)

        print("\n📊 Accuracy Metrics:")
        print(f"  Mean Absolute Error: {mae:.2f}")
        print(f"  Maximum Difference: {max_diff:.2f}")

        if mae < 5.0:
            print("  ✓ Excellent match!")
        elif mae < 10.0:
            print("  ✓ Good match")
        elif mae < 20.0:
            print("  ⚠ Acceptable match")
        else:
            print("  ⚠ Significant differences detected")
    else:
        print("  ✗ Dimension mismatch!")


def demo_all_test_images():
    """Test decoder on all available test images."""
    print_header("Demo 4: Testing All Images")

    test_images_dir = Path("test_images")
    if not test_images_dir.exists():
        print(
            "Error: test_images directory not found. Run create_test_images.py first."
        )
        return

    jpeg_files = list(test_images_dir.glob("*.jpg"))

    if not jpeg_files:
        print("No test images found.")
        return

    print(f"\nFound {len(jpeg_files)} test images\n")

    for jpeg_file in sorted(jpeg_files):
        print(f"Testing: {jpeg_file.name}")

        try:
            decoder = JPEGDecoder()
            image = decoder.decode_file(str(jpeg_file))

            print("  ✓ Decoded successfully")
            print(f"    Size: {decoder.width}×{decoder.height}")
            print(f"    Components: {len(decoder.components)}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

        print()


def demo_save_output():
    """Demonstrate saving decoded images."""
    print_header("Demo 5: Saving Decoded Images")

    test_image = "test_images/gradient.jpg"
    output_image = "demo_output.png"

    if not Path(test_image).exists():
        print(f"Error: {test_image} not found.")
        return

    print(f"\nDecoding: {test_image}")
    print(f"Saving to: {output_image}")

    decoder = JPEGDecoder()
    image = decoder.decode_file(test_image)

    # Save as PNG
    pil_image = Image.fromarray(image)
    pil_image.save(output_image)

    print("✓ Image saved successfully!")
    print(f"  Original: {test_image}")
    print(f"  Decoded: {output_image}")

    # Verify file was created
    if Path(output_image).exists():
        file_size = Path(output_image).stat().st_size
        print(f"  Output file size: {file_size:,} bytes")


def main():
    """Run all demo functions."""
    print("\n" + "=" * 70)
    print("  JPEG DECODER DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo showcases the JPEG decoder implementation.")
    print("Make sure you've run 'python create_test_images.py' first.")

    demos = [
        demo_basic_decoding,
        demo_image_info,
        demo_comparison,
        demo_all_test_images,
        demo_save_output,
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\nError in {demo.__name__}: {e}")
            import traceback

            traceback.print_exc()

    print_header("Demo Complete!")
    print("\nFor more options, try:")
    print("  python main.py test_images/gradient.jpg --info")
    print("  python main.py test_images/gradient.jpg --compare")
    print("  python main.py test_images/gradient.jpg -o output.png")
    print()


if __name__ == "__main__":
    main()
