"""
JPEG Decoder Main Application

This is the main entry point for demonstrating the JPEG decoder functionality.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from jpeg_decoder import JPEGDecoder


def main():
    """Main function to run the JPEG decoder."""
    parser = argparse.ArgumentParser(
        description="JPEG Decoder - Decode and display JPEG images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decode a JPEG file and save as PNG
  python main.py input.jpg -o output.png
  
  # Decode and display image information
  python main.py input.jpg --info
  
  # Compare with PIL decoder
  python main.py input.jpg --compare
        """,
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input JPEG file path",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output image file path (PNG format)",
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Display image information",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with PIL decoder",
    )

    args = parser.parse_args()

    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        return 1

    try:
        print(f"Decoding JPEG image: {args.input}")

        # Decode using custom decoder
        decoder = JPEGDecoder()
        decoded_image = decoder.decode_file(str(input_path))

        print("Successfully decoded image:")
        print(f"  - Dimensions: {decoder.width} x {decoder.height}")
        print(f"  - Components: {len(decoder.components)}")
        print(f"  - Image shape: {decoded_image.shape}")

        if args.info:
            # Display detailed information
            print("\nDetailed Information:")
            print(f"  - Quantization tables: {len(decoder.quantization_tables)}")
            print(f"  - DC Huffman tables: {len(decoder.huffman_dc_tables)}")
            print(f"  - AC Huffman tables: {len(decoder.huffman_ac_tables)}")

            print("\nComponents:")
            for i, comp in enumerate(decoder.components):
                print(f"  Component {i}:")
                print(f"    - ID: {comp['id']}")
                print(f"    - H sampling: {comp['h_sampling']}")
                print(f"    - V sampling: {comp['v_sampling']}")
                print(f"    - Quantization table: {comp['qt_id']}")
                print(f"    - DC table: {comp['dc_table_id']}")
                print(f"    - AC table: {comp['ac_table_id']}")

        if args.compare:
            # Compare with PIL decoder
            print("\nComparing with PIL decoder...")
            pil_image = np.array(Image.open(str(input_path)))

            print(f"  PIL image shape: {pil_image.shape}")
            print(f"  Custom decoder shape: {decoded_image.shape}")

            # Calculate differences
            if pil_image.shape == decoded_image.shape:
                diff = np.abs(
                    pil_image.astype(np.float32) - decoded_image.astype(np.float32)
                )
                mae = np.mean(diff)
                max_diff = np.max(diff)

                print("\nImage Comparison:")
                print(f"  - Mean Absolute Error: {mae:.2f}")
                print(f"  - Maximum Difference: {max_diff:.2f}")

                if mae < 5.0:
                    print("  - Quality: Excellent match!")
                elif mae < 10.0:
                    print("  - Quality: Good match")
                elif mae < 20.0:
                    print("  - Quality: Acceptable match")
                else:
                    print("  - Quality: Significant differences detected")
            else:
                print("  Warning: Image dimensions don't match")

        if args.output:
            # Save output image
            output_path = Path(args.output)
            output_image = Image.fromarray(decoded_image)
            output_image.save(str(output_path))
            print(f"\nSaved decoded image to: {args.output}")

        return 0

    except Exception as e:
        print(f"Error decoding JPEG: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
