# JPEG Decoder - Final Project

A complete implementation of a JPEG decoder from scratch in Python. This project demonstrates understanding of JPEG compression, Huffman coding, DCT transforms, and color space conversions.

## 🚀 Quick Start

**New to this project?** Check out [QUICKSTART.md](QUICKSTART.md) for a step-by-step guide!

```bash
# 1. Install dependencies
uv sync

# 2. Generate test images
source .venv/bin/activate
python create_test_images.py

# 3. Decode a JPEG
python main.py test_images/gradient.jpg --info
```

## Overview

This JPEG decoder implements the baseline JPEG decoding algorithm, which includes:

- **JPEG Marker Parsing**: Reading and interpreting JPEG file structure
- **Huffman Decoding**: Decoding variable-length Huffman codes
- **Quantization Table Processing**: Handling DCT coefficient dequantization
- **Inverse DCT (IDCT)**: Converting frequency domain to spatial domain
- **YCbCr to RGB Conversion**: Converting color spaces for display
- **Chroma Subsampling**: Handling different sampling factors

## Features

- ✅ Full baseline JPEG decoding support
- ✅ Support for grayscale and color (YCbCr) images
- ✅ Huffman table decoding (DC and AC coefficients)
- ✅ Quantization table support
- ✅ Inverse Discrete Cosine Transform (IDCT)
- ✅ Color space conversion (YCbCr → RGB)
- ✅ Command-line interface for easy usage
- ✅ Image comparison with PIL for validation

## Installation

This project uses `uv` for dependency management. Make sure you have `uv` installed.

```bash
# Clone the repository
git clone <repository-url>
cd videofinal

# Install dependencies
uv sync
```

### Dependencies

- Python 3.10+
- numpy: Matrix operations and numerical computing
- pillow: Image I/O and comparison
- scipy: Image upsampling for chroma components

## Usage

### Basic Usage

Decode a JPEG file:

```bash
python main.py input.jpg
```

### Save Decoded Image

Decode and save as PNG:

```bash
python main.py input.jpg -o output.png
```

### Display Image Information

Show detailed decoding information:

```bash
python main.py input.jpg --info
```

### Compare with PIL Decoder

Validate decoder accuracy:

```bash
python main.py input.jpg --compare
```

### Generate Test Images

Create sample JPEG images for testing:

```bash
python create_test_images.py
```

This will create test images in the `test_images/` directory:
- `gradient.jpg`: RGB gradient pattern
- `color_blocks.jpg`: Colored blocks for color testing
- `checkerboard.jpg`: Black and white checkerboard pattern
- `text.jpg`: Image with text overlay
- `grayscale.jpg`: Grayscale gradient

## Project Structure

```
videofinal/
├── main.py                    # Command-line interface
├── jpeg_decoder.py            # Core JPEG decoder implementation
├── create_test_images.py      # Test image generator
├── test_images/               # Sample JPEG test images
│   ├── gradient.jpg
│   ├── color_blocks.jpg
│   ├── checkerboard.jpg
│   ├── text.jpg
│   └── grayscale.jpg
├── pyproject.toml             # Project dependencies
└── README.md                  # This file
```

## How It Works

### JPEG Decoding Pipeline

1. **File Parsing**: Read JPEG file and parse markers
2. **Extract Metadata**: Read image dimensions, components, and tables
3. **Huffman Decoding**: Decode compressed data using Huffman tables
4. **Dequantization**: Multiply DCT coefficients by quantization tables
5. **IDCT**: Convert frequency domain to spatial domain
6. **Color Conversion**: Convert YCbCr to RGB color space
7. **Output**: Return decoded RGB image

### Key Components

#### JPEGDecoder Class

The main decoder class that handles the entire decoding process:

```python
from jpeg_decoder import JPEGDecoder

decoder = JPEGDecoder()
image = decoder.decode_file('image.jpg')
```

#### Marker Parsing

Handles JPEG markers including:
- `SOI` (Start of Image): 0xFFD8
- `DQT` (Define Quantization Table): 0xFFDB
- `SOF0` (Start of Frame): 0xFFC0
- `DHT` (Define Huffman Table): 0xFFC4
- `SOS` (Start of Scan): 0xFFDA
- `EOI` (End of Image): 0xFFD9

#### Huffman Decoding

Implements variable-length Huffman code decoding for both DC and AC coefficients.

#### IDCT (Inverse Discrete Cosine Transform)

Converts 8×8 blocks from frequency domain to spatial domain using optimized IDCT computation.

#### Color Space Conversion

Converts YCbCr color space to RGB using standard conversion formulas:
```
R = Y + 1.402 × Cr
G = Y - 0.344136 × Cb - 0.714136 × Cr
B = Y + 1.772 × Cb
```

## Examples

### Example 1: Basic Decoding

```python
from jpeg_decoder import decode_jpeg
import matplotlib.pyplot as plt

# Decode image
image = decode_jpeg('test_images/gradient.jpg')

# Display using matplotlib
plt.imshow(image)
plt.axis('off')
plt.show()
```

### Example 2: Using the Decoder Class

```python
from jpeg_decoder import JPEGDecoder

# Create decoder instance
decoder = JPEGDecoder()

# Decode file
image = decoder.decode_file('input.jpg')

# Access metadata
print(f"Image size: {decoder.width}x{decoder.height}")
print(f"Number of components: {len(decoder.components)}")
print(f"Quantization tables: {len(decoder.quantization_tables)}")
```

### Example 3: Command-Line Usage

```bash
# Test the decoder with generated images
python main.py test_images/gradient.jpg --info --compare

# Decode and save output
python main.py test_images/color_blocks.jpg -o decoded_output.png
```

## Implementation Details

### Huffman Table Construction

Huffman tables are built from the DHT (Define Huffman Table) segment, creating a lookup dictionary that maps (code, bit_length) tuples to symbol values.

### Zigzag Ordering

DCT coefficients are stored in zigzag order to group low-frequency components together. The decoder reorders them back to the standard 8×8 block format.

### Bit Stream Processing

The decoder maintains a bit buffer and bit counter to efficiently read variable-length Huffman codes from the compressed data stream, handling byte stuffing (0xFF00 → 0xFF).

### MCU (Minimum Coded Unit) Processing

Images are processed in MCUs, which can contain multiple 8×8 blocks depending on the chroma subsampling factors.

## Testing

Test the decoder with the provided test images:

```bash
# Generate test images
python create_test_images.py

# Test each image
python main.py test_images/gradient.jpg --compare
python main.py test_images/color_blocks.jpg --compare
python main.py test_images/checkerboard.jpg --compare
python main.py test_images/text.jpg --compare
python main.py test_images/grayscale.jpg --compare
```

## Limitations

- Supports baseline JPEG only (not progressive)
- Limited to 8-bit precision
- Does not support all JPEG markers and extensions
- Optimized for correctness over speed

## Performance

The decoder prioritizes clarity and correctness over raw performance. For production use, consider using optimized libraries like PIL/Pillow or libjpeg.

## Future Improvements

- [ ] Progressive JPEG support
- [ ] Arithmetic coding support
- [ ] Multi-threaded decoding
- [ ] Performance optimizations (JIT compilation, vectorization)
- [ ] Support for more color spaces
- [ ] EXIF metadata extraction

## Technical References

- ITU-T T.81: JPEG Standard
- JPEG File Interchange Format (JFIF)
- DCT and IDCT algorithms
- Huffman coding theory

## License

This is an educational project for learning purposes.

## Author

Final Project - Video Processing Course

## Acknowledgments

- JPEG specification: ITU-T Recommendation T.81
- Various JPEG decoding resources and tutorials
- NumPy and SciPy communities for scientific computing tools
