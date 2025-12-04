# Quick Start Guide

Get started with the JPEG decoder in 3 simple steps!

## Step 1: Setup

Install dependencies using `uv`:

```bash
# Navigate to project directory
cd videofinal

# Install dependencies (creates virtual environment automatically)
uv sync
```

## Step 2: Generate Test Images

Create sample JPEG images for testing:

```bash
# Activate virtual environment
source .venv/bin/activate

# Generate test images
python create_test_images.py
```

This creates 5 test images in the `test_images/` directory.

## Step 3: Run the Decoder

Try these commands:

```bash
# Basic decoding
python main.py test_images/gradient.jpg

# Show detailed info
python main.py test_images/color_blocks.jpg --info

# Compare with PIL
python main.py test_images/checkerboard.jpg --compare

# Save decoded output
python main.py test_images/text.jpg -o my_output.png

# Run the demo
python demo.py

# Run unit tests
pytest test_decoder.py -v
```

## Example Output

```
$ python main.py test_images/gradient.jpg --info

Decoding JPEG image: test_images/gradient.jpg
Successfully decoded image:
  - Dimensions: 256 x 256
  - Components: 3
  - Image shape: (256, 256, 3)

Detailed Information:
  - Quantization tables: 2
  - DC Huffman tables: 2
  - AC Huffman tables: 2

Components:
  Component 0:
    - ID: 1
    - H sampling: 2
    - V sampling: 2
    - Quantization table: 0
    - DC table: 0
    - AC table: 0
  ...
```

## Using the Decoder in Your Code

```python
from jpeg_decoder import decode_jpeg
import matplotlib.pyplot as plt

# Decode a JPEG file
image = decode_jpeg('your_image.jpg')

# Display it
plt.imshow(image)
plt.show()

# Or save it
from PIL import Image
Image.fromarray(image).save('output.png')
```

## Need Help?

- See `README.md` for full documentation
- Check `PROJECT_SUMMARY.md` for technical details
- Run `python main.py --help` for CLI options

## All Commands at a Glance

```bash
# Setup
uv sync
source .venv/bin/activate

# Generate test images
python create_test_images.py

# Decode images
python main.py <input.jpg> [options]

# Run demo
python demo.py

# Run tests
pytest test_decoder.py -v
```

That's it! You're ready to decode JPEG images. 🎉
