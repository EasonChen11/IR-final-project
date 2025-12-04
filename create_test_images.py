"""
Script to create test JPEG images for the decoder.

This script generates various JPEG test images including:
- Simple gradient images
- Color patterns
- Text-based images
"""

import numpy as np
from PIL import Image, ImageDraw


def create_gradient_image(filename: str, width: int = 256, height: int = 256):
    """
    Create a simple gradient image.

    Parameters:
    filename (str): Output filename.
    width (int): Image width.
    height (int): Image height.
    """
    # Create gradient
    gradient = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            gradient[i, j, 0] = int(255 * i / height)  # Red gradient
            gradient[i, j, 1] = int(255 * j / width)  # Green gradient
            gradient[i, j, 2] = 128  # Constant blue

    img = Image.fromarray(gradient)
    img.save(filename, "JPEG", quality=85)
    print(f"Created: {filename}")


def create_color_blocks(filename: str, width: int = 256, height: int = 256):
    """
    Create an image with colored blocks.

    Parameters:
    filename (str): Output filename.
    width (int): Image width.
    height (int): Image height.
    """
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # Define colors
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
    ]

    block_width = width // 4
    block_height = height // 2

    for i, color in enumerate(colors):
        row = i // 4
        col = i % 4
        x1 = col * block_width
        y1 = row * block_height
        x2 = x1 + block_width
        y2 = y1 + block_height
        draw.rectangle([x1, y1, x2, y2], fill=color)

    img.save(filename, "JPEG", quality=90)
    print(f"Created: {filename}")


def create_pattern_image(filename: str, width: int = 256, height: int = 256):
    """
    Create a checkerboard pattern image.

    Parameters:
    filename (str): Output filename.
    width (int): Image width.
    height (int): Image height.
    """
    pattern = np.zeros((height, width, 3), dtype=np.uint8)
    block_size = 16

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                pattern[i : i + block_size, j : j + block_size] = [255, 255, 255]
            else:
                pattern[i : i + block_size, j : j + block_size] = [0, 0, 0]

    img = Image.fromarray(pattern)
    img.save(filename, "JPEG", quality=95)
    print(f"Created: {filename}")


def create_text_image(filename: str, width: int = 400, height: int = 200):
    """
    Create an image with text.

    Parameters:
    filename (str): Output filename.
    width (int): Image width.
    height (int): Image height.
    """
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw text
    text = "JPEG Decoder\nTest Image"

    # Calculate text position (center)
    bbox = draw.textbbox((0, 0), text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill=(0, 0, 0))

    # Draw border
    draw.rectangle([0, 0, width - 1, height - 1], outline=(255, 0, 0), width=3)

    img.save(filename, "JPEG", quality=85)
    print(f"Created: {filename}")


def create_grayscale_image(filename: str, width: int = 256, height: int = 256):
    """
    Create a grayscale gradient image.

    Parameters:
    filename (str): Output filename.
    width (int): Image width.
    height (int): Image height.
    """
    gradient = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        gradient[i, :] = int(255 * i / height)

    img = Image.fromarray(gradient, mode="L")
    img.save(filename, "JPEG", quality=90)
    print(f"Created: {filename}")


def main():
    """Generate all test images."""
    print("Generating test JPEG images...\n")

    # Create test_images directory if it doesn't exist
    import os

    os.makedirs("test_images", exist_ok=True)

    # Generate test images
    create_gradient_image("test_images/gradient.jpg")
    create_color_blocks("test_images/color_blocks.jpg")
    create_pattern_image("test_images/checkerboard.jpg")
    create_text_image("test_images/text.jpg")
    create_grayscale_image("test_images/grayscale.jpg")

    print("\nAll test images created successfully!")
    print("Test images are located in the 'test_images/' directory")


if __name__ == "__main__":
    main()
