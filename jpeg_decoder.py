"""
JPEG Decoder Implementation

This module implements a complete JPEG decoder that can parse and decode
baseline JPEG images. It handles:
- JPEG marker parsing
- Huffman table decoding
- Quantization tables
- Inverse Discrete Cosine Transform (IDCT)
- YCbCr to RGB color space conversion
"""

import struct
from typing import Dict, List, Optional

import numpy as np


class JPEGDecoder:
    """
    A complete JPEG decoder implementation.

    This decoder supports baseline JPEG images with YCbCr color space
    and standard Huffman coding.
    """

    # JPEG markers
    MARKERS = {
        0xFFD8: "SOI",  # Start of Image
        0xFFE0: "APP0",  # Application segment 0 (JFIF)
        0xFFDB: "DQT",  # Define Quantization Table
        0xFFC0: "SOF0",  # Start of Frame (Baseline DCT)
        0xFFC4: "DHT",  # Define Huffman Table
        0xFFDA: "SOS",  # Start of Scan
        0xFFD9: "EOI",  # End of Image
        0xFFFE: "COM",  # Comment
    }

    def __init__(self):
        """Initialize the JPEG decoder with empty state."""
        self.width: int = 0
        self.height: int = 0
        self.components: List[Dict] = []
        self.quantization_tables: Dict[int, np.ndarray] = {}
        self.huffman_dc_tables: Dict[int, Dict] = {}
        self.huffman_ac_tables: Dict[int, Dict] = {}
        self.data: Optional[bytes] = None
        self.bit_buffer: int = 0
        self.bit_count: int = 0
        self.data_position: int = 0

        # IDCT cosine table for optimization
        self._init_idct_table()

    def _init_idct_table(self):
        """Pre-compute cosine values for IDCT optimization."""
        self.idct_table = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x in range(8):
            for y in range(8):
                for u in range(8):
                    for v in range(8):
                        cu = 1.0 / np.sqrt(2.0) if u == 0 else 1.0
                        cv = 1.0 / np.sqrt(2.0) if v == 0 else 1.0
                        self.idct_table[x, y, u, v] = (
                            0.25
                            * cu
                            * cv
                            * np.cos((2 * x + 1) * u * np.pi / 16.0)
                            * np.cos((2 * y + 1) * v * np.pi / 16.0)
                        )

    def decode_file(self, filename: str) -> np.ndarray:
        """
        Decode a JPEG file and return the image as a numpy array.

        Parameters:
        filename (str): Path to the JPEG file.

        Returns:
        np.ndarray: Decoded image in RGB format with shape (height, width, 3).
        """
        with open(filename, "rb") as f:
            self.data = f.read()

        return self.decode()

    def decode(self) -> np.ndarray:
        """
        Decode JPEG data and return the image.

        Returns:
        np.ndarray: Decoded image in RGB format.
        """
        self.data_position = 0

        # Parse JPEG markers
        while self.data_position < len(self.data):
            marker = self._read_marker()

            if marker is None:
                continue

            if marker == 0xFFD8:  # SOI
                continue
            elif marker == 0xFFD9:  # EOI
                break
            elif marker == 0xFFDB:  # DQT
                self._parse_dqt()
            elif marker == 0xFFC0:  # SOF0
                self._parse_sof0()
            elif marker == 0xFFC4:  # DHT
                self._parse_dht()
            elif marker == 0xFFDA:  # SOS
                self._parse_sos()
                image_data = self._decode_scan()
                return self._process_image(image_data)
            else:
                # Skip unknown marker segments
                self._skip_segment()

        raise ValueError("Invalid JPEG file: No image data found")

    def _read_marker(self) -> Optional[int]:
        """
        Read the next JPEG marker from the data stream.

        Returns:
        Optional[int]: The marker value or None if not found.
        """
        if self.data_position >= len(self.data) - 1:
            return None

        # Look for 0xFF
        while self.data_position < len(self.data) - 1:
            if self.data[self.data_position] == 0xFF:
                next_byte = self.data[self.data_position + 1]
                if next_byte != 0x00 and next_byte != 0xFF:
                    marker = (0xFF << 8) | next_byte
                    self.data_position += 2
                    return marker
            self.data_position += 1

        return None

    def _skip_segment(self):
        """Skip the current marker segment."""
        if self.data_position >= len(self.data) - 2:
            return

        length = struct.unpack(
            ">H", self.data[self.data_position : self.data_position + 2]
        )[0]
        self.data_position += length

    def _parse_dqt(self):
        """Parse Define Quantization Table (DQT) segment."""
        length = struct.unpack(
            ">H", self.data[self.data_position : self.data_position + 2]
        )[0]
        self.data_position += 2

        end_position = self.data_position + length - 2

        while self.data_position < end_position:
            qt_info = self.data[self.data_position]
            self.data_position += 1

            qt_precision = (qt_info >> 4) & 0x0F
            qt_id = qt_info & 0x0F

            qt_size = 64 * (2 if qt_precision else 1)
            qt_data = self.data[self.data_position : self.data_position + qt_size]
            self.data_position += qt_size

            if qt_precision:
                # 16-bit precision
                qt_values = struct.unpack(f">{64}H", qt_data)
            else:
                # 8-bit precision
                qt_values = struct.unpack(f"{64}B", qt_data)

            # Store in zigzag order
            self.quantization_tables[qt_id] = np.array(
                qt_values, dtype=np.float32
            ).reshape(8, 8)

    def _parse_sof0(self):
        """Parse Start of Frame (SOF0) segment."""
        # Skip length field
        self.data_position += 2

        # Skip precision field
        self.data_position += 1

        self.height = struct.unpack(
            ">H", self.data[self.data_position : self.data_position + 2]
        )[0]
        self.data_position += 2

        self.width = struct.unpack(
            ">H", self.data[self.data_position : self.data_position + 2]
        )[0]
        self.data_position += 2

        num_components = self.data[self.data_position]
        self.data_position += 1

        self.components = []
        for _ in range(num_components):
            component_id = self.data[self.data_position]
            self.data_position += 1

            sampling_factors = self.data[self.data_position]
            h_sampling = (sampling_factors >> 4) & 0x0F
            v_sampling = sampling_factors & 0x0F
            self.data_position += 1

            qt_id = self.data[self.data_position]
            self.data_position += 1

            self.components.append(
                {
                    "id": component_id,
                    "h_sampling": h_sampling,
                    "v_sampling": v_sampling,
                    "qt_id": qt_id,
                    "dc_table_id": 0,
                    "ac_table_id": 0,
                }
            )

    def _parse_dht(self):
        """Parse Define Huffman Table (DHT) segment."""
        length = struct.unpack(
            ">H", self.data[self.data_position : self.data_position + 2]
        )[0]
        self.data_position += 2

        end_position = self.data_position + length - 2

        while self.data_position < end_position:
            ht_info = self.data[self.data_position]
            self.data_position += 1

            ht_type = (ht_info >> 4) & 0x0F  # 0 = DC, 1 = AC
            ht_id = ht_info & 0x0F

            # Read the number of codes for each length
            num_codes = list(self.data[self.data_position : self.data_position + 16])
            self.data_position += 16

            # Read the Huffman values
            total_codes = sum(num_codes)
            huffman_values = list(
                self.data[self.data_position : self.data_position + total_codes]
            )
            self.data_position += total_codes

            # Build Huffman table
            huffman_table = self._build_huffman_table(num_codes, huffman_values)

            if ht_type == 0:
                self.huffman_dc_tables[ht_id] = huffman_table
            else:
                self.huffman_ac_tables[ht_id] = huffman_table

    def _build_huffman_table(self, num_codes: List[int], values: List[int]) -> Dict:
        """
        Build a Huffman lookup table.

        Parameters:
        num_codes (List[int]): Number of codes for each bit length.
        values (List[int]): Huffman values.

        Returns:
        Dict: Huffman lookup table mapping codes to values.
        """
        huffman_table = {}
        code = 0
        value_index = 0

        for bit_length in range(1, 17):
            for _ in range(num_codes[bit_length - 1]):
                if value_index < len(values):
                    huffman_table[(code, bit_length)] = values[value_index]
                    value_index += 1
                    code += 1
            code <<= 1

        return huffman_table

    def _parse_sos(self):
        """Parse Start of Scan (SOS) segment."""
        # Skip length field
        self.data_position += 2

        num_components = self.data[self.data_position]
        self.data_position += 1

        for _ in range(num_components):
            component_id = self.data[self.data_position]
            self.data_position += 1

            table_ids = self.data[self.data_position]
            dc_table_id = (table_ids >> 4) & 0x0F
            ac_table_id = table_ids & 0x0F
            self.data_position += 1

            # Update component table IDs
            for comp in self.components:
                if comp["id"] == component_id:
                    comp["dc_table_id"] = dc_table_id
                    comp["ac_table_id"] = ac_table_id

        # Skip spectral selection and successive approximation
        self.data_position += 3

    def _decode_scan(self) -> List[np.ndarray]:
        """
        Decode the scan data (compressed image data).

        Returns:
        List[np.ndarray]: Decoded MCU blocks for each component.
        """
        self.bit_buffer = 0
        self.bit_count = 0

        # Calculate MCU dimensions
        max_h = max(comp["h_sampling"] for comp in self.components)
        max_v = max(comp["v_sampling"] for comp in self.components)

        mcu_width = (self.width + 8 * max_h - 1) // (8 * max_h)
        mcu_height = (self.height + 8 * max_v - 1) // (8 * max_v)

        # Initialize component data
        component_data = []
        dc_predictors = []

        for comp in self.components:
            blocks_h = mcu_width * comp["h_sampling"]
            blocks_v = mcu_height * comp["v_sampling"]
            component_data.append(
                np.zeros((blocks_v * 8, blocks_h * 8), dtype=np.float32)
            )
            dc_predictors.append(0)

        # Decode MCUs
        for mcu_y in range(mcu_height):
            for mcu_x in range(mcu_width):
                for comp_idx, comp in enumerate(self.components):
                    for v in range(comp["v_sampling"]):
                        for h in range(comp["h_sampling"]):
                            # Decode 8x8 block
                            block = self._decode_block(
                                comp["dc_table_id"],
                                comp["ac_table_id"],
                                comp["qt_id"],
                                dc_predictors[comp_idx],
                            )
                            dc_predictors[comp_idx] = block[0, 0]

                            # Place block in component data
                            block_x = (mcu_x * comp["h_sampling"] + h) * 8
                            block_y = (mcu_y * comp["v_sampling"] + v) * 8

                            component_data[comp_idx][
                                block_y : block_y + 8, block_x : block_x + 8
                            ] = block

        return component_data

    def _decode_block(
        self, dc_table_id: int, ac_table_id: int, qt_id: int, dc_predictor: float
    ) -> np.ndarray:
        """
        Decode a single 8x8 block.

        Parameters:
        dc_table_id (int): DC Huffman table ID.
        ac_table_id (int): AC Huffman table ID.
        qt_id (int): Quantization table ID.
        dc_predictor (float): Previous DC value for prediction.

        Returns:
        np.ndarray: Decoded 8x8 block after IDCT.
        """
        # Zigzag pattern for reordering coefficients
        zigzag = np.array(
            [
                0,
                1,
                5,
                6,
                14,
                15,
                27,
                28,
                2,
                4,
                7,
                13,
                16,
                26,
                29,
                42,
                3,
                8,
                12,
                17,
                25,
                30,
                41,
                43,
                9,
                11,
                18,
                24,
                31,
                40,
                44,
                53,
                10,
                19,
                23,
                32,
                39,
                45,
                52,
                54,
                20,
                22,
                33,
                38,
                46,
                51,
                55,
                60,
                21,
                34,
                37,
                47,
                50,
                56,
                59,
                61,
                35,
                36,
                48,
                49,
                57,
                58,
                62,
                63,
            ]
        )

        coefficients = np.zeros(64, dtype=np.float32)

        # Decode DC coefficient
        dc_length = self._decode_huffman(self.huffman_dc_tables[dc_table_id])
        if dc_length > 0:
            dc_value = self._receive_bits(dc_length)
            dc_value = self._extend(dc_value, dc_length)
        else:
            dc_value = 0

        coefficients[0] = dc_value + dc_predictor

        # Decode AC coefficients
        k = 1
        while k < 64:
            ac_symbol = self._decode_huffman(self.huffman_ac_tables[ac_table_id])

            if ac_symbol == 0x00:  # EOB (End of Block)
                break

            if ac_symbol == 0xF0:  # ZRL (16 zeros)
                k += 16
                continue

            run_length = (ac_symbol >> 4) & 0x0F
            ac_length = ac_symbol & 0x0F

            k += run_length

            if k >= 64:
                break

            if ac_length > 0:
                ac_value = self._receive_bits(ac_length)
                ac_value = self._extend(ac_value, ac_length)
                coefficients[k] = ac_value

            k += 1

        # Reorder using zigzag pattern
        block_1d = coefficients[zigzag]
        block = block_1d.reshape(8, 8)

        # Dequantize
        qt = self.quantization_tables[qt_id]
        block = block * qt

        # Apply IDCT
        block = self._idct(block)

        return block

    def _decode_huffman(self, huffman_table: Dict) -> int:
        """
        Decode a Huffman code from the bit stream.

        Parameters:
        huffman_table (Dict): Huffman lookup table.

        Returns:
        int: Decoded value.
        """
        code = 0
        for bit_length in range(1, 17):
            bit = self._next_bit()
            code = (code << 1) | bit

            if (code, bit_length) in huffman_table:
                return huffman_table[(code, bit_length)]

        raise ValueError(f"Invalid Huffman code: {bin(code)}")

    def _next_bit(self) -> int:
        """
        Read the next bit from the bit stream.

        Returns:
        int: The next bit (0 or 1).
        """
        if self.bit_count == 0:
            self.bit_buffer = self.data[self.data_position]
            self.data_position += 1

            # Handle byte stuffing (0xFF00 -> 0xFF)
            if self.bit_buffer == 0xFF:
                next_byte = self.data[self.data_position]
                if next_byte == 0x00:
                    self.data_position += 1
                else:
                    # Marker found, back up
                    self.data_position -= 1

            self.bit_count = 8

        bit = (self.bit_buffer >> 7) & 1
        self.bit_buffer = (self.bit_buffer << 1) & 0xFF
        self.bit_count -= 1

        return bit

    def _receive_bits(self, num_bits: int) -> int:
        """
        Read a specified number of bits from the stream.

        Parameters:
        num_bits (int): Number of bits to read.

        Returns:
        int: The value of the bits read.
        """
        value = 0
        for _ in range(num_bits):
            value = (value << 1) | self._next_bit()
        return value

    def _extend(self, value: int, num_bits: int) -> int:
        """
        Extend the sign of a value.

        Parameters:
        value (int): The value to extend.
        num_bits (int): Number of bits in the value.

        Returns:
        int: Sign-extended value.
        """
        vt = 1 << (num_bits - 1)
        if value < vt:
            return value - (1 << num_bits) + 1
        return value

    def _idct(self, block: np.ndarray) -> np.ndarray:
        """
        Perform Inverse Discrete Cosine Transform (IDCT) on an 8x8 block.

        Parameters:
        block (np.ndarray): 8x8 block of DCT coefficients.

        Returns:
        np.ndarray: 8x8 block of spatial domain values.
        """
        # Optimized IDCT using separable transforms
        result = np.zeros((8, 8), dtype=np.float32)

        for x in range(8):
            for y in range(8):
                sum_val = 0.0
                for u in range(8):
                    for v in range(8):
                        sum_val += block[v, u] * self.idct_table[x, y, u, v]
                result[y, x] = sum_val

        return result

    def _process_image(self, component_data: List[np.ndarray]) -> np.ndarray:
        """
        Process component data and convert to RGB image.

        Parameters:
        component_data (List[np.ndarray]): List of component arrays.

        Returns:
        np.ndarray: RGB image array.
        """
        # Crop to actual image size
        y_data = component_data[0][: self.height, : self.width]

        if len(component_data) == 1:
            # Grayscale image
            image = np.clip(y_data + 128, 0, 255).astype(np.uint8)
            return np.stack([image, image, image], axis=-1)

        # Color image (YCbCr)
        cb_data = component_data[1]
        cr_data = component_data[2]

        # Upsample Cb and Cr if necessary
        if cb_data.shape != y_data.shape:
            from scipy.ndimage import zoom

            zoom_factor_h = y_data.shape[0] / cb_data.shape[0]
            zoom_factor_w = y_data.shape[1] / cb_data.shape[1]
            cb_data = zoom(cb_data, (zoom_factor_h, zoom_factor_w), order=1)
            cr_data = zoom(cr_data, (zoom_factor_h, zoom_factor_w), order=1)

        # Crop to actual size
        cb_data = cb_data[: self.height, : self.width]
        cr_data = cr_data[: self.height, : self.width]

        # Convert YCbCr to RGB
        r = y_data + 1.402 * cr_data + 128
        g = y_data - 0.344136 * cb_data - 0.714136 * cr_data + 128
        b = y_data + 1.772 * cb_data + 128

        # Clip and convert to uint8
        r = np.clip(r, 0, 255).astype(np.uint8)
        g = np.clip(g, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)

        return np.stack([r, g, b], axis=-1)


def decode_jpeg(filename: str) -> np.ndarray:
    """
    Convenience function to decode a JPEG file.

    Parameters:
    filename (str): Path to the JPEG file.

    Returns:
    np.ndarray: Decoded RGB image.
    """
    decoder = JPEGDecoder()
    return decoder.decode_file(filename)
