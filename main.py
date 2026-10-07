import time

from PIL import Image
import numpy as np
from pathlib import Path

from function import synthesize_texture


# ============================================================
# Read image
# ============================================================

IMAGE_PATH = Path(
    "images"
) / "texture 1.png"

image = Image.open(
    IMAGE_PATH
).convert("RGB")

image = np.array(image)


# ============================================================
# Texture synthesis setup
# ============================================================

output_size = (100, 100)
window_size = 5
epsilon = 0.1


# ============================================================
# Run Texture Synthesis
# ============================================================

print("\n--- Texture Synthesis ---")
print("Input shape:", image.shape)
print("Output size:", output_size)
print("Window size:", window_size)

start_time = time.perf_counter()

output, known_mask = synthesize_texture(
    image,
    output_size,
    window_size,
    epsilon
)

end_time = time.perf_counter()


# ============================================================
# Results
# ============================================================

synthesis_time = (
    end_time - start_time
)

print("\n--- Texture Synthesis Result ---")

print(
    "Output shape:",
    output.shape
)

expected_pixels = (
    output_size[0] * output_size[1]
)

known_pixels = int(
    np.sum(known_mask)
)

unknown_pixels = (
    expected_pixels - known_pixels
)

print(
    "Expected pixels:",
    expected_pixels
)

print(
    "Known pixels:",
    known_pixels
)

print(
    "Unknown pixels:",
    unknown_pixels
)

print(
    f"Synthesis time: "
    f"{synthesis_time:.2f} seconds"
)


# ============================================================
# Save result
# ============================================================

output_path = Path(
    f"texture_result_{output_size[0]}x{output_size[1]}.png"
)

Image.fromarray(output).save(
    output_path
)

print(
    "Saved:",
    output_path
)