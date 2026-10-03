from PIL import Image
import numpy as np
from pathlib import Path
from function import get_neighborhood, calculate_distance, synthesize_texture, find_frontier_pixels, calculate_masked_distance, find_candidates

# =========================
# Read image
# =========================

IMAGE_PATH = Path("images") / "texture 1.png"
image = Image.open(IMAGE_PATH).convert("RGB")
image = np.array(image)

# =========================
# Test get_neighborhood
# =========================

neighborhood = get_neighborhood(image, 2, 2, 3)

# =========================
# Test calculate_distance
# =========================

x = 2
y = 2

A = get_neighborhood(image, x, y, 5)

B = get_neighborhood(image, x + 10, y + 10, 5)

distance_AB = calculate_distance(A, B)

# =========================
# Test synthesize_texture
# =========================

output_size = (30, 30)
window_size = 5

output, known_mask = synthesize_texture(
    image,
    output_size,
    window_size
)

# =========================
# Test find_frontier_pixels
# =========================

frontier = find_frontier_pixels(known_mask)

# =========================
# Test calculate_masked_distance
# =========================

target = np.array([
    [[10, 10, 10], [20, 20, 20], [30, 30, 30]],
    [[40, 40, 40], [0, 0, 0],    [60, 60, 60]],
    [[70, 70, 70], [80, 80, 80], [90, 90, 90]]
])

candidate = np.array([
    [[11, 11, 11], [19, 19, 19], [31, 31, 31]],
    [[39, 39, 39], [50, 50, 50], [61, 61, 61]],
    [[72, 72, 72], [79, 79, 79], [91, 91, 91]]
])

mask = np.array([
    [True, True, True],
    [True, False, True],
    [True, True, True]
])

distance = calculate_masked_distance(
    target,
    candidate,
    mask
)

print("\n--- Texture Synthesis ---")

print(
    "Known pixels:",
    np.sum(known_mask)
)

print(
    "Unknown pixels:",
    np.sum(~known_mask)
)

result = Image.fromarray(output)

result.save("texture_result.png")

print("Saved: texture_result.png")