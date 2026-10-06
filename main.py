import time
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
# Test synthesize_texture
# =========================

# ============================================================
# Profile one find_candidates()
# ============================================================

output_size = (100, 100)
window_size = 5

# ------------------------------------------------------------
# Create a small output
# ------------------------------------------------------------

output = np.zeros(
    (100, 100, 3),
    dtype=np.uint8
)

known_mask = np.zeros(
    (100, 100),
    dtype=bool
)

# ------------------------------------------------------------
# Create seed
# ------------------------------------------------------------

source_x = image.shape[1] // 2
source_y = image.shape[0] // 2

seed = get_neighborhood(
    image,
    source_x,
    source_y,
    window_size
)

radius = window_size // 2

center_x = 50
center_y = 50

y_start = center_y - radius
y_end = center_y + radius + 1

x_start = center_x - radius
x_end = center_x + radius + 1

output[
    y_start:y_end,
    x_start:x_end
] = seed

known_mask[
    y_start:y_end,
    x_start:x_end
] = True

# ------------------------------------------------------------
# Find one frontier pixel
# ------------------------------------------------------------

frontier = find_frontier_pixels(
    known_mask
)

x, y = frontier[0]

print("\n--- Profiling one frontier pixel ---")
print("Target pixel:", (x, y))

# ------------------------------------------------------------
# Get target neighborhood
# ------------------------------------------------------------

target_neighborhood = get_neighborhood(
    output,
    x,
    y,
    window_size
)

target_mask = get_neighborhood(
    known_mask,
    x,
    y,
    window_size
)

# ------------------------------------------------------------
# Prepare source once
# ------------------------------------------------------------

image_float = image.astype(
    np.float32,
    copy=False
)

# ------------------------------------------------------------
# Profile find_candidates
# ------------------------------------------------------------

start = time.perf_counter()

candidates = find_candidates(
    image_float,
    target_neighborhood,
    target_mask,
    window_size,
    epsilon=0.1,
    profile=True
)

total = time.perf_counter() - start

print(
    f"\nExternal timing: {total:.6f} s"
)

print(
    "Number of candidates:",
    len(candidates)
)