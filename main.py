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

output_size = (100, 100)
window_size = 5

start_time = time.perf_counter()

output, known_mask = synthesize_texture(
    image,
    output_size,
    window_size
)

print("\n--- Texture Synthesis Result ---")

print("Output shape:", output.shape)

print("Expected pixels:", output_size[0] * output_size[1])

print("Known pixels:", np.sum(known_mask))

print("Unknown pixels:", np.sum(~known_mask))

result = Image.fromarray(output)

result.save("texture_result.png")

print("Saved: texture_result.png")

end_time = time.perf_counter()

print(
    f"Synthesis time: "
    f"{end_time - start_time:.2f} seconds"
)