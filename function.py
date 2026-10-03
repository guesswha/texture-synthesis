import numpy as np
from scipy.signal import correlate2d

def get_neighborhood(image, x, y, window_size):
    radius = window_size // 2

    # --------------------------------------------------
    # Padding
    # --------------------------------------------------

    if image.ndim == 3:
        # RGB image
        padded = np.pad(
            image,
            (
                (radius, radius),
                (radius, radius),
                (0, 0)
            ),
            mode="edge"
        )

    elif image.ndim == 2:
        # Boolean mask
        padded = np.pad(
            image,
            (
                (radius, radius),
                (radius, radius)
            ),
            mode="constant",
            constant_values=False
        )

    else:
        raise ValueError(
            "image must be 2D or 3D."
        )

    # Shift coordinate because of padding
    x_padded = x + radius
    y_padded = y + radius

    # --------------------------------------------------
    # Extract full neighborhood
    # --------------------------------------------------

    neighborhood = padded[
        y_padded - radius:y_padded + radius + 1,
        x_padded - radius:x_padded + radius + 1
    ]

    return neighborhood

def calculate_distance(A, B):
    difference = A.astype(float) - B.astype(float)

    squared_difference = difference ** 2

    distance = np.sum(squared_difference)

    return distance

def synthesize_texture(image, output_size, window_size, epsilon=0.1):
    """
    Synthesize a new texture using Efros-Leung
    style texture synthesis.
    """

    if window_size % 2 == 0:
        raise ValueError(
            "window_size must be odd."
        )

    height, width = output_size

    if height < window_size or width < window_size:
        raise ValueError(
            "output_size must be at least window_size."
        )

    # --------------------------------------------------
    # Create output
    # --------------------------------------------------

    output = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    # False = unknown
    # True  = known
    known_mask = np.zeros(
        (height, width),
        dtype=bool
    )

    # --------------------------------------------------
    # Create seed
    # --------------------------------------------------

    source_x = image.shape[1] // 2
    source_y = image.shape[0] // 2

    seed = get_neighborhood(
        image,
        source_x,
        source_y,
        window_size
    )

    # --------------------------------------------------
    # Put seed in center
    # --------------------------------------------------

    center_x = width // 2
    center_y = height // 2

    radius = window_size // 2

    # --------------------------------------------------
    # Precompute source texture windows
    # --------------------------------------------------

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

    frontier_set = set()

    for yy in range(y_start, y_end):
        for xx in range(x_start, x_end):

            neighbors = [
                (xx - 1, yy),
                (xx + 1, yy),
                (xx, yy - 1),
                (xx, yy + 1)
            ]

            for nx, ny in neighbors:

                if (
                    0 <= nx < width
                    and 0 <= ny < height
                    and not known_mask[ny, nx]
                ):
                    frontier_set.add((nx, ny))

    # --------------------------------------------------
    # Synthesis loop
    # --------------------------------------------------
    
    image_float = image.astype(
        np.float32,
        copy=False
    )

    while frontier_set:

        # --------------------------------------------------
        # Choose a frontier pixel
        # --------------------------------------------------

        x, y = frontier_set.pop()

        # --------------------------------------------------
        # Skip if already known
        # --------------------------------------------------

        if known_mask[y, x]:
            continue

        # --------------------------------------------------
        # Target neighborhood
        # --------------------------------------------------

        target_neighborhood = get_neighborhood(
            output,
            x,
            y,
            window_size
        )

        # --------------------------------------------------
        # Target mask
        # --------------------------------------------------

        target_mask = get_neighborhood(
            known_mask,
            x,
            y,
            window_size
        )

        # --------------------------------------------------
        # Find candidates
        # --------------------------------------------------

        candidates = find_candidates(
            image_float,
            target_neighborhood,
            target_mask,
            window_size,
            epsilon
        )

        if not candidates:
            continue

        # --------------------------------------------------
        # Choose best candidate
        # --------------------------------------------------

        candidate_index = np.random.randint(
            len(candidates)
            )

        candidate = candidates[candidate_index]

        candidate_x = candidate[0]
        candidate_y = candidate[1]

        # --------------------------------------------------
        # Copy pixel
        # --------------------------------------------------

        output[y, x] = image[
            candidate_y,
            candidate_x
        ]

        # --------------------------------------------------
        # Mark known
        # --------------------------------------------------

        known_mask[y, x] = True

        # --------------------------------------------------
        # Add new frontier pixels
        # --------------------------------------------------

        neighbors = [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1)
        ]

        for nx, ny in neighbors:

            if (
                0 <= nx < width
                and 0 <= ny < height
                and not known_mask[ny, nx]
            ):
                frontier_set.add((nx, ny))

    return output, known_mask

def find_frontier_pixels(known_mask):
    """
    Find unknown pixels that are adjacent to known pixels.

    True  = known
    False = unknown

    Returns:
        list of (x, y)
    """

    height, width = known_mask.shape

    frontier = []

    for y in range(height):
        for x in range(width):

            # Pixel that is known is not frontier
            if known_mask[y, x]:
                continue

            # Check 4 directions
            neighbors = [
                (x - 1, y),  # left
                (x + 1, y),  # right
                (x, y - 1),  # up
                (x, y + 1)   # down
            ]

            for nx, ny in neighbors:

                # Check if the coordinate is in the image
                if (
                    0 <= nx < width
                    and 0 <= ny < height
                ):

                    # If there are neighborhoods
                    if known_mask[ny, nx]:
                        frontier.append((x, y))
                        break

    return frontier

def calculate_masked_distance(
    target,
    candidate,
    mask
):
    """
    Calculate SSD using only known pixels.

    target:
        Neighborhood from output image.

    candidate:
        Neighborhood from source texture.

    mask:
        True  -> use this pixel
        False -> ignore this pixel.
    """

    target = target.astype(np.float32)
    candidate = candidate.astype(np.float32)

    # Difference between target and candidate
    difference = target - candidate

    # Squared difference
    squared_difference = difference ** 2

    # For RGB, each pixel has 3 channels.
    # Sum RGB differences first.
    pixel_error = np.sum(
        squared_difference,
        axis=2
    )

    # Only keep known pixels
    masked_error = pixel_error[mask]

    # Avoid division by zero
    if masked_error.size == 0:
        return np.inf

    # Mean SSD over known pixels
    return np.mean(masked_error)

def find_candidates(
    image,
    target_neighborhood,
    target_mask,
    window_size,
    epsilon=0.1
):
    """
    Find candidate patches using masked SSD.

    The masked SSD is calculated using 2D correlation
    instead of comparing every patch with Python loops.
    """

    if window_size % 2 == 0:
        raise ValueError(
            "window_size must be odd."
        )

    if target_neighborhood.shape[:2] != (
        window_size,
        window_size
    ):
        raise ValueError(
            "target_neighborhood has wrong shape."
        )

    if target_mask.shape != (
        window_size,
        window_size
    ):
        raise ValueError(
            "target_mask has wrong shape."
        )

    # --------------------------------------------------
    # No known pixels
    # --------------------------------------------------

    if not np.any(target_mask):
        return []

    # --------------------------------------------------
    # Convert to float32
    # --------------------------------------------------

    source = image

    target = target_neighborhood.astype(
        np.float32,
        copy=False
    )

    mask = target_mask.astype(
        np.float32
    )

    # --------------------------------------------------
    # Number of known pixels
    # --------------------------------------------------

    known_count = np.sum(mask)

    # --------------------------------------------------
    # Constant term:
    #
    # sum(mask * target^2)
    # --------------------------------------------------

    target_squared = (
        target ** 2
    )

    constant = np.sum(
        target_squared * mask[:, :, None]
    )

    # --------------------------------------------------
    # Distance map
    # --------------------------------------------------

    distance_map = None

    for channel in range(3):

        source_channel = source[:, :, channel]

        target_channel = target[:, :, channel]

        # --------------------------------------------------
        # sum(mask * source^2)
        # --------------------------------------------------

        source_squared = (
            source_channel ** 2
        )

        term_a = correlate2d(
            source_squared,
            mask,
            mode="valid"
        )

        # --------------------------------------------------
        # sum(mask * source * target)
        # --------------------------------------------------

        target_masked = (
            target_channel * mask
        )

        term_b = correlate2d(
            source_channel,
            target_masked,
            mode="valid"
        )

        # --------------------------------------------------
        # SSD:
        #
        # A - 2B + C
        # --------------------------------------------------

        channel_distance = (
            term_a
            - 2.0 * term_b
        )

        if distance_map is None:
            distance_map = channel_distance
        else:
            distance_map += channel_distance

    distance_map += constant

    # --------------------------------------------------
    # Mean masked SSD
    # --------------------------------------------------

    distance_map /= known_count

    # Numerical precision can sometimes produce
    # tiny negative values.
    distance_map = np.maximum(
        distance_map,
        0
    )

    # --------------------------------------------------
    # Find minimum distance
    # --------------------------------------------------

    min_distance = np.min(
        distance_map
    )

    threshold = (
        min_distance * (1 + epsilon)
    )

    # --------------------------------------------------
    # Find candidates
    # --------------------------------------------------

    rows, cols = np.where(
        distance_map <= threshold
    )

    radius = window_size // 2

    candidates = []

    for row, col in zip(rows, cols):

        x = int(col + radius)
        y = int(row + radius)

        distance = float(
            distance_map[row, col]
        )

        candidates.append(
            (x, y, distance)
        )

    return candidates

