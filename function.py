import numpy as np

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

    image_float = image.astype(
        np.float32,
        copy=False
    )

    source_windows = np.lib.stride_tricks.sliding_window_view(
        image_float,
        (window_size, window_size),
        axis=(0, 1)
    )

    source_windows = source_windows.transpose(
        0, 1, 3, 4, 2
    )

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

    total_pixels = height * width

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
            image,
            target_neighborhood,
            target_mask,
            window_size,
            epsilon,
            windows=source_windows
        )

        if not candidates:
            continue

        # --------------------------------------------------
        # Choose best candidate
        # --------------------------------------------------

        candidate = min(
            candidates,
            key=lambda item: item[2]
        )

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
    epsilon=0.1,
    chunk_rows=32,
    windows=None
):
    """
    Find candidate patches using masked distance.

    Source image windows are reused when `windows`
    is provided.
    """

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

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

    if not np.any(target_mask):
        return []

    height, width = image.shape[:2]
    radius = window_size // 2

    candidate_height = height - window_size + 1
    candidate_width = width - window_size + 1

    # --------------------------------------------------
    # Create source windows only if not already provided
    # --------------------------------------------------

    if windows is None:

        image_float = image.astype(
            np.float32,
            copy=False
        )

        windows = np.lib.stride_tricks.sliding_window_view(
            image_float,
            (window_size, window_size),
            axis=(0, 1)
        )

        windows = windows.transpose(
            0, 1, 3, 4, 2
        )

    # --------------------------------------------------
    # Target
    # --------------------------------------------------

    target_float = target_neighborhood.astype(
        np.float32,
        copy=False
    )

    target_known = target_float[target_mask]

    # --------------------------------------------------
    # Store distances
    # --------------------------------------------------

    distances = np.empty(
        (candidate_height, candidate_width),
        dtype=np.float32
    )

    # --------------------------------------------------
    # Calculate distances
    # --------------------------------------------------

    for row_start in range(
        0,
        candidate_height,
        chunk_rows
    ):

        row_end = min(
            row_start + chunk_rows,
            candidate_height
        )

        chunk = windows[
            row_start:row_end
        ]

        chunk_known = chunk[
            :, :, target_mask, :
        ]

        difference = (
            chunk_known
            - target_known
        )

        pixel_error = np.sum(
            difference * difference,
            axis=3
        )

        distances[
            row_start:row_end
        ] = np.mean(
            pixel_error,
            axis=2
        )

    # --------------------------------------------------
    # Find threshold
    # --------------------------------------------------

    min_distance = np.min(distances)

    threshold = (
        min_distance * (1 + epsilon)
    )

    # --------------------------------------------------
    # Find candidates
    # --------------------------------------------------

    rows, cols = np.where(
        distances <= threshold
    )

    # --------------------------------------------------
    # Convert to coordinates
    # --------------------------------------------------

    candidates = []

    for row, col in zip(rows, cols):

        x = int(col + radius)
        y = int(row + radius)

        candidates.append(
            (
                x,
                y,
                float(distances[row, col])
            )
        )

    return candidates

