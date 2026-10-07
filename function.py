import numpy as np
import time
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

    # --------------------------------------------------
    # Cache for term_a
    # --------------------------------------------------

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

        candidates = find_candidates_one_a(
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

    # --------------------------------------------------
    # Mask Cache Statistics
    # --------------------------------------------------

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

def find_candidates_correlate2d(
    image,
    target_neighborhood,
    target_mask,
    window_size,
    epsilon=0.1,
    profile=False,
    mask_cache=None,
    cache_stats=None
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

    profile_start = time.perf_counter()

    source = image

    target = target_neighborhood.astype(
        np.float32,
        copy=False
    )

    mask = target_mask.astype(
        np.float32
    )

    prepare_time = time.perf_counter() - profile_start

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

    term_a_time = 0.0
    term_b_time = 0.0

    distance_map = None

    # --------------------------------------------------
    # Cache key for mask
    # --------------------------------------------------

    if mask_cache is not None:

        MAX_MASK_CACHE = 4

        mask_key = target_mask.tobytes()

        if mask_key in mask_cache:

            if cache_stats is not None:
                cache_stats["hits"] += 1

            term_a_total = mask_cache[mask_key]

        else:

            # --------------------------------------------------
            # Calculate term_a for all RGB channels
            # and immediately sum them.
            # --------------------------------------------------

            if cache_stats is not None:
                cache_stats["misses"] += 1

            term_a_total = None

            for channel in range(3):

                source_channel = source[:, :, channel]

                source_squared = (
                    source_channel ** 2
                )

                start = time.perf_counter()

                term_a = correlate2d(
                    source_squared,
                    mask,
                    mode="valid"
                )

                term_a_time += (
                    time.perf_counter() - start
                )

                if term_a_total is None:

                    term_a_total = term_a

                else:

                    term_a_total += term_a

            # --------------------------------------------------
            # Store only ONE matrix
            # --------------------------------------------------

            mask_cache[mask_key] = term_a_total
            if len(mask_cache) > MAX_MASK_CACHE:

                oldest_key = next(
                    iter(mask_cache)
                )

                del mask_cache[oldest_key]

    else:

        term_a_total = None

        for channel in range(3):

            source_channel = source[:, :, channel]

            source_squared = (
                source_channel ** 2
            )

            start = time.perf_counter()

            term_a = correlate2d(
                source_squared,
                mask,
                mode="valid"
            )

            term_a_time += (
                time.perf_counter() - start
            )

            if term_a_total is None:

                term_a_total = term_a

            else:

                term_a_total += term_a

    # --------------------------------------------------
    # Calculate distance
    # --------------------------------------------------

    distance_map = None

    for channel in range(3):

        source_channel = source[:, :, channel]

        target_channel = target[:, :, channel]

    # --------------------------------------------------
    # term_b
    # --------------------------------------------------

        target_masked = (
            target_channel * mask
        )

        start = time.perf_counter()

        term_b = correlate2d(
            source_channel,
            target_masked,
            mode="valid"
        )

        term_b_time += (
            time.perf_counter()
            - start
        )

    # --------------------------------------------------
    # -2 * term_b
    # --------------------------------------------------

        channel_distance = (
            -2.0 * term_b
        )

        if distance_map is None:

            distance_map = channel_distance

        else:

            distance_map += channel_distance


# --------------------------------------------------
# Add term_a_total once
# --------------------------------------------------

    distance_map += term_a_total

# --------------------------------------------------
# Add constant term
# --------------------------------------------------

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

    distance_end_time = time.perf_counter()
    
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

    if profile:
        total_time = (
            time.perf_counter()
            - profile_start
        )

        print("\n--- find_candidates profiling ---")
        print(
            f"Preparation:    {prepare_time:.6f} s"
        )
        print(
            f"correlate2d A:  {term_a_time:.6f} s"
        )
        print(
            f"correlate2d B:  {term_b_time:.6f} s"
        )
        print(
            f"Distance/rest:  "
            f"{distance_end_time - profile_start - prepare_time - term_a_time - term_b_time:.6f} s"
        )
        print(
            f"Total:          {total_time:.6f} s"
        )

    return candidates

def find_candidates_batched(
    image,
    target_neighborhood,
    target_mask,
    window_size,
    epsilon=0.1,
    batch_rows=32
):
    """
    Find candidate patches using batched vectorized
    masked SSD.

    This version avoids correlate2d().

    batch_rows:
        Number of source-patch rows processed at once.
    """

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

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

    source = image.astype(
        np.float32,
        copy=False
    )

    target = target_neighborhood.astype(
        np.float32,
        copy=False
    )

    mask = target_mask.astype(
        np.float32,
        copy=False
    )

    # --------------------------------------------------
    # Source patch dimensions
    # --------------------------------------------------

    height, width, channels = source.shape

    radius = window_size // 2

    patch_height = (
        height - window_size + 1
    )

    patch_width = (
        width - window_size + 1
    )

    # --------------------------------------------------
    # Flatten target and mask
    # --------------------------------------------------

    target_flat = target.reshape(-1)

    mask_flat = np.repeat(
        mask.reshape(-1),
        channels
    )

    # Only known values
    target_masked = (
        target_flat * mask_flat
    )

    known_count = np.sum(mask_flat)

    # --------------------------------------------------
    # Constant:
    #
    # sum(M * T^2)
    # --------------------------------------------------

    constant = np.sum(
        target_flat ** 2
        * mask_flat
    )

    # --------------------------------------------------
    # Create sliding-window view
    #
    # Shape before transpose:
    #
    # (patch_rows,
    #  patch_cols,
    #  channels,
    #  window,
    #  window)
    # --------------------------------------------------

    windows = np.lib.stride_tricks.sliding_window_view(
        source,
        (
            window_size,
            window_size
        ),
        axis=(0, 1)
    )

    # Rearrange:
    #
    # (patch_rows,
    #  patch_cols,
    #  window,
    #  window,
    #  channels)

    windows = windows.transpose(
        0,
        1,
        3,
        4,
        2
    )

    # --------------------------------------------------
    # Store distances
    # --------------------------------------------------

    distances = np.empty(
        (
            patch_height,
            patch_width
        ),
        dtype=np.float32
    )

    # --------------------------------------------------
    # Process source patches in batches
    # --------------------------------------------------

    for row_start in range(
        0,
        patch_height,
        batch_rows
    ):

        row_end = min(
            row_start + batch_rows,
            patch_height
        )

        batch = windows[
            row_start:row_end
        ]

        batch_count = (
            row_end - row_start
        )

        # --------------------------------------------------
        # Flatten patches
        #
        # Shape:
        #
        # (batch_count * patch_width,
        #  window_size * window_size * 3)
        # --------------------------------------------------

        patches = batch.reshape(
            batch_count * patch_width,
            -1
        )

        # --------------------------------------------------
        # Masked SSD
        #
        # D =
        # sum(M*C^2)
        # - 2 sum(M*C*T)
        # + sum(M*T^2)
        # --------------------------------------------------

        term_a = np.einsum(
            "ij,j->i",
            patches * patches,
            mask_flat
        )

        term_b = patches @ target_masked

        batch_distance = (
            term_a
            - 2.0 * term_b
            + constant
        )

        # Mean SSD
        batch_distance /= known_count

        distances[
            row_start:row_end
        ] = batch_distance.reshape(
            batch_count,
            patch_width
        )

    # --------------------------------------------------
    # Numerical safety
    # --------------------------------------------------

    distances = np.maximum(
        distances,
        0
    )

    # --------------------------------------------------
    # Minimum distance
    # --------------------------------------------------

    min_distance = np.min(
        distances
    )

    threshold = (
        min_distance * (1 + epsilon)
    )

    # --------------------------------------------------
    # Candidate positions
    # --------------------------------------------------

    rows, cols = np.where(
        distances <= threshold
    )

    candidates = []

    for row, col in zip(
        rows,
        cols
    ):

        x = int(
            col + radius
        )

        y = int(
            row + radius
        )

        distance = float(
            distances[row, col]
        )

        candidates.append(
            (
                x,
                y,
                distance
            )
        )

    return candidates

def find_candidates_one_a(
    image,
    target_neighborhood,
    target_mask,
    window_size,
    epsilon=0.1
):
    """
    Find candidates using masked SSD.

    Optimization:
        RGB term_a is combined into one
        correlate2d() call.

    Old:
        3 × correlate2d(term_a)
        3 × correlate2d(term_b)

    New:
        1 × correlate2d(term_a)
        3 × correlate2d(term_b)
    """

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

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

    if not np.any(target_mask):
        return []

    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------

    source = image.astype(
        np.float32,
        copy=False
    )

    target = target_neighborhood.astype(
        np.float32,
        copy=False
    )

    mask = target_mask.astype(
        np.float32,
        copy=False
    )

    # --------------------------------------------------
    # Number of known pixels
    # --------------------------------------------------

    known_count = np.sum(mask)

    # --------------------------------------------------
    # Constant term
    #
    # C = sum(M * T^2)
    # --------------------------------------------------

    constant = np.sum(
        target ** 2
        * mask[:, :, None]
    )

    # --------------------------------------------------
    # TERM A
    #
    # A = sum over RGB:
    #
    #     correlate2d(C_R^2, M)
    #   + correlate2d(C_G^2, M)
    #   + correlate2d(C_B^2, M)
    #
    # Because correlation is linear:
    #
    # A =
    # correlate2d(
    #     C_R^2 + C_G^2 + C_B^2,
    #     M
    # )
    #
    # --------------------------------------------------

    source_squared_total = np.sum(
        source ** 2,
        axis=2
    )

    term_a_total = correlate2d(
        source_squared_total,
        mask,
        mode="valid"
    )

    # --------------------------------------------------
    # TERM B
    #
    # B = B_R + B_G + B_B
    #
    # This still needs 3 correlations because
    # target RGB values are different.
    # --------------------------------------------------

    distance_map = None

    for channel in range(3):

        source_channel = (
            source[:, :, channel]
        )

        target_channel = (
            target[:, :, channel]
        )

        target_masked = (
            target_channel * mask
        )

        term_b = correlate2d(
            source_channel,
            target_masked,
            mode="valid"
        )

        channel_distance = (
            -2.0 * term_b
        )

        if distance_map is None:

            distance_map = (
                channel_distance
            )

        else:

            distance_map += (
                channel_distance
            )

    # --------------------------------------------------
    # Complete SSD
    #
    # D = A - 2B + C
    # --------------------------------------------------

    distance_map += term_a_total

    distance_map += constant

    # --------------------------------------------------
    # Mean masked SSD
    # --------------------------------------------------

    distance_map /= known_count

    # --------------------------------------------------
    # Numerical safety
    # --------------------------------------------------

    distance_map = np.maximum(
        distance_map,
        0
    )

    # --------------------------------------------------
    # Candidate threshold
    # --------------------------------------------------

    min_distance = np.min(
        distance_map
    )

    threshold = (
        min_distance
        * (1 + epsilon)
    )

    # --------------------------------------------------
    # Find candidates
    # --------------------------------------------------

    rows, cols = np.where(
        distance_map <= threshold
    )

    radius = window_size // 2

    candidates = []

    for row, col in zip(
        rows,
        cols
    ):

        x = int(
            col + radius
        )

        y = int(
            row + radius
        )

        distance = float(
            distance_map[row, col]
        )

        candidates.append(
            (x, y, distance)
        )

    return candidates