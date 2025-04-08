# detection.py

import sys
import traceback
from typing import Tuple
from collections import Counter

import numpy as np
from PIL import Image  # Import Image for type hinting
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d  # Import for smoothing profiles


def find_dominant_spacing(
    profile: np.ndarray, min_spacing: int = 3, prominence_ratio: float = 0.05
) -> int:
    """
    Analyzes a 1D profile (e.g., summed gradients) to find the most frequent spacing
    between significant peaks.

    Args:
        profile: The 1D NumPy array to analyze.
        min_spacing: Minimum distance (in indices) between detected peaks.
        prominence_ratio: Minimum prominence of peaks relative to profile range.

    Returns:
        The most frequent spacing (mode), or 0 if insufficient peaks are found.
    """
    if profile is None or len(profile) < min_spacing * 2:
        # print("Profile too short for analysis.", file=sys.stderr) # Debugging print
        return 0  # Not enough data to find spacing

    profile_range = np.ptp(profile)  # Peak-to-peak range (max - min)
    min_prominence = profile_range * prominence_ratio
    if min_prominence == 0:  # Handle flat profiles or profiles with very low range
        # If the profile is flat, any small noise could become a peak.
        # If the range is very small, the prominence threshold might be too low.
        # Consider using a small absolute minimum prominence, or returning 0.
        # Let's use a small fraction of the mean as a fallback, or 1 if mean is 0.
        profile_mean = np.mean(profile)
        min_prominence = max(1.0, profile_mean * 0.01) if profile_mean > 0 else 1.0
        # print(f"Adjusted min_prominence due to low range: {min_prominence}", file=sys.stderr) # Debugging print

    # Find peaks
    # Note: 'distance' is the minimum separation, 'prominence' is the minimum vertical height relative to neighbors
    peaks, properties = find_peaks(
        profile, distance=min_spacing, prominence=min_prominence
    )
    # print(f"Profile length: {len(profile)}, Found {len(peaks)} peaks at indices: {peaks} with prominences: {properties.get('prominences')}", file=sys.stderr) # Debugging print

    if len(peaks) < 2:
        # Need at least two peaks to calculate spacing
        # print(f"Warning: Found only {len(peaks)} significant peaks. Cannot determine spacing reliably.", file=sys.stderr)
        return 0

    # Calculate distances between consecutive peaks
    spacings = np.diff(peaks)

    if len(spacings) == 0:
        # print("No spacings calculated.", file=sys.stderr) # Debugging print
        return 0

    # Find the most frequent spacing (mode)
    spacing_counts = Counter(spacings)
    # print(f"Spacing counts: {spacing_counts}", file=sys.stderr) # Debugging print

    if not spacing_counts:
        # print("Spacing counts are empty.", file=sys.stderr) # Debugging print
        return 0

    # Get the most common spacing. If there's a tie, most_common(1) returns one of them.
    most_common_spacing, count = spacing_counts.most_common(1)[0]

    # Optional: Add a check for confidence? E.g., if the mode count is very low?
    # print(f"Most common spacing: {most_common_spacing} (count: {count})", file=sys.stderr) # Debugging print

    return int(most_common_spacing)


def detect_grid(
    image: Image.Image, min_grid_size: int = 4, smoothing_sigma: float = 1.0
) -> Tuple[int, int]:
    """
    Analyzes the input image to detect the underlying pixel grid dimensions using
    gradient analysis and peak spacing with optional smoothing.

    Args:
        image: The PIL Image object to analyze.
        min_grid_size: Minimum expected grid dimension (W or H) used for peak finding distance.
                       Should generally be >= 1.
        smoothing_sigma: Standard deviation for Gaussian kernel used to smooth gradient profiles.
                         Set to 0 or None to disable smoothing.

    Returns:
        A tuple containing the detected grid width and height (grid_w, grid_h).
        Returns (0, 0) if detection fails. Prints errors to stderr.
    """
    # print(f"Analyzing image to detect grid dimensions (min_grid_size={min_grid_size}, smoothing_sigma={smoothing_sigma})...") # Debugging print

    try:
        # 1. Convert to Grayscale and NumPy array
        # Use 'LA' to handle transparency if present, then select L channel
        gray_image = (
            image.split()[0] if image.mode in ("RGBA", "LA") else image.convert("L")
        )
        img_array = np.array(gray_image, dtype=np.float32)
        img_h, img_w = img_array.shape

        # Ensure min_grid_size is at least 1 for peak distance
        actual_min_spacing = max(1, min_grid_size)

        # Check if image is large enough for analysis based on minimum spacing needed for peak finding
        # We need at least 2*actual_min_spacing length in a profile to find a spacing.
        if img_h < actual_min_spacing * 2 or img_w < actual_min_spacing * 2:
            print(
                f"Error: Image dimensions ({img_w}x{img_h}) too small for analysis with min_spacing={actual_min_spacing}.",
                file=sys.stderr,
            )
            return (0, 0)

        # 2. Calculate Gradients (Absolute difference between adjacent pixels)
        # Vertical edges (changes along rows) -> contribute to horizontal profile
        gradient_h_lines = np.abs(
            np.diff(img_array, axis=1, append=img_array[:, -1:])
        )  # Keep same width
        # Horizontal edges (changes along columns) -> contribute to vertical profile
        gradient_v_lines = np.abs(
            np.diff(img_array, axis=0, append=img_array[-1:, :])
        )  # Keep same height

        # 3. Sum Gradients to create 1D profiles
        # Sum vertical edge gradients along columns to get horizontal profile
        profile_h = np.sum(gradient_h_lines, axis=0)
        # Sum horizontal edge gradients along rows to get vertical profile
        profile_v = np.sum(gradient_v_lines, axis=1)

        # 4. Optional Smoothing
        if smoothing_sigma and smoothing_sigma > 0:
            # print(f"Applying Gaussian smoothing with sigma={smoothing_sigma}...") # Debugging print
            profile_v_smooth = gaussian_filter1d(profile_v, sigma=smoothing_sigma)
            profile_h_smooth = gaussian_filter1d(profile_h, sigma=smoothing_sigma)
        else:
            # print("Skipping smoothing.") # Debugging print
            profile_v_smooth = profile_v  # Use raw profile
            profile_h_smooth = profile_h  # Use raw profile

        # 5. Find Dominant Spacing for Width (analyzing vertical profile for horizontal lines)
        # print("Analyzing vertical profile (for height)...") # Debugging print
        detected_grid_h = find_dominant_spacing(
            profile_v_smooth, min_spacing=actual_min_spacing
        )

        # 6. Find Dominant Spacing for Height (analyzing horizontal profile for vertical lines)
        # print("Analyzing horizontal profile (for width)...") # Debugging print
        detected_grid_w = find_dominant_spacing(
            profile_h_smooth, min_spacing=actual_min_spacing
        )

        if detected_grid_w <= 0 or detected_grid_h <= 0:
            # find_dominant_spacing returns 0 on failure/insufficient peaks
            print(
                f"Warning: Failed to detect a reliable grid spacing (w={detected_grid_w}, h={detected_grid_h}).",
                file=sys.stderr,
            )
            return (0, 0)  # Indicate failure

        # print(f"Algorithm finished. Detected dimensions: {detected_grid_w}x{detected_grid_h}") # Debugging print
        return detected_grid_w, detected_grid_h

    except ImportError:
        print(
            "Error: SciPy or NumPy library not found. Please install them (`pip install numpy scipy pillow`).",
            file=sys.stderr,
        )
        return (0, 0)
    except Exception as e:
        print(
            f"An unexpected error occurred during grid detection algorithm: {e}",
            file=sys.stderr,
        )
        traceback.print_exc()  # Print detailed traceback for debugging
        return (0, 0)  # Indicate failure
