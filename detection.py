# detection.py

import sys
import traceback
from typing import Tuple
from collections import Counter

import numpy as np
from PIL import Image  # Import Image for type hinting
from scipy.signal import find_peaks
# Optional: from scipy.ndimage import gaussian_filter1d # For smoothing profiles

def find_dominant_spacing(profile: np.ndarray, min_spacing: int = 3, prominence_ratio: float = 0.05) -> int:
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
        return 0 # Not enough data to find spacing

    profile_range = np.ptp(profile) # Peak-to-peak range (max - min)
    min_prominence = profile_range * prominence_ratio
    if min_prominence == 0: # Handle flat profiles
        min_prominence = 1 # Set a minimal prominence if range is zero

    # Find peaks
    peaks, _ = find_peaks(profile, distance=min_spacing, prominence=min_prominence)

    if len(peaks) < 2:
        # Need at least two peaks to calculate spacing
        # Commenting out print from library code - maybe raise an error or let caller handle 0
        # print(f"Warning: Found only {len(peaks)} significant peaks. Cannot determine spacing reliably.", file=sys.stderr)
        return 0

    # Calculate distances between consecutive peaks
    spacings = np.diff(peaks)

    if len(spacings) == 0:
        return 0

    # Find the most frequent spacing (mode)
    spacing_counts = Counter(spacings)
    most_common_spacing = spacing_counts.most_common(1)[0][0]

    return int(most_common_spacing)


def detect_grid(image: Image.Image, min_grid_size: int = 4) -> Tuple[int, int]:
    """
    Analyzes the input image to detect the underlying pixel grid dimensions using
    gradient analysis and peak spacing.

    Args:
        image: The PIL Image object to analyze.
        min_grid_size: Minimum expected grid dimension (W or H) used for peak finding.

    Returns:
        A tuple containing the detected grid width and height (grid_w, grid_h).
        Returns (0, 0) if detection fails. Prints errors to stderr.
    """
    print("Analyzing image to detect grid dimensions...") # Keep progress print

    try:
        # 1. Convert to Grayscale and NumPy array
        gray_image = image.convert('L')
        img_array = np.array(gray_image, dtype=np.float32)
        img_h, img_w = img_array.shape

        if img_h < min_grid_size * 2 or img_w < min_grid_size * 2:
             print(f"Error: Image dimensions ({img_w}x{img_h}) too small for analysis with min_grid_size={min_grid_size}.", file=sys.stderr)
             return (0, 0)

        # 2. Calculate Gradients
        gradient_v = np.abs(np.diff(img_array, axis=0))
        gradient_h = np.abs(np.diff(img_array, axis=1))

        # 3. Sum Gradients
        sum_grad_v = np.sum(gradient_v, axis=0)
        sum_grad_h = np.sum(gradient_h, axis=1)

        # Optional Smoothing
        # profile_v = gaussian_filter1d(sum_grad_v, sigma=1.5)
        # profile_h = gaussian_filter1d(sum_grad_h, sigma=1.5)
        profile_v = sum_grad_v # Use raw profile
        profile_h = sum_grad_h # Use raw profile


        # 4. Find Dominant Spacing for Width (from vertical edges / horizontal profile)
        # print("Analyzing vertical grid lines (for width)...") # Less verbose from library
        detected_grid_w = find_dominant_spacing(profile_h, min_spacing=min_grid_size)

        # 5. Find Dominant Spacing for Height (from horizontal edges / vertical profile)
        # print("Analyzing horizontal grid lines (for height)...") # Less verbose from library
        detected_grid_h = find_dominant_spacing(profile_v, min_spacing=min_grid_size)


        if detected_grid_w <= 1 or detected_grid_h <= 1:
             print("Warning: Failed to detect a reliable grid spacing.", file=sys.stderr)
             return (0, 0) # Indicate failure

        # Keep final success print here or move decision to caller
        # print(f"Algorithm finished. Detected dimensions: {detected_grid_w}x{detected_grid_h}")
        return detected_grid_w, detected_grid_h

    except Exception as e:
        print(f"An unexpected error occurred during grid detection algorithm: {e}", file=sys.stderr)
        traceback.print_exc() # Print detailed traceback for debugging
        return (0, 0) # Indicate failure