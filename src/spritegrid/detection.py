# detection.py

import sys
import traceback
from typing import Tuple
from collections import Counter

import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def find_dominant_spacing(
    profile: np.ndarray, min_spacing: int = 3, prominence_ratio: float = 0.05
) -> Tuple[int, float]:
    """
    Analyzes a 1D profile to find the most frequent spacing between significant peaks.

    Args:
        profile: The 1D NumPy array to analyze.
        min_spacing: Minimum distance between detected peaks.
        prominence_ratio: Minimum prominence of peaks relative to profile range.

    Returns:
        Tuple of (spacing, confidence) where:
        - spacing: The most frequent spacing (mode), or 0 if detection fails
        - confidence: Float 0-1 indicating how consistent the spacing is
    """
    if profile is None or len(profile) < min_spacing * 2:
        return 0, 0.0

    profile_range = np.ptp(profile)
    min_prominence = profile_range * prominence_ratio
    if min_prominence == 0:
        profile_mean = np.mean(profile)
        min_prominence = max(1.0, profile_mean * 0.01) if profile_mean > 0 else 1.0

    # Find peaks
    peaks, properties = find_peaks(
        profile, distance=min_spacing, prominence=min_prominence
    )

    if len(peaks) < 2:
        return 0, 0.0

    # Calculate distances between consecutive peaks
    spacings = np.diff(peaks)

    if len(spacings) == 0:
        return 0, 0.0

    # Find the most frequent spacing (mode)
    spacing_counts = Counter(spacings)

    if not spacing_counts:
        return 0, 0.0

    most_common_spacing, mode_count = spacing_counts.most_common(1)[0]

    # Calculate confidence: what fraction of spacings match the mode (within tolerance)
    tolerance = max(1, most_common_spacing // 4)  # Allow 25% deviation
    matching_count = sum(
        count for spacing, count in spacing_counts.items()
        if abs(spacing - most_common_spacing) <= tolerance
    )
    confidence = matching_count / len(spacings) if spacings.size > 0 else 0.0

    # Also factor in: did we find enough peaks for the image size?
    expected_peaks = len(profile) / most_common_spacing if most_common_spacing > 0 else 0
    if expected_peaks > 0:
        peak_coverage = len(peaks) / expected_peaks
        # Combine spacing consistency with peak coverage
        confidence = (confidence + min(1.0, peak_coverage)) / 2

    return int(most_common_spacing), confidence


def find_grid_offset(profile: np.ndarray, spacing: int) -> int:
    """Find the phase offset of a repeating grid pattern in a 1D gradient profile.

    Scans offset values 0..(spacing-1) and returns the one that maximises the
    sum of profile values at positions offset, offset+spacing, offset+2*spacing, ...

    Args:
        profile: 1D gradient profile.
        spacing: Known grid cell size in pixels.

    Returns:
        Best offset (0-indexed pixel position where the first grid line falls).
    """
    if spacing <= 0 or len(profile) < spacing:
        return 0

    best_offset = 0
    best_score = -1.0

    for offset in range(spacing):
        indices = np.arange(offset, len(profile), spacing)
        score = float(profile[indices].sum())
        if score > best_score:
            best_score = score
            best_offset = offset

    return best_offset


def detect_grid(
    image: Image.Image,
    min_grid_size: int = 4,
    smoothing_sigma: float = 1.0,
    min_confidence: float = 0.4,
) -> Tuple[int, int]:
    """
    Analyzes the input image to detect the underlying pixel grid dimensions.

    Uses gradient analysis and peak spacing detection. Returns (0, 0) if no
    reliable grid is detected (e.g., image is already clean pixel art).

    Args:
        image: The PIL Image object to analyze.
        min_grid_size: Minimum expected grid dimension for peak finding.
        smoothing_sigma: Gaussian smoothing sigma (0 to disable).
        min_confidence: Minimum confidence threshold (0-1) to accept detection.

    Returns:
        Tuple (grid_w, grid_h) or (0, 0) if detection fails or confidence is low.
    """
    result = detect_grid_with_offset(image, min_grid_size, smoothing_sigma, min_confidence)
    return result[0], result[1]


def detect_grid_with_offset(
    image: Image.Image,
    min_grid_size: int = 4,
    smoothing_sigma: float = 1.0,
    min_confidence: float = 0.4,
) -> Tuple[int, int, int, int]:
    """
    Analyzes the input image to detect grid dimensions *and* the phase offset.

    Returns (grid_w, grid_h, offset_x, offset_y).  The offset tells you the
    x/y pixel position where the first grid column/row boundary falls.
    Use the half-cell shift (offset_x + grid_w//2) as the first sample centre.

    Returns (0, 0, 0, 0) if no reliable grid is detected.

    Args:
        image: The PIL Image object to analyze.
        min_grid_size: Minimum expected grid dimension for peak finding.
        smoothing_sigma: Gaussian smoothing sigma (0 to disable).
        min_confidence: Minimum confidence threshold (0-1) to accept detection.

    Returns:
        Tuple (grid_w, grid_h, offset_x, offset_y) or (0, 0, 0, 0).
    """
    try:
        # Convert to grayscale
        gray_image = (
            image.split()[0] if image.mode in ("RGBA", "LA") else image.convert("L")
        )
        img_array = np.array(gray_image, dtype=np.float32)
        img_h, img_w = img_array.shape

        actual_min_spacing = max(1, min_grid_size)

        # Image too small for analysis
        if img_h < actual_min_spacing * 2 or img_w < actual_min_spacing * 2:
            return (0, 0)

        # Calculate gradients
        gradient_h = np.abs(np.diff(img_array, axis=1, append=img_array[:, -1:]))
        gradient_v = np.abs(np.diff(img_array, axis=0, append=img_array[-1:, :]))

        # Sum to create 1D profiles
        profile_h = np.sum(gradient_h, axis=0)
        profile_v = np.sum(gradient_v, axis=1)

        # Optional smoothing
        if smoothing_sigma and smoothing_sigma > 0:
            profile_v = gaussian_filter1d(profile_v, sigma=smoothing_sigma)
            profile_h = gaussian_filter1d(profile_h, sigma=smoothing_sigma)

        # Find dominant spacing with confidence
        grid_h, conf_h = find_dominant_spacing(profile_v, min_spacing=actual_min_spacing)
        grid_w, conf_w = find_dominant_spacing(profile_h, min_spacing=actual_min_spacing)

        # Check if detection failed
        if grid_w <= 0 or grid_h <= 0:
            return (0, 0, 0, 0)

        # Check confidence threshold
        avg_confidence = (conf_h + conf_w) / 2
        if avg_confidence < min_confidence:
            print(
                f"Grid detection confidence too low ({avg_confidence:.2f} < {min_confidence}). "
                "Image may already be clean pixel art.",
                file=sys.stderr,
            )
            return (0, 0, 0, 0)

        # Check grid aspect ratio - genuine pixel art grids are roughly square
        grid_ratio = grid_w / grid_h
        if grid_ratio < 0.5 or grid_ratio > 2.0:
            print(
                f"Detected grid {grid_w}x{grid_h} has inconsistent aspect ratio ({grid_ratio:.2f}). "
                "Image may already be clean pixel art.",
                file=sys.stderr,
            )
            return (0, 0, 0, 0)

        # Detect grid phase offset using the already-computed gradient profiles
        offset_x = find_grid_offset(profile_h, grid_w)
        offset_y = find_grid_offset(profile_v, grid_h)

        return grid_w, grid_h, offset_x, offset_y

    except ImportError:
        print(
            "Error: SciPy or NumPy not found. Install with: pip install numpy scipy pillow",
            file=sys.stderr,
        )
        return (0, 0, 0, 0)
    except Exception as e:
        print(f"Grid detection error: {e}", file=sys.stderr)
        traceback.print_exc()
        return (0, 0, 0, 0)
