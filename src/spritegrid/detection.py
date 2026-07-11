# detection.py

import sys
import traceback
from dataclasses import dataclass, replace
from typing import Sequence, Tuple
from collections import Counter

import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class GradientAnalysis:
    """Canonical image signals consumed by grid detection."""

    grayscale: np.ndarray
    gradient_h: np.ndarray
    gradient_v: np.ndarray
    raw_profile_h: np.ndarray
    raw_profile_v: np.ndarray
    profile_h: np.ndarray
    profile_v: np.ndarray
    smoothing_sigma: float
    grayscale_method: str


@dataclass(frozen=True)
class SpacingAnalysis:
    """Every value used to choose a repeating spacing on one axis."""

    profile: np.ndarray
    min_spacing: int
    prominence_ratio: float
    profile_range: float
    min_prominence: float
    peaks: np.ndarray
    prominences: np.ndarray
    spacings: np.ndarray
    spacing_counts: tuple[tuple[int, int], ...]
    selected_spacing: int
    mode_count: int
    tolerance: int
    matching_count: int
    spacing_consistency: float
    expected_peaks: float
    peak_coverage: float
    confidence: float


@dataclass(frozen=True)
class OffsetAnalysis:
    """Canonical phase scan for one axis."""

    profile: np.ndarray
    spacing: int
    scores: np.ndarray
    selected_offset: int
    selected_score: float


@dataclass(frozen=True)
class GridDetectionAnalysis:
    """Complete, renderer-independent account of one grid decision."""

    signals: GradientAnalysis | None
    horizontal: SpacingAnalysis
    vertical: SpacingAnalysis
    offset_x: OffsetAnalysis
    offset_y: OffsetAnalysis
    min_grid_size: int
    min_confidence: float
    average_confidence: float
    aspect_ratio: float
    rejection_reason: str | None
    result: Tuple[int, int, int, int]


def _empty_spacing(profile: np.ndarray, min_spacing: int, prominence_ratio: float,
                   profile_range: float = 0.0, min_prominence: float = 0.0) -> SpacingAnalysis:
    return SpacingAnalysis(
        profile=np.asarray(profile), min_spacing=min_spacing,
        prominence_ratio=prominence_ratio, profile_range=profile_range,
        min_prominence=min_prominence, peaks=np.array([], dtype=int),
        prominences=np.array([], dtype=float), spacings=np.array([], dtype=int),
        spacing_counts=(), selected_spacing=0, mode_count=0, tolerance=0,
        matching_count=0, spacing_consistency=0.0, expected_peaks=0.0,
        peak_coverage=0.0, confidence=0.0,
    )


def analyze_dominant_spacing(
    profile: np.ndarray, min_spacing: int = 3, prominence_ratio: float = 0.05
) -> SpacingAnalysis:
    """Return the exact intermediates used by :func:`find_dominant_spacing`."""
    profile = np.asarray(profile) if profile is not None else np.array([])
    if len(profile) < min_spacing * 2:
        return _empty_spacing(profile, min_spacing, prominence_ratio)

    profile_range = float(np.ptp(profile))
    min_prominence = profile_range * prominence_ratio
    if min_prominence == 0:
        profile_mean = float(np.mean(profile))
        min_prominence = max(1.0, profile_mean * 0.01) if profile_mean > 0 else 1.0

    peaks, properties = find_peaks(
        profile, distance=min_spacing, prominence=min_prominence
    )
    prominences = np.asarray(properties.get("prominences", []), dtype=float)
    if len(peaks) < 2:
        result = _empty_spacing(
            profile, min_spacing, prominence_ratio, profile_range, min_prominence
        )
        return replace(result, peaks=peaks, prominences=prominences)

    spacings = np.diff(peaks)
    spacing_counts = Counter(int(value) for value in spacings)
    most_common_spacing, mode_count = spacing_counts.most_common(1)[0]
    tolerance = max(1, most_common_spacing // 4)
    matching_count = sum(
        count for spacing, count in spacing_counts.items()
        if abs(spacing - most_common_spacing) <= tolerance
    )
    consistency = matching_count / len(spacings)
    expected_peaks = len(profile) / most_common_spacing
    peak_coverage = len(peaks) / expected_peaks
    confidence = (consistency + min(1.0, peak_coverage)) / 2
    return SpacingAnalysis(
        profile=profile, min_spacing=min_spacing, prominence_ratio=prominence_ratio,
        profile_range=profile_range, min_prominence=float(min_prominence),
        peaks=peaks, prominences=prominences, spacings=spacings,
        spacing_counts=tuple(spacing_counts.items()),
        selected_spacing=int(most_common_spacing), mode_count=int(mode_count),
        tolerance=int(tolerance), matching_count=int(matching_count),
        spacing_consistency=float(consistency), expected_peaks=float(expected_peaks),
        peak_coverage=float(peak_coverage), confidence=float(confidence),
    )


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
    analysis = analyze_dominant_spacing(profile, min_spacing, prominence_ratio)
    return analysis.selected_spacing, analysis.confidence


def analyze_grid_offset(profile: np.ndarray, spacing: int) -> OffsetAnalysis:
    """Return every score used by :func:`find_grid_offset`."""
    profile = np.asarray(profile)
    if spacing <= 0 or len(profile) < spacing:
        return OffsetAnalysis(profile, spacing, np.array([], dtype=float), 0, 0.0)

    scores = np.array(
        [float(profile[np.arange(offset, len(profile), spacing)].sum())
         for offset in range(spacing)],
        dtype=float,
    )
    selected_offset = int(np.argmax(scores))
    return OffsetAnalysis(
        profile, spacing, scores, selected_offset, float(scores[selected_offset])
    )


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
    return analyze_grid_offset(profile, spacing).selected_offset


def analyze_gradient_profiles(
    image: Image.Image, smoothing_sigma: float = 1.0
) -> GradientAnalysis:
    """Compute the detector's grayscale, gradients, sums, and smoothed profiles."""
    if image.mode in ("RGBA", "LA"):
        gray_image = image.split()[0]
        grayscale_method = f"{image.mode} first channel"
    else:
        gray_image = image.convert("L")
        grayscale_method = f"{image.mode} -> L"
    grayscale = np.array(gray_image, dtype=np.float32)
    gradient_h = np.abs(np.diff(grayscale, axis=1, append=grayscale[:, -1:]))
    gradient_v = np.abs(np.diff(grayscale, axis=0, append=grayscale[-1:, :]))
    raw_profile_h = np.sum(gradient_h, axis=0)
    raw_profile_v = np.sum(gradient_v, axis=1)
    profile_h = raw_profile_h.copy()
    profile_v = raw_profile_v.copy()
    if smoothing_sigma and smoothing_sigma > 0:
        profile_h = gaussian_filter1d(profile_h, sigma=smoothing_sigma)
        profile_v = gaussian_filter1d(profile_v, sigma=smoothing_sigma)
    return GradientAnalysis(
        grayscale, gradient_h, gradient_v, raw_profile_h, raw_profile_v,
        profile_h, profile_v, smoothing_sigma, grayscale_method
    )


def compute_gradient_profiles(
    image: Image.Image,
    smoothing_sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the 1D horizontal and vertical gradient profiles of an image.

    profile_h has length = image width (drives grid_w / offset_x);
    profile_v has length = image height (drives grid_h / offset_y).

    This is the per-image signal that grid detection consumes; exposing it
    separately lets callers aggregate profiles across multiple frames (see
    detect_grid_across_frames) before peak detection.

    Args:
        image: The PIL Image object to analyze.
        smoothing_sigma: Gaussian smoothing sigma (0 to disable).

    Returns:
        Tuple (profile_h, profile_v) of 1D NumPy arrays.
    """
    analysis = analyze_gradient_profiles(image, smoothing_sigma)
    return analysis.profile_h, analysis.profile_v


def detect_grid_from_profiles(
    profile_h: np.ndarray,
    profile_v: np.ndarray,
    min_grid_size: int = 4,
    min_confidence: float = 0.4,
) -> Tuple[int, int, int, int]:
    """
    Run grid detection on precomputed 1D gradient profiles.

    profile_h drives grid_w / offset_x (its length is the image width);
    profile_v drives grid_h / offset_y (its length is the image height).

    Returns (grid_w, grid_h, offset_x, offset_y) or (0, 0, 0, 0) if no
    reliable grid is detected (failure, low confidence, or implausible
    aspect ratio).

    Args:
        profile_h: Horizontal gradient profile (length = image width).
        profile_v: Vertical gradient profile (length = image height).
        min_grid_size: Minimum expected grid dimension for peak finding.
        min_confidence: Minimum confidence threshold (0-1) to accept detection.

    Returns:
        Tuple (grid_w, grid_h, offset_x, offset_y) or (0, 0, 0, 0).
    """
    return analyze_grid_from_profiles(
        profile_h, profile_v, min_grid_size, min_confidence, report=True
    ).result


def analyze_grid_from_profiles(
    profile_h: np.ndarray,
    profile_v: np.ndarray,
    min_grid_size: int = 4,
    min_confidence: float = 0.4,
    *,
    signals: GradientAnalysis | None = None,
    report: bool = False,
) -> GridDetectionAnalysis:
    """Run the canonical grid decision and retain every intermediate value."""
    profile_h = np.asarray(profile_h)
    profile_v = np.asarray(profile_v)
    actual_min_spacing = max(1, min_grid_size)
    horizontal = analyze_dominant_spacing(profile_h, actual_min_spacing)
    vertical = analyze_dominant_spacing(profile_v, actual_min_spacing)
    grid_w = horizontal.selected_spacing
    grid_h = vertical.selected_spacing
    average_confidence = (horizontal.confidence + vertical.confidence) / 2
    aspect_ratio = grid_w / grid_h if grid_h > 0 else 0.0
    rejection_reason = None

    if (
        len(profile_v) < actual_min_spacing * 2
        or len(profile_h) < actual_min_spacing * 2
    ):
        rejection_reason = "profiles_too_short"
    elif grid_w <= 0 or grid_h <= 0:
        rejection_reason = "spacing_not_found"
    elif average_confidence < min_confidence:
        rejection_reason = "confidence_below_threshold"
        if report:
            print(
                f"Grid detection confidence too low ({average_confidence:.2f} < {min_confidence}). "
                "Image may already be clean pixel art.", file=sys.stderr,
            )
    elif aspect_ratio < 0.5 or aspect_ratio > 2.0:
        rejection_reason = "aspect_ratio_out_of_range"
        if report:
            print(
                f"Detected grid {grid_w}x{grid_h} has inconsistent aspect ratio "
                f"({aspect_ratio:.2f}). Image may already be clean pixel art.",
                file=sys.stderr,
            )

    if rejection_reason is None:
        offset_x = analyze_grid_offset(profile_h, grid_w)
        offset_y = analyze_grid_offset(profile_v, grid_h)
        result = (grid_w, grid_h, offset_x.selected_offset, offset_y.selected_offset)
    else:
        offset_x = analyze_grid_offset(profile_h, 0)
        offset_y = analyze_grid_offset(profile_v, 0)
        result = (0, 0, 0, 0)

    return GridDetectionAnalysis(
        signals=signals, horizontal=horizontal, vertical=vertical,
        offset_x=offset_x, offset_y=offset_y, min_grid_size=min_grid_size,
        min_confidence=min_confidence, average_confidence=average_confidence,
        aspect_ratio=aspect_ratio, rejection_reason=rejection_reason, result=result,
    )


def analyze_grid(
    image: Image.Image,
    min_grid_size: int = 4,
    smoothing_sigma: float = 1.0,
    min_confidence: float = 0.4,
    *,
    report: bool = False,
) -> GridDetectionAnalysis:
    """Analyze an image once and expose the exact data behind the result."""
    signals = analyze_gradient_profiles(image, smoothing_sigma)
    return analyze_grid_from_profiles(
        signals.profile_h, signals.profile_v, min_grid_size, min_confidence,
        signals=signals, report=report,
    )


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
        return analyze_grid(
            image, min_grid_size, smoothing_sigma, min_confidence, report=True
        ).result
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


def detect_grid_across_frames(
    frames: Sequence[Image.Image],
    min_grid_size: int = 4,
    smoothing_sigma: float = 1.0,
    min_confidence: float = 0.4,
) -> Tuple[int, int, int, int]:
    """
    Detect a single shared grid across multiple animation frames.

    Sums the per-frame 1D gradient profiles *before* peak detection. Grid
    lines fall at fixed pixel positions in every frame, so their peaks add
    coherently, while moving content edges land at different positions and
    average down. Detection therefore becomes *more* robust as the frame
    count grows, and every frame is guaranteed the same grid -> the
    downsampled animation is temporally stable (no resolution/phase jitter).

    All frames must share the same dimensions (the animation orchestrator
    normalises sizes before calling this).

    Args:
        frames: Sequence of PIL Image objects, all the same size.
        min_grid_size: Minimum expected grid dimension for peak finding.
        smoothing_sigma: Gaussian smoothing sigma (0 to disable).
        min_confidence: Minimum confidence threshold (0-1) to accept detection.

    Returns:
        Tuple (grid_w, grid_h, offset_x, offset_y) or (0, 0, 0, 0).

    Raises:
        ValueError: if frames have differing dimensions.
    """
    frames = list(frames)
    if not frames:
        return (0, 0, 0, 0)
    if len(frames) == 1:
        return detect_grid_with_offset(
            frames[0], min_grid_size, smoothing_sigma, min_confidence
        )

    sizes = {f.size for f in frames}
    if len(sizes) > 1:
        raise ValueError(
            f"All frames must share the same dimensions for grid detection; got {sizes}"
        )

    sum_h = None
    sum_v = None
    for frame in frames:
        profile_h, profile_v = compute_gradient_profiles(frame, smoothing_sigma)
        if sum_h is None:
            sum_h = profile_h.astype(np.float64)
            sum_v = profile_v.astype(np.float64)
        else:
            sum_h += profile_h
            sum_v += profile_v

    return detect_grid_from_profiles(sum_h, sum_v, min_grid_size, min_confidence)
