# animation.py
"""Multi-frame (animation) support for SpriteGrid.

The grid in AI-generated pixel art is a property of the *whole animation*, not
of any single frame. Detecting it per frame makes the output resolution, grid
phase, and palette jitter between frames, so the animation flickers and the
frames no longer line up. This module detects ONE shared grid across all frames
(see :func:`spritegrid.detection.detect_grid_across_frames`) and downsamples
every frame against that identical lattice, so the cleaned animation is
temporally stable.

Supported I/O (all dependency-light; ``cv2`` is imported lazily only for video):
    - animated GIF / APNG
    - a directory of numbered PNG frames
    - video (.mp4 / .webm / .mov / ...) via opencv
"""

import glob
import os
import re
import sys

import numpy as np
from PIL import Image, ImageSequence

from .detection import detect_grid_across_frames
from .main import (
    apply_aspect_ratio,
    apply_resolution,
    create_downsampled_image,
    load_image,
)
from .utils import enforce_symmetry

VIDEO_EXTS = (".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v")
_FRAME_DIR_EXTS = (".png", ".gif", ".jpg", ".jpeg", ".webp", ".bmp")


# --------------------------------------------------------------------------- #
# Source classification                                                       #
# --------------------------------------------------------------------------- #
def is_animated_source(source: str) -> bool:
    """Return True if *source* should be treated as a multi-frame animation.

    True for: a local directory of frames, a video file (by extension), or an
    animated image (GIF/APNG with >1 frame). Returns False for plain stills and
    for anything that can't be cheaply inspected (e.g. remote URLs), so callers
    fall back to the single-image path. Never raises.
    """
    try:
        if isinstance(source, str) and os.path.isdir(source):
            return True
        if isinstance(source, str) and source.lower().endswith(VIDEO_EXTS):
            return True
        # Cheap local peek; avoid downloading remote URLs just to classify.
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            return False
        with Image.open(source) as img:
            return bool(
                getattr(img, "is_animated", False) and getattr(img, "n_frames", 1) > 1
            )
    except Exception:
        return False


def _natural_key(name: str):
    """Sort key that orders ``frame2`` before ``frame10`` (numeric-aware)."""
    return [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in re.split(r"(\d+)", name)
    ]


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def _import_cv2():
    try:
        import cv2  # noqa: PLC0415 (lazy: video path only)

        return cv2
    except ImportError as exc:  # pragma: no cover - opencv is a declared dep
        raise ImportError(
            "Video support requires opencv-python (cv2), which is a declared "
            "spritegrid dependency. Reinstall with `uv sync` / `pip install opencv-python`."
        ) from exc


def _normalize_sizes(frames: list[Image.Image]) -> list[Image.Image]:
    """Resize any odd-sized frames to match the first frame (NEAREST), with a warning.

    Cross-frame grid detection sums per-frame profiles, which requires a common
    frame size. Mild size drift (e.g. a hand-assembled frame folder) is repaired
    here rather than rejected.
    """
    if not frames:
        return frames
    base = frames[0].size
    mismatched = False
    out = []
    for frame in frames:
        if frame.size != base:
            mismatched = True
            frame = frame.resize(base, resample=Image.NEAREST)
        out.append(frame)
    if mismatched:
        print(
            f"Warning: frames had differing sizes; normalised all to "
            f"{base[0]}x{base[1]}.",
            file=sys.stderr,
        )
    return out


def _coalesce_animation(img: Image.Image):
    """Flatten a possibly-delta-encoded GIF/APNG into full RGBA frames.

    Composites each frame onto a running canvas so partial frames and disposal
    semantics resolve to standalone images. Fully-opaque frames simply replace
    the canvas (the common case), so independent frames are preserved exactly.
    Returns ``(frames, durations_ms)``.
    """
    frames: list[Image.Image] = []
    durations: list[int] = []
    canvas = None
    for raw in ImageSequence.Iterator(img):
        frame = raw.convert("RGBA")
        if canvas is None:
            canvas = frame.copy()
        else:
            canvas = Image.alpha_composite(canvas, frame)
        frames.append(canvas.copy())
        durations.append(
            int(raw.info.get("duration", img.info.get("duration", 100)) or 100)
        )
    return frames, durations


def _load_video_frames(path: str):
    cv2 = _import_cv2()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames: list[Image.Image] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb).convert("RGBA"))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from video: {path}")
    duration = int(round(1000.0 / fps)) if fps and fps > 0 else 100
    meta = {
        "durations": [duration] * len(frames),
        "loop": 0,
        "kind": "video",
        "fps": fps if fps and fps > 0 else 10.0,
    }
    return _normalize_sizes(frames), meta


def load_frames(source: str):
    """Load animation frames from a file, URL, or directory.

    Returns ``(frames, meta)`` where ``frames`` are equal-size, fully-coalesced
    RGBA images and ``meta`` is ``{durations, loop, kind}`` (``kind`` is one of
    ``single``/``gif``/``apng``/``frame_dir``/``video``).
    """
    meta = {"durations": None, "loop": 0, "kind": "single"}

    # Directory of numbered frames.
    if isinstance(source, str) and os.path.isdir(source):
        paths = sorted(
            (
                p
                for p in glob.glob(os.path.join(source, "*"))
                if p.lower().endswith(_FRAME_DIR_EXTS)
            ),
            key=lambda p: _natural_key(os.path.basename(p)),
        )
        if not paths:
            raise ValueError(f"No image frames found in directory: {source}")
        frames = [Image.open(p).convert("RGBA") for p in paths]
        meta["kind"] = "frame_dir"
        return _normalize_sizes(frames), meta

    # Video file.
    if isinstance(source, str) and source.lower().endswith(VIDEO_EXTS):
        return _load_video_frames(source)

    # Single image, animated GIF/APNG, or URL (reuses still-image loader).
    img = load_image(source)
    if img is None:
        raise ValueError(f"Could not load image from: {source}")

    if getattr(img, "is_animated", False) and getattr(img, "n_frames", 1) > 1:
        frames, durations = _coalesce_animation(img)
        meta["kind"] = "gif" if (img.format == "GIF") else "apng"
        meta["durations"] = durations
        meta["loop"] = img.info.get("loop", 0)
        return _normalize_sizes(frames), meta

    return [img.convert("RGBA")], meta


# --------------------------------------------------------------------------- #
# Saving                                                                       #
# --------------------------------------------------------------------------- #
def _looks_like_dir_output(path: str) -> bool:
    return os.path.isdir(path) or os.path.splitext(path)[1] == ""


def _duration_list(durations, n: int):
    if durations is None:
        return None
    if isinstance(durations, (int, float)):
        return [int(durations)] * n
    out = [int(x) for x in durations]
    if not out:
        return None
    if len(out) < n:
        out = out + [out[-1]] * (n - len(out))
    return out[:n]


def _shared_palette_quantize(frames: list[Image.Image]) -> list[Image.Image]:
    """Quantize every frame against ONE shared adaptive palette.

    ``create_downsampled_image``'s quantization is deterministic, so the only
    place per-frame flicker can creep in is GIF's per-frame adaptive palette.
    Building a single palette from all frames removes that flicker. Index 255 is
    reserved for transparency (alpha == 0).
    """
    w, h = frames[0].size
    montage = Image.new("RGB", (w, h * len(frames)))
    for i, frame in enumerate(frames):
        montage.paste(frame.convert("RGB"), (0, i * h))
    # 255 colours so palette index 255 is free for a transparency slot.
    palette_img = montage.convert("P", palette=Image.ADAPTIVE, colors=255)

    out = []
    for frame in frames:
        p = frame.convert("RGB").quantize(palette=palette_img, dither=Image.Dither.NONE)
        if frame.mode == "RGBA":
            transparent = frame.getchannel("A").point(lambda a: 255 if a == 0 else 0)
            transparent = transparent.convert("1")
            if transparent.getbbox() is not None:
                p.paste(255, transparent)
                p.info["transparency"] = 255
        out.append(p)
    return out


def _save_gif(frames, path, durations, loop):
    rgba = [f.convert("RGBA") for f in frames]
    pal_frames = _shared_palette_quantize(rgba)
    dur = _duration_list(durations, len(frames)) or 100
    save_kwargs = dict(
        save_all=True,
        append_images=pal_frames[1:],
        duration=dur,
        loop=loop or 0,
        disposal=2,
        optimize=False,
    )
    if "transparency" in pal_frames[0].info:
        save_kwargs["transparency"] = 255
    pal_frames[0].save(path, **save_kwargs)
    print(f"Success: animation saved to '{path}' ({len(frames)} frames)")


def _save_apng(frames, path, durations, loop):
    rgba = [f.convert("RGBA") for f in frames]
    dur = _duration_list(durations, len(frames)) or 100
    rgba[0].save(
        path,
        save_all=True,
        append_images=rgba[1:],
        duration=dur,
        loop=loop or 0,
    )
    print(f"Success: animation saved to '{path}' ({len(frames)} frames)")


def _save_video(frames, path, durations):
    cv2 = _import_cv2()
    dur = _duration_list(durations, len(frames))
    if dur:
        avg_ms = sum(dur) / len(dur)
        fps = 1000.0 / avg_ms if avg_ms > 0 else 10.0
    else:
        fps = 10.0

    rgb_frames = [np.array(f.convert("RGB")) for f in frames]
    h, w = rgb_frames[0].shape[:2]
    # Many codecs reject odd dimensions; pad by one pixel where needed.
    pad_w, pad_h = w + (w % 2), h + (h % 2)

    ext = os.path.splitext(path)[1].lower()
    fourcc_str = "VP80" if ext == ".webm" else "mp4v"
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*fourcc_str), fps, (pad_w, pad_h)
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open a video writer for '{path}' (codec '{fourcc_str}'). "
            "Try a .mp4 output, or upscale tiny frames with --res."
        )
    try:
        for rgb in rgb_frames:
            if (pad_w, pad_h) != (w, h):
                padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
                padded[:h, :w] = rgb
                rgb = padded
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    print(f"Success: video saved to '{path}' ({len(frames)} frames @ {fps:.1f} fps)")


def save_frames(frames, path, durations=None, loop: int = 0) -> None:
    """Write *frames* out, dispatching on the output path.

    ``.gif`` -> animated GIF (shared palette); ``.png``/``.apng`` -> APNG;
    ``.mp4``/``.webm``/... -> video; an existing directory or an extension-less
    path -> numbered ``frame_0000.png`` files.
    """
    if not frames:
        raise ValueError("No frames to save.")

    if _looks_like_dir_output(path):
        os.makedirs(path, exist_ok=True)
        pad = max(4, len(str(len(frames) - 1)))
        for i, frame in enumerate(frames):
            frame.convert("RGBA").save(
                os.path.join(path, f"frame_{i:0{pad}d}.png")
            )
        print(f"Success: {len(frames)} frames saved to '{path}/'")
        return

    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTS:
        _save_video(frames, path, durations)
    elif ext == ".gif":
        _save_gif(frames, path, durations, loop)
    elif ext in (".png", ".apng"):
        _save_apng(frames, path, durations, loop)
    else:
        raise ValueError(
            f"Unsupported animation output extension: '{ext}'. Use .gif, .png "
            "(APNG), a video extension (.mp4/.webm), or a directory."
        )


# --------------------------------------------------------------------------- #
# Processing                                                                   #
# --------------------------------------------------------------------------- #
def _shared_crop(frames: list[Image.Image]) -> list[Image.Image]:
    """Crop every frame to the UNION of per-frame content boxes.

    Cropping each frame independently would make the sprite size jitter as the
    content moves; a single shared box keeps every output frame the same size.
    """
    if any(f.mode != "RGBA" for f in frames):
        return frames  # mirrors the still-image crop guard (RGBA only)

    union = None
    for frame in frames:
        alpha = np.array(frame.getchannel("A"))
        ys, xs = np.where(alpha > 0)
        if len(xs) == 0:
            continue
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        if union is None:
            union = bbox
        else:
            union[0] = min(union[0], bbox[0])
            union[1] = min(union[1], bbox[1])
            union[2] = max(union[2], bbox[2])
            union[3] = max(union[3], bbox[3])

    if union is None:
        return frames  # every frame fully transparent
    return [frame.crop(tuple(union)) for frame in frames]


def process_frames(
    frames,
    *,
    min_grid: int = 4,
    quantize: int = 8,
    offset=None,
    auto_offset: bool = False,
    crop: bool = False,
    symmetric: bool = False,
    res=None,
    aspect_ratio=None,
    verbose: bool = True,
) -> list[Image.Image]:
    """Downsample a list of frames against ONE shared grid.

    Pure (no file I/O), so both the CLI animation path and the ComfyUI batch
    node reuse it. Detects a single grid + phase offset across all frames, then
    downsamples each frame to the identical cell count. Returns frames unchanged
    if no grid is found or the animation is already clean (idempotence).
    """
    frames = _normalize_sizes(list(frames))
    if not frames:
        return []

    grid_w, grid_h, det_ox, det_oy = detect_grid_across_frames(
        frames, min_grid_size=min_grid
    )
    if grid_w <= 0 or grid_h <= 0:
        if verbose:
            print("No consistent grid detected across frames; returning unchanged.")
        return list(frames)

    if offset is not None:
        off_x, off_y = offset
    elif auto_offset:
        off_x, off_y = det_ox, det_oy
    else:
        off_x, off_y = 0, 0

    width, height = frames[0].size
    num_cells_w = max(1, round(width / grid_w))
    num_cells_h = max(1, round(height / grid_h))

    if num_cells_w == width and num_cells_h == height:
        if verbose:
            print("Animation already appears to be clean pixel art; returning unchanged.")
        return list(frames)

    if verbose:
        print(
            f"Shared grid {grid_w}x{grid_h} px/cell -> {num_cells_w}x{num_cells_h} "
            f"cells (offset {off_x},{off_y}) across {len(frames)} frame(s)."
        )

    out = [
        create_downsampled_image(
            frame,
            grid_w,
            grid_h,
            num_cells_w,
            num_cells_h,
            quantize,
            offset_x=off_x,
            offset_y=off_y,
        )
        for frame in frames
    ]

    if symmetric:
        out = [enforce_symmetry(frame) for frame in out]
    if crop:
        out = _shared_crop(out)
    if res is not None:
        out = [apply_resolution(frame, res) for frame in out]
    elif aspect_ratio is not None:
        out = [apply_aspect_ratio(frame, aspect_ratio) for frame in out]

    return out


def process_animation(
    source: str,
    output_file=None,
    *,
    min_grid: int = 4,
    quantize: int = 8,
    offset=None,
    auto_offset: bool = False,
    crop: bool = False,
    symmetric: bool = False,
    res=None,
    aspect_ratio=None,
    fps=None,
    duration=None,
    show: bool = False,
) -> list[Image.Image]:
    """Load an animation, downsample every frame against one shared grid, save it.

    Output frame timing precedence: explicit ``duration`` (ms) > ``fps`` >
    the source animation's original per-frame timing > 100 ms.
    """
    frames, meta = load_frames(source)
    print(f"Loaded {len(frames)} frame(s) from '{source}' (kind={meta['kind']}).")

    if len(frames) == 1:
        print("Single frame detected; processing as a still image.")

    out = process_frames(
        frames,
        min_grid=min_grid,
        quantize=quantize,
        offset=offset,
        auto_offset=auto_offset,
        crop=crop,
        symmetric=symmetric,
        res=res,
        aspect_ratio=aspect_ratio,
    )

    if duration is not None:
        durations = [int(duration)] * len(out)
    elif fps is not None and fps > 0:
        durations = [int(round(1000.0 / fps))] * len(out)
    else:
        durations = meta.get("durations")
    loop = meta.get("loop", 0)

    if output_file:
        save_frames(out, output_file, durations=durations, loop=loop)
    elif show:
        out[0].show(title=f"{source} (frame 1/{len(out)})")
        print(
            "Note: --show displays the first frame only; use -o to save the full animation."
        )
    else:
        print(
            f"Processed {len(out)} frame(s). No -o/--output given, so nothing was written."
        )

    return out
