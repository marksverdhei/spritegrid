"""Optional Manim walkthroughs for SpriteGrid's still-image pipeline.

This module deliberately has no top-level Manim import. Recording a walkthrough is
cheap and dependency-free; Manim is imported only when the caller asks to render the
trace. Most importantly, the recorder only observes copies of images that the normal
pipeline already produced. It never participates in a transformation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .detection import GridDetectionAnalysis, SpacingAnalysis


@dataclass
class WalkthroughStep:
    """One observable stage in a SpriteGrid transformation."""

    kind: str
    title: str
    image: Image.Image
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkthroughTrace:
    """Immutable-by-convention snapshots consumed by the Manim renderer."""

    source: str
    steps: list[WalkthroughStep] = field(default_factory=list)

    def add(
        self,
        kind: str,
        title: str,
        image: Image.Image,
        **details: Any,
    ) -> None:
        # copy() is the boundary that prevents later in-place PIL operations from
        # rewriting history (and prevents the renderer from touching live output).
        self.steps.append(WalkthroughStep(kind, title, image.copy(), details))


def record_grid_discovery(
    trace: WalkthroughTrace,
    image: Image.Image,
    analysis: GridDetectionAnalysis,
    applied_offset_x: int,
    applied_offset_y: int,
) -> None:
    """Record the canonical analysis object returned by the detector."""

    trace.add(
        "grid",
        "How SpriteGrid discovers the grid",
        image,
        analysis=analysis,
        applied_offset_x=applied_offset_x,
        applied_offset_y=applied_offset_y,
    )


def record_sampling(
    trace: WalkthroughTrace,
    source: Image.Image,
    output: Image.Image,
    grid_w: float,
    grid_h: float,
    offset_x: int,
    offset_y: int,
    kernel_size: tuple[int, int] = (3, 3),
) -> None:
    """Record exact sample centres and one representative local colour kernel."""

    cells_w, cells_h = output.size
    source_w, source_h = source.size
    centers_x = np.clip(
        (np.arange(cells_w) * grid_w + grid_w / 2).astype(int) + offset_x,
        0,
        source_w - 1,
    )
    centers_y = np.clip(
        (np.arange(cells_h) * grid_h + grid_h / 2).astype(int) + offset_y,
        0,
        source_h - 1,
    )

    cell_x, cell_y = cells_w // 2, cells_h // 2
    center_x, center_y = int(centers_x[cell_x]), int(centers_y[cell_y])
    kernel_w, kernel_h = kernel_size
    x0 = max(0, center_x - kernel_w // 2)
    y0 = max(0, center_y - kernel_h // 2)
    x1 = min(source_w, center_x + kernel_w // 2 + 1)
    y1 = min(source_h, center_y + kernel_h // 2 + 1)

    trace.add(
        "sampling",
        "Sample each cell's local colour",
        output,
        source=source.copy(),
        centers_x=centers_x,
        centers_y=centers_y,
        representative_cell=(cell_x, cell_y),
        representative_center=(center_x, center_y),
        kernel_box=(x0, y0, x1, y1),
        sampled_color=output.getpixel((cell_x, cell_y)),
        sample_method="channel-wise median",
        grid_w=grid_w,
        grid_h=grid_h,
    )


def _require_manim():
    try:
        import manim
    except ImportError as exc:
        raise RuntimeError(
            "Manim is required to render a walkthrough. "
            "Install SpriteGrid with: pip install 'spritegrid[walkthrough]'"
        ) from exc
    return manim


def render_walkthrough(
    trace: WalkthroughTrace,
    output_path: str | Path,
    *,
    quality: str = "medium_quality",
) -> Path:
    """Render *trace* to MP4 and return the requested output path.

    The render happens after SpriteGrid has completed its output transformation.
    Rendering errors therefore cannot change the output image.
    """

    manim = _require_manim()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Walkthrough output must use the .mp4 extension")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class SpriteGridWalkthrough(manim.Scene):
        def _heading(self, text: str):
            return manim.Text(text, font_size=34, weight="BOLD").to_edge(manim.UP)

        def _image(self, image: Image.Image, height: float = 4.7):
            rgba = image.convert("RGBA")
            if rgba.getextrema()[3][0] < 255:
                # Transparency is otherwise nearly invisible against Manim's black
                # background. Composite the renderer's copy over a checkerboard;
                # the recorded pipeline image remains untouched.
                tile = max(1, min(rgba.size) // 16)
                checker = Image.new("RGBA", rgba.size, (214, 214, 214, 255))
                pixels = np.asarray(checker).copy()
                yy, xx = np.indices((rgba.height, rgba.width))
                dark = ((xx // tile) + (yy // tile)) % 2 == 0
                pixels[dark, :3] = (166, 166, 166)
                checker = Image.fromarray(pixels)
                rgba = Image.alpha_composite(checker, rgba)
            mob = manim.ImageMobject(np.array(rgba))
            mob.set_resampling_algorithm(manim.RESAMPLING_ALGORITHMS["nearest"])
            mob.height = height
            return mob

        def _caption(self, text: str):
            return manim.Text(text, font_size=23).to_edge(manim.DOWN)

        def _show_image_step(self, step: WalkthroughStep) -> None:
            heading = self._heading(step.title)
            picture = self._image(step.image)
            caption_text = step.details.get(
                "caption",
                f"{step.image.width} x {step.image.height}  |  {step.image.mode}",
            )
            caption = self._caption(caption_text)
            self.play(
                manim.Write(heading), manim.FadeIn(picture), manim.FadeIn(caption)
            )
            self.wait(0.7)
            self.play(
                manim.FadeOut(heading), manim.FadeOut(picture), manim.FadeOut(caption)
            )

        def _show_grid(self, step: WalkthroughStep) -> None:
            analysis: GridDetectionAnalysis = step.details["analysis"]
            signals = analysis.signals
            if signals is None:
                raise ValueError("Image walkthrough requires detector signal diagnostics")

            source = self._image(step.image, height=3.7)
            gray = self._array_image(signals.grayscale, height=3.7)
            first = manim.Group(
                manim.Group(source, manim.Text("input", font_size=21)).arrange(
                    manim.DOWN, buff=0.15
                ),
                manim.Arrow(manim.LEFT, manim.RIGHT, buff=0.1),
                manim.Group(gray, manim.Text("detector signal", font_size=21)).arrange(
                    manim.DOWN, buff=0.15
                ),
            ).arrange(manim.RIGHT, buff=0.45)
            self._detector_slide(
                "1. Convert to the detector's grayscale signal",
                first,
                f"Canonical conversion: {signals.grayscale_method}. Transparency is not composited.",
            )

            grad_h = self._array_image(signals.gradient_h, height=3.25)
            grad_v = self._array_image(signals.gradient_v, height=3.25)
            gradients = manim.Group(
                manim.Group(
                    grad_h,
                    manim.Text("|pixel[x+1] - pixel[x]|", font_size=19),
                    manim.Text("x / width evidence", font_size=18, color=manim.BLUE_C),
                ).arrange(manim.DOWN, buff=0.1),
                manim.Group(
                    grad_v,
                    manim.Text("|pixel[y+1] - pixel[y]|", font_size=19),
                    manim.Text("y / height evidence", font_size=18, color=manim.GREEN_C),
                ).arrange(manim.DOWN, buff=0.1),
            ).arrange(manim.RIGHT, buff=0.5)
            self._detector_slide(
                "2. Measure every adjacent-pixel change",
                gradients,
                "Bright pixels are large absolute differences. The last row/column is repeated, so its difference is zero.",
            )

            charts = manim.VGroup(
                self._profile_chart(
                    signals.raw_profile_h, signals.profile_h, analysis.horizontal,
                    "x profile: sum each gradient column", manim.BLUE_C,
                ),
                self._profile_chart(
                    signals.raw_profile_v, signals.profile_v, analysis.vertical,
                    "y profile: sum each gradient row", manim.GREEN_C,
                ),
            ).arrange(manim.DOWN, buff=0.35)
            self._detector_slide(
                "3. Collapse the image into two 1D edge profiles",
                charts,
                f"Gray = raw sums. Colour = Gaussian-smoothed signal (sigma = {signals.smoothing_sigma:g}).",
            )

            candidates = manim.VGroup(
                self._candidate_panel("x / grid width", analysis.horizontal),
                self._candidate_panel("y / grid height", analysis.vertical),
            ).arrange(manim.RIGHT, buff=0.55)
            self._detector_slide(
                "4. Find peaks, then measure every local interval",
                candidates,
                "SciPy find_peaks uses the shown minimum distance and prominence. Adjacent peak gaps are the local spacing candidates.",
                hold=2.0,
            )

            confidence = manim.VGroup(
                self._confidence_panel("x / width", analysis.horizontal),
                self._confidence_panel("y / height", analysis.vertical),
            ).arrange(manim.RIGHT, buff=0.55)
            gate = manim.Text(
                f"mean confidence = {analysis.average_confidence:.3f}  "
                f"(required >= {analysis.min_confidence:g})    "
                f"aspect = {analysis.aspect_ratio:.3f}  (required 0.5..2.0)",
                font_size=21,
            ).next_to(confidence, manim.DOWN, buff=0.3)
            decision = manim.VGroup(confidence, gate)
            self._detector_slide(
                "5. Select the modal spacing and test confidence",
                decision,
                "confidence = (spacing consistency + capped peak coverage) / 2",
                hold=2.0,
            )

            grid_w, grid_h, detected_x, detected_y = analysis.result
            if analysis.rejection_reason is None:
                phases = manim.VGroup(
                    self._phase_panel("x phase", analysis.offset_x, manim.BLUE_C),
                    self._phase_panel("y phase", analysis.offset_y, manim.GREEN_C),
                ).arrange(manim.RIGHT, buff=0.7)
                self._detector_slide(
                    "6. Scan every possible grid phase",
                    phases,
                    "For each offset 0..spacing-1, sum the smoothed profile at offset + k*spacing; the maximum wins.",
                    hold=1.8,
                )
            else:
                rejected = manim.Text(
                    f"Rejected: {analysis.rejection_reason.replace('_', ' ')}",
                    font_size=35, color=manim.RED_C,
                )
                self._detector_slide(
                    "6. Apply the detector's acceptance gates", rejected,
                    "No phase scan runs after a rejected spacing decision.",
                )

            picture = self._image(step.image, height=4.5)
            overlay = self._grid_overlay(
                picture, step.image.size, grid_w, grid_h, detected_x, detected_y
            )
            applied = (
                step.details["applied_offset_x"], step.details["applied_offset_y"]
            )
            if grid_w and grid_h:
                outcome = (
                    f"accepted cell {grid_w} x {grid_h}px | detected phase "
                    f"({detected_x}, {detected_y}) | sampling offset {applied}"
                )
            else:
                outcome = "no reliable repeating grid selected"
            self._detector_slide(
                "7. The accepted grid, over the original pixels",
                manim.Group(picture, overlay), outcome, hold=1.5,
            )

        def _detector_slide(self, title, body, caption, hold=1.25):
            heading = self._heading(title)
            if heading.width > 12.4:
                heading.scale_to_fit_width(12.4)
            if body.width > 12.2:
                body.scale_to_fit_width(12.2)
            if body.height > 5.2:
                body.scale_to_fit_height(5.2)
            body.move_to(manim.ORIGIN + manim.DOWN * 0.05)
            footer = self._caption(caption)
            if footer.width > 12.2:
                footer.scale_to_fit_width(12.2)
            self.play(manim.Write(heading), manim.FadeIn(body), manim.FadeIn(footer))
            self.wait(hold)
            self.play(manim.FadeOut(heading), manim.FadeOut(body), manim.FadeOut(footer))

        def _array_image(self, values: np.ndarray, height: float):
            values = np.asarray(values, dtype=float)
            low, span = float(values.min()), float(np.ptp(values))
            pixels = ((values - low) / (span or 1.0) * 255).astype(np.uint8)
            mob = manim.ImageMobject(pixels)
            mob.set_resampling_algorithm(manim.RESAMPLING_ALGORITHMS["nearest"])
            mob.height = height
            return mob

        def _line(self, values, width, height, color, low=None, high=None):
            values = np.asarray(values, dtype=float)
            low = float(values.min()) if low is None else low
            high = float(values.max()) if high is None else high
            normalized = (values - low) / ((high - low) or 1.0)
            points = [np.array([
                -width / 2 + width * i / max(1, len(values) - 1),
                -height / 2 + height * value, 0,
            ]) for i, value in enumerate(normalized)]
            line = manim.VMobject(color=color, stroke_width=2)
            line.set_points_as_corners(points)
            return line

        def _profile_chart(self, raw, smooth, spacing, label, color):
            raw = np.asarray(raw, dtype=float)
            smooth = np.asarray(smooth, dtype=float)
            low = float(min(raw.min(), smooth.min()))
            high = float(max(raw.max(), smooth.max()))
            width, height = 9.4, 1.25
            frame = manim.Rectangle(width=width, height=height, stroke_opacity=0.35)
            raw_line = self._line(raw, width, height, manim.GRAY_B, low, high)
            smooth_line = self._line(smooth, width, height, color, low, high)
            peaks = manim.VGroup()
            for index in spacing.peaks:
                x = -width / 2 + width * int(index) / max(1, len(smooth) - 1)
                y = -height / 2 + height * (smooth[int(index)] - low) / ((high - low) or 1)
                peaks.add(manim.Dot([x, y, 0], radius=0.045, color=manim.RED_C))
            plot = manim.VGroup(frame, raw_line, smooth_line, peaks)
            title = manim.Text(label, font_size=19)
            return manim.VGroup(title, plot).arrange(manim.DOWN, buff=0.08)

        def _candidate_panel(self, label: str, spacing: SpacingAnalysis):
            def preview(values):
                shown = [str(int(value)) for value in values[:12]]
                rows = [", ".join(shown[index:index + 6])
                        for index in range(0, len(shown), 6)]
                value = "\n".join(rows) or "none"
                if len(values) > 12:
                    value += f"\n... ({len(values)} total)"
                return value

            gaps = preview(spacing.spacings)
            peaks = preview(spacing.peaks)
            lines = manim.VGroup(
                manim.Text(label, font_size=25, weight="BOLD"),
                manim.Text(
                    f"distance >= {spacing.min_spacing}px\nprominence >= {spacing.min_prominence:.2f}",
                    font_size=19, line_spacing=0.8,
                ),
                manim.Text(
                    f"peak indices\n{peaks}",
                    font_size=18, line_spacing=0.8,
                ),
                manim.Text(f"adjacent gaps\n{gaps}", font_size=18, line_spacing=0.8),
            ).arrange(manim.DOWN, aligned_edge=manim.LEFT, buff=0.22)
            return manim.SurroundingRectangle(lines, buff=0.22).add(lines)

        def _confidence_panel(self, label: str, spacing: SpacingAnalysis):
            counts = ", ".join(f"{gap}:{count}" for gap, count in spacing.spacing_counts)
            counts = counts or "none"
            total = len(spacing.spacings)
            text = manim.VGroup(
                manim.Text(label, font_size=24, weight="BOLD"),
                manim.Text(f"gap counts  {counts}", font_size=18),
                manim.Text(
                    f"mode {spacing.selected_spacing}px +/- {spacing.tolerance}px\n"
                    f"consistency {spacing.matching_count}/{total} = {spacing.spacing_consistency:.3f}",
                    font_size=18, line_spacing=0.8,
                ),
                manim.Text(
                    f"expected peaks {spacing.expected_peaks:.2f}\n"
                    f"coverage {len(spacing.peaks)}/{spacing.expected_peaks:.2f} = {spacing.peak_coverage:.3f}",
                    font_size=18, line_spacing=0.8,
                ),
                manim.Text(f"axis confidence {spacing.confidence:.3f}", font_size=22),
            ).arrange(manim.DOWN, aligned_edge=manim.LEFT, buff=0.18)
            return manim.SurroundingRectangle(text, buff=0.22).add(text)

        def _phase_panel(self, label, offset, color):
            scores = np.asarray(offset.scores, dtype=float)
            maximum = float(scores.max()) if scores.size else 1.0
            bars = manim.VGroup()
            for index, score in enumerate(scores):
                bar = manim.Rectangle(
                    width=0.32, height=2.4 * float(score) / (maximum or 1.0),
                    fill_opacity=0.85, stroke_width=0,
                    color=manim.YELLOW if index == offset.selected_offset else color,
                )
                bars.add(bar)
            bars.arrange(manim.RIGHT, aligned_edge=manim.DOWN, buff=0.07)
            labels = manim.Text(
                " ".join(str(i) for i in range(len(scores))), font_size=14
            ).next_to(bars, manim.DOWN, buff=0.12)
            title = manim.Text(
                f"{label}: selected {offset.selected_offset}, score {offset.selected_score:.2f}",
                font_size=21,
            )
            return manim.VGroup(title, bars, labels).arrange(manim.DOWN, buff=0.15)

        def _grid_overlay(self, picture, image_size, grid_w, grid_h, phase_x, phase_y):
            overlay = manim.VGroup()
            if grid_w <= 0 or grid_h <= 0:
                return overlay
            image_w, image_h = image_size
            left, right = picture.get_left()[0], picture.get_right()[0]
            bottom, top = picture.get_bottom()[1], picture.get_top()[1]
            for x in range(phase_x, image_w, grid_w):
                sx = left + (x / image_w) * (right - left)
                overlay.add(manim.Line([sx, bottom, 0], [sx, top, 0],
                                       color=manim.RED, stroke_width=1))
            for y in range(phase_y, image_h, grid_h):
                sy = top - (y / image_h) * (top - bottom)
                overlay.add(manim.Line([left, sy, 0], [right, sy, 0],
                                       color=manim.RED, stroke_width=1))
            return overlay

        def _show_sampling(self, step: WalkthroughStep) -> None:
            heading = self._heading(step.title)
            source = self._image(step.details["source"], height=4.5).shift(
                manim.LEFT * 2.5
            )
            result = self._image(step.image, height=4.5).shift(manim.RIGHT * 2.5)
            arrow = manim.Arrow(source.get_right(), result.get_left(), buff=0.25)

            x0, y0, x1, y1 = step.details["kernel_box"]
            left, right = source.get_left()[0], source.get_right()[0]
            bottom, top = source.get_bottom()[1], source.get_top()[1]
            box = manim.Rectangle(
                width=(x1 - x0) / step.details["source"].width * (right - left),
                height=(y1 - y0) / step.details["source"].height * (top - bottom),
                color=manim.YELLOW,
                stroke_width=4,
            )
            box.move_to(
                [
                    left
                    + ((x0 + x1) / 2 / step.details["source"].width) * (right - left),
                    top
                    - ((y0 + y1) / 2 / step.details["source"].height) * (top - bottom),
                    0,
                ]
            )
            color = step.details["sampled_color"]
            method = step.details["sample_method"]
            caption = self._caption(
                f"local 3 x 3 {method} -> output colour {color}"
            )
            labels = manim.VGroup(
                manim.Text("source cells", font_size=22).next_to(source, manim.DOWN),
                manim.Text("one pixel per cell", font_size=22).next_to(
                    result, manim.DOWN
                ),
            )
            self.play(manim.Write(heading), manim.FadeIn(source), manim.FadeIn(result))
            self.play(
                manim.Create(box),
                manim.GrowArrow(arrow),
                manim.FadeIn(labels),
                manim.FadeIn(caption),
            )
            self.wait(1.2)
            self.play(
                *[
                    manim.FadeOut(m)
                    for m in (heading, source, result, arrow, box, labels, caption)
                ]
            )

        def construct(self) -> None:
            intro = manim.VGroup(
                manim.Text("SpriteGrid", font_size=58, weight="BOLD"),
                manim.Text("an exact transformation walkthrough", font_size=28),
            ).arrange(manim.DOWN)
            self.play(manim.FadeIn(intro, shift=manim.UP * 0.3))
            self.wait(0.8)
            self.play(manim.FadeOut(intro))
            for step in trace.steps:
                if step.kind == "grid":
                    self._show_grid(step)
                elif step.kind == "sampling":
                    self._show_sampling(step)
                else:
                    self._show_image_step(step)
            outro = manim.VGroup(
                manim.Text("Same pipeline. Same pixels.", font_size=42, weight="BOLD"),
                manim.Text("The walkthrough only observes snapshots.", font_size=25),
            ).arrange(manim.DOWN)
            self.play(manim.FadeIn(outro))
            self.wait(1.0)

    qualities = {
        "low_quality",
        "medium_quality",
        "high_quality",
        "production_quality",
    }
    if quality not in qualities:
        raise ValueError(f"Unknown Manim quality: {quality}")

    with manim.tempconfig(
        {
            "quality": quality,
            "format": "mp4",
            "media_dir": str(output_path.parent),
            "output_file": output_path.stem,
            "preview": False,
            "disable_caching": True,
        }
    ):
        scene = SpriteGridWalkthrough()
        scene.render()
        rendered = Path(scene.renderer.file_writer.movie_file_path)

    if rendered != output_path:
        rendered.replace(output_path)
    return output_path
