"""Optional Manim walkthroughs for SpriteGrid's still-image pipeline.

This module deliberately has no top-level Manim import. Recording a walkthrough is
cheap and dependency-free; Manim is imported only when the caller asks to render the
trace. Most importantly, the recorder only observes copies of images that the normal
pipeline already produced. It never participates in a transformation.

The renderer is a single continuous scene in the spirit of a 3Blue1Brown episode:
one persistent image stays on screen from the first frame to the last, and every
stage of the algorithm is shown as a transformation of (or annotation anchored to)
that same image -- crossfaded in place, swept into 1D profiles, scanned by a phase
comb, and finally collapsed into the true-resolution sprite. There are no slide
cuts; nothing fades to black between steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .detection import GridDetectionAnalysis


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
        """One continuous shot. `self.picture` is created in the intro and is on
        screen until the final frame; every stage transforms it in place or draws
        on top of it. Headings and captions are likewise swapped via
        FadeTransform so the layout never resets."""

        # ---------------------------------------------------------------- text

        def _retitle(self, text: str):
            new = manim.Text(text, font_size=31, weight="BOLD").to_edge(manim.UP)
            if new.width > 12.6:
                new.scale_to_fit_width(12.6)
            if self.heading is None:
                self.heading = new
                return manim.Write(new)
            anim = manim.FadeTransform(self.heading, new)
            self.heading = new
            return anim

        def _recaption(self, text: str):
            new = manim.Text(text, font_size=21).to_edge(manim.DOWN)
            if new.width > 12.6:
                new.scale_to_fit_width(12.6)
            if self.caption is None:
                self.caption = new
                return manim.FadeIn(new)
            anim = manim.FadeTransform(self.caption, new)
            self.caption = new
            return anim

        # -------------------------------------------------------------- images

        def _to_mobject(self, image: Image.Image, height: float):
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

        def _array_to_mobject(self, values: np.ndarray, height: float):
            values = np.asarray(values, dtype=float)
            low, span = float(values.min()), float(np.ptp(values))
            pixels = ((values - low) / (span or 1.0) * 255).astype(np.uint8)
            mob = manim.ImageMobject(pixels)
            mob.set_resampling_algorithm(manim.RESAMPLING_ALGORITHMS["nearest"])
            mob.height = height
            return mob

        def _swap_picture(self, new_mobject, *extra_anims, run_time=1.0):
            """Crossfade the persistent picture into `new_mobject` IN PLACE --
            the one mechanism that lets 'the same image' change representation
            (colour -> grayscale -> gradient -> output) without ever leaving
            the screen."""
            new_mobject.height = self.picture.height
            new_mobject.move_to(self.picture)
            self.play(
                manim.FadeTransform(self.picture, new_mobject),
                *extra_anims,
                run_time=run_time,
            )
            self.picture = new_mobject

        def _pixel_to_point(self, size, px: float, py: float) -> np.ndarray:
            """Map source-image pixel coordinates onto the persistent picture."""
            w, h = size
            left, right = self.picture.get_left()[0], self.picture.get_right()[0]
            top, bottom = self.picture.get_top()[1], self.picture.get_bottom()[1]
            return np.array(
                [left + (px / w) * (right - left), top - (py / h) * (top - bottom), 0]
            )

        # -------------------------------------------------------------- charts

        def _polyline(self, values, width, height, color, low, high, stroke=2.5):
            values = np.asarray(values, dtype=float)
            normalized = (values - low) / ((high - low) or 1.0)
            points = [
                np.array(
                    [
                        -width / 2 + width * i / max(1, len(values) - 1),
                        -height / 2 + height * value,
                        0,
                    ]
                )
                for i, value in enumerate(normalized)
            ]
            line = manim.VMobject(color=color, stroke_width=stroke)
            line.set_points_as_corners(points)
            return line

        def _chart(self, raw, smooth, color, width=7.6, height=1.3):
            """Frame + raw line + smoothed line, plus a locator that maps a
            profile index to the smoothed curve's scene position (evaluated
            lazily so the chart can be positioned first)."""
            raw = np.asarray(raw, dtype=float)
            smooth = np.asarray(smooth, dtype=float)
            low = float(min(raw.min(), smooth.min()))
            high = float(max(raw.max(), smooth.max()))
            frame = manim.Rectangle(
                width=width, height=height, stroke_opacity=0.3, stroke_width=1.5
            )
            raw_line = self._polyline(raw, width, height, manim.GRAY_C, low, high, 1.8)
            smooth_line = self._polyline(smooth, width, height, color, low, high)
            group = manim.VGroup(frame, raw_line, smooth_line)

            def locate(index: int) -> np.ndarray:
                x = frame.get_left()[0] + width * int(index) / max(1, len(smooth) - 1)
                y = frame.get_bottom()[1] + height * (
                    (smooth[int(index)] - low) / ((high - low) or 1.0)
                )
                return np.array([x, y, 0])

            return group, frame, raw_line, smooth_line, locate

        # ---------------------------------------------------- detection stages

        def _stage_signal(self, signals):
            """Colour -> grayscale -> gradient, all as in-place morphs of the
            persistent picture."""
            self.play(
                self._retitle("Step 1 - the image becomes a signal"),
                self._recaption(
                    f"Grayscale via {signals.grayscale_method}; "
                    "grid evidence lives in brightness changes, not colours."
                ),
            )
            self._swap_picture(
                self._array_to_mobject(signals.grayscale, self.picture.height)
            )
            self.wait(0.6)

            self.play(
                self._retitle("Step 2 - measure every adjacent-pixel change"),
                self._recaption(
                    "|pixel[x+1] - pixel[x]| : bright = a hard edge. "
                    "Upscaled 'fake' pixels leave edges on a regular lattice."
                ),
            )
            self._swap_picture(
                self._array_to_mobject(signals.gradient_h, self.picture.height)
            )
            self.wait(0.6)

        def _stage_collapse(self, signals, analysis):
            """Sweep a scan bar across the persistent picture while the 1D
            profile draws itself in sync -- the image is literally being
            summed into the curve."""
            pic = self.picture
            top, bottom = pic.get_top()[1], pic.get_bottom()[1]
            left, right = pic.get_left()[0], pic.get_right()[0]

            chart_x, frame_x, raw_x, smooth_x, self.locate_x = self._chart(
                signals.raw_profile_h, signals.profile_h, manim.BLUE_C
            )
            chart_x.move_to(manim.RIGHT * 2.9 + manim.UP * 1.7)
            label_x = manim.Text(
                "x profile: sum of each gradient column", font_size=19
            ).next_to(chart_x, manim.UP, buff=0.12)
            chart_y, frame_y, raw_y, smooth_y, self.locate_y = self._chart(
                signals.raw_profile_v, signals.profile_v, manim.GREEN_C
            )
            chart_y.move_to(manim.RIGHT * 2.9 + manim.DOWN * 0.9)
            label_y = manim.Text(
                "y profile: sum of each gradient row", font_size=19
            ).next_to(chart_y, manim.UP, buff=0.12)

            self.play(
                self._retitle("Step 3 - collapse the image into two 1D profiles"),
                self._recaption(
                    "A vertical scan bar sums columns; a horizontal one sums rows."
                ),
                manim.FadeIn(frame_x),
                manim.FadeIn(label_x),
            )
            bar_v = manim.Line(
                [left, bottom, 0], [left, top, 0], color=manim.BLUE_C, stroke_width=5
            )
            self.add(bar_v)
            self.play(
                manim.Create(raw_x),
                bar_v.animate.shift(manim.RIGHT * (right - left)),
                run_time=2.4,
                rate_func=manim.linear,
            )
            # The row sums read the *vertical* gradient: morph the picture to
            # that signal before the second sweep so what is summed is shown.
            self._swap_picture(
                self._array_to_mobject(signals.gradient_v, self.picture.height),
                manim.FadeOut(bar_v),
                manim.FadeIn(frame_y),
                manim.FadeIn(label_y),
                run_time=0.9,
            )

            bar_h = manim.Line(
                [left, top, 0], [right, top, 0], color=manim.GREEN_C, stroke_width=5
            )
            self.add(bar_h)
            self.play(
                manim.Create(raw_y),
                bar_h.animate.shift(manim.DOWN * (top - bottom)),
                run_time=2.4,
                rate_func=manim.linear,
            )
            self.play(manim.FadeOut(bar_h))

            self.play(
                self._retitle("Step 4 - smooth away single-pixel noise"),
                self._recaption(
                    f"Gaussian filter, sigma = {signals.smoothing_sigma:g}. "
                    "The raw sums stay behind in gray."
                ),
                raw_x.animate.set_stroke(opacity=0.35),
                raw_y.animate.set_stroke(opacity=0.35),
                manim.TransformFromCopy(raw_x, smooth_x),
                manim.TransformFromCopy(raw_y, smooth_y),
                run_time=1.4,
            )
            self.charts = manim.VGroup(chart_x, label_x, chart_y, label_y)
            self.wait(0.5)

        def _stage_peaks(self, analysis):
            """Peaks pop onto the curves, adjacent gaps get measured with
            braces, and the gap lengths tally into the histogram that elects
            the grid size."""
            h, v = analysis.horizontal, analysis.vertical
            self.play(
                self._retitle("Step 5 - find the repeating peaks"),
                self._recaption(
                    f"scipy find_peaks: distance >= {h.min_spacing}px, "
                    f"prominence >= {h.min_prominence:.1f}. Every red dot is one grid line."
                ),
            )
            dots_x = manim.VGroup(
                *[
                    manim.Dot(self.locate_x(p), radius=0.045, color=manim.RED_C)
                    for p in h.peaks
                ]
            )
            dots_y = manim.VGroup(
                *[
                    manim.Dot(self.locate_y(p), radius=0.045, color=manim.RED_C)
                    for p in v.peaks
                ]
            )
            self.play(
                manim.LaggedStart(
                    *[manim.FadeIn(d, scale=2.2) for d in (*dots_x, *dots_y)],
                    lag_ratio=0.04,
                ),
                run_time=2.0,
            )
            self.peak_dots = manim.VGroup(dots_x, dots_y)
            self.wait(0.4)

            # Measure a handful of adjacent gaps explicitly, then tally them.
            self.play(
                self._retitle("Step 6 - the distance between neighbours IS the grid"),
                self._recaption(
                    "Measure every adjacent gap, count how often each length occurs, "
                    "and elect the mode."
                ),
            )
            n_show = min(4, len(h.spacings))
            braces = manim.VGroup()
            gap_labels = manim.VGroup()
            baseline = self.locate_x(0)[1] - 0.28
            for i in range(n_show):
                a = self.locate_x(h.peaks[i])
                b = self.locate_x(h.peaks[i + 1])
                brace = manim.BraceBetweenPoints(
                    [a[0], baseline, 0], [b[0], baseline, 0], color=manim.YELLOW
                )
                label = manim.Integer(int(h.spacings[i]), font_size=22).next_to(
                    brace, manim.DOWN, buff=0.06
                )
                braces.add(brace)
                gap_labels.add(label)
            if n_show:
                self.play(
                    manim.LaggedStart(
                        *[
                            manim.AnimationGroup(manim.GrowFromCenter(br), manim.FadeIn(la))
                            for br, la in zip(braces, gap_labels)
                        ],
                        lag_ratio=0.4,
                    ),
                    run_time=1.8,
                )

            counts = ", ".join(
                f"{gap}px x{count}"
                for gap, count in sorted(h.spacing_counts, key=lambda kv: -kv[1])[:4]
            )
            tally = manim.Text(
                f"gap tally:  {counts or 'none'}\n"
                f"mode = {h.selected_spacing}px  (grid width candidate)",
                font_size=21,
                line_spacing=0.9,
                color=manim.BLUE_C,
            ).next_to(self.charts, manim.DOWN, buff=0.35)
            move_anims = [
                label.animate.move_to(tally.get_center() + manim.UP * 0.02).set_opacity(0)
                for label in gap_labels
            ]
            self.play(*move_anims, manim.FadeIn(tally), run_time=1.2)
            self.remove(*gap_labels)
            vertical_result = manim.Text(
                f"same procedure on the y profile -> mode = {v.selected_spacing}px",
                font_size=21,
                color=manim.GREEN_C,
            ).next_to(tally, manim.DOWN, buff=0.18)
            self.play(manim.FadeIn(vertical_result), manim.FadeOut(braces))
            self.wait(0.8)
            self.tally = manim.VGroup(tally, vertical_result)

        def _stage_confidence(self, analysis) -> bool:
            """Show the acceptance gates on screen; returns True if accepted."""
            h, v = analysis.horizontal, analysis.vertical
            verdict_ok = analysis.rejection_reason is None
            gate = manim.Text(
                f"consistency: x {h.spacing_consistency:.2f} / y {v.spacing_consistency:.2f}   "
                f"coverage: x {h.peak_coverage:.2f} / y {v.peak_coverage:.2f}\n"
                f"confidence = mean {analysis.average_confidence:.3f} "
                f"(need >= {analysis.min_confidence:g})    "
                f"cell aspect {analysis.aspect_ratio:.2f} (need 0.5..2.0)",
                font_size=20,
                line_spacing=0.9,
            )
            gate.next_to(self.tally, manim.DOWN, buff=0.3)
            if gate.width > 8.6:
                gate.scale_to_fit_width(8.6)
            verdict = manim.Text(
                "ACCEPTED" if verdict_ok else
                f"REJECTED: {str(analysis.rejection_reason).replace('_', ' ')}",
                font_size=26,
                weight="BOLD",
                color=manim.GREEN_C if verdict_ok else manim.RED_C,
            ).next_to(gate, manim.DOWN, buff=0.2)
            self.play(
                self._retitle("Step 7 - should we trust this grid?"),
                self._recaption(
                    "confidence = (spacing consistency + capped peak coverage) / 2, "
                    "averaged over both axes."
                ),
                manim.FadeIn(gate),
            )
            self.play(manim.FadeIn(verdict, scale=1.4))
            self.wait(1.0)
            self.gates = manim.VGroup(gate, verdict)
            return verdict_ok

        def _stage_phase(self, step, analysis, source_image):
            """The signature shot: a comb of grid lines slides across the
            persistent picture, its score updating live, until it clicks into
            the phase that lines up with the real lattice."""
            grid_w, grid_h, phase_x, phase_y = analysis.result
            scores = np.asarray(analysis.offset_x.scores, dtype=float)
            size = source_image.size

            # Bring the original colours back for the overlay: same image, same
            # place -- just its colour representation again.
            clear = [
                manim.FadeOut(self.charts),
                manim.FadeOut(self.peak_dots),
                manim.FadeOut(self.tally),
                manim.FadeOut(self.gates),
            ]
            self._swap_picture(
                self._to_mobject(source_image, self.picture.height),
                *clear,
                run_time=1.2,
            )
            self.play(
                self.picture.animate.move_to(manim.LEFT * 3.3 + manim.DOWN * 0.15).set(
                    height=4.6
                ),
                self._retitle("Step 8 - slide the comb until it clicks"),
                self._recaption(
                    f"Try every phase 0..{max(0, grid_w - 1)}: score = sum of the x profile "
                    f"at offset + k*{grid_w}. The lattice only lines up once."
                ),
            )

            tracker = manim.ValueTracker(0)

            def comb():
                offset = int(tracker.get_value())
                lines = manim.VGroup()
                bottom = self.picture.get_bottom()[1]
                top = self.picture.get_top()[1]
                for x in range(offset, size[0], max(1, grid_w)):
                    p = self._pixel_to_point(size, x, 0)
                    lines.add(
                        manim.Line(
                            [p[0], bottom, 0],
                            [p[0], top, 0],
                            color=manim.BLUE_C,
                            stroke_width=1.6,
                        )
                    )
                return lines

            def scorecard():
                offset = int(tracker.get_value())
                value = scores[offset] if offset < len(scores) else 0.0
                best = " <- best" if scores.size and offset == int(np.argmax(scores)) else ""
                text = manim.Text(
                    f"phase {offset:>3d}   score {value:,.0f}{best}",
                    font_size=22,
                    color=manim.YELLOW if best else manim.WHITE,
                )
                return text.next_to(self.picture, manim.DOWN, buff=0.22)

            comb_mob = manim.always_redraw(comb)
            score_mob = manim.always_redraw(scorecard)
            self.play(manim.FadeIn(comb_mob), manim.FadeIn(score_mob))
            if grid_w > 1:
                self.play(
                    tracker.animate.set_value(grid_w - 1),
                    run_time=min(3.2, 0.35 * grid_w),
                    rate_func=manim.linear,
                )
                self.play(tracker.animate.set_value(phase_x), run_time=0.9)
            self.wait(0.5)

            # The comb clicks in: freeze it, recolour it, and complete the grid
            # with the horizontal lines from the (identical) y-axis scan.
            final_comb = comb()
            self.add(final_comb)
            self.remove(comb_mob)
            rows = manim.VGroup()
            left, right = self.picture.get_left()[0], self.picture.get_right()[0]
            for y in range(phase_y, size[1], max(1, grid_h)):
                p = self._pixel_to_point(size, 0, y)
                rows.add(
                    manim.Line(
                        [left, p[1], 0], [right, p[1], 0],
                        color=manim.RED_C, stroke_width=1.4,
                    )
                )
            applied = (step.details["applied_offset_x"], step.details["applied_offset_y"])
            self.play(
                final_comb.animate.set_color(manim.RED_C).set_stroke(width=1.4),
                manim.Create(rows),
                manim.FadeOut(score_mob),
                self._retitle("Step 9 - the hidden lattice, recovered"),
                self._recaption(
                    f"cell {grid_w} x {grid_h}px, phase ({phase_x}, {phase_y}), "
                    f"sampling offset {applied} - drawn over the untouched original."
                ),
                run_time=1.4,
            )
            self.grid_lines = manim.VGroup(final_comb, rows)
            self.wait(1.0)

        def _stage_sampling(self, step):
            """Sample centres appear inside the lattice, one kernel is inspected
            up close, then the persistent picture collapses into the sprite."""
            details = step.details
            source = details["source"]
            size = source.size
            centers_x, centers_y = details["centers_x"], details["centers_y"]

            # Keep the dot count watchable: full lattice when small, a
            # representative row + column cross when large.
            cell_x, cell_y = details["representative_cell"]
            if len(centers_x) * len(centers_y) <= 700:
                pairs = [(x, y) for x in centers_x for y in centers_y]
            else:
                pairs = [(x, centers_y[cell_y]) for x in centers_x]
                pairs += [(centers_x[cell_x], y) for y in centers_y]
            dots = manim.VGroup(
                *[
                    manim.Dot(self._pixel_to_point(size, x, y), radius=0.028,
                              color=manim.YELLOW)
                    for x, y in pairs
                ]
            )
            self.play(
                self._retitle("Step 10 - one sample point per cell"),
                self._recaption(
                    "Each cell is reduced to the pixel at its centre "
                    f"(here showing {len(pairs)} of {len(centers_x) * len(centers_y)} centres)."
                ),
                manim.LaggedStart(*[manim.FadeIn(d, scale=3) for d in dots],
                                  lag_ratio=0.002),
                run_time=2.0,
            )

            # Zoom in on one kernel without cutting away: box + magnified callout.
            x0, y0, x1, y1 = details["kernel_box"]
            box = manim.Rectangle(
                width=abs(
                    self._pixel_to_point(size, x1, 0)[0]
                    - self._pixel_to_point(size, x0, 0)[0]
                ),
                height=abs(
                    self._pixel_to_point(size, 0, y1)[1]
                    - self._pixel_to_point(size, 0, y0)[1]
                ),
                color=manim.YELLOW,
                stroke_width=3.5,
            ).move_to(self._pixel_to_point(size, (x0 + x1) / 2, (y0 + y1) / 2))
            kernel_img = self._to_mobject(source.crop((x0, y0, x1, y1)), 1.9)
            kernel_img.move_to(manim.RIGHT * 2.1 + manim.UP * 0.9)
            zoom_lines = manim.VGroup(
                manim.DashedLine(box.get_corner(manim.UR), kernel_img.get_corner(manim.UL),
                                 stroke_width=1.5, color=manim.YELLOW),
                manim.DashedLine(box.get_corner(manim.DR), kernel_img.get_corner(manim.DL),
                                 stroke_width=1.5, color=manim.YELLOW),
            )
            color = details["sampled_color"]
            rgb = tuple(color[:3]) if isinstance(color, tuple) else (color,) * 3
            swatch = manim.Square(1.3, fill_opacity=1.0, stroke_width=2)
            swatch.set_fill(manim.rgb_to_color([c / 255 for c in rgb[:3]]))
            swatch.next_to(kernel_img, manim.RIGHT, buff=1.1)
            arrow = manim.Arrow(kernel_img.get_right(), swatch.get_left(), buff=0.15)
            median = manim.Text(details["sample_method"], font_size=19).next_to(
                arrow, manim.UP, buff=0.08
            )
            swatch_label = manim.Text(str(tuple(rgb)), font_size=19).next_to(
                swatch, manim.DOWN, buff=0.12
            )
            self.play(
                self._retitle("Step 11 - a 3 x 3 median beats a single pixel"),
                self._recaption(
                    "The centre pixel could be an upscaling artifact; the median of its "
                    "3 x 3 neighbourhood is robust to it."
                ),
                manim.Create(box),
                manim.FadeIn(kernel_img, zoom_lines),
            )
            self.play(manim.GrowArrow(arrow), manim.FadeIn(median),
                      manim.FadeIn(swatch, swatch_label))
            self.wait(1.2)

            # The collapse: the SAME picture crossfades into the sprite, then
            # every annotation it accumulated is no longer needed.
            output_mob = self._to_mobject(step.image, self.picture.height)
            output_mob.move_to(self.picture)
            cells_w, cells_h = step.image.size
            self.play(
                self._retitle("Step 12 - every cell collapses to one true pixel"),
                self._recaption(
                    f"{size[0]} x {size[1]} px  ->  {cells_w} x {cells_h} px. "
                    "Same picture - the fake resolution is simply gone."
                ),
                manim.FadeTransform(self.picture, output_mob),
                manim.FadeOut(dots),
                manim.FadeOut(self.grid_lines),
                manim.FadeOut(box),
                manim.FadeOut(zoom_lines),
                manim.FadeOut(kernel_img),
                manim.FadeOut(arrow),
                manim.FadeOut(median),
                manim.FadeOut(swatch),
                manim.FadeOut(swatch_label),
                run_time=1.8,
            )
            self.picture = output_mob
            # Beat: shrink towards true scale and back, to feel the size change.
            self.play(self.picture.animate.scale(0.28), run_time=0.9)
            self.wait(0.4)
            self.play(
                self.picture.animate.scale(1 / 0.28).move_to(manim.DOWN * 0.15),
                run_time=0.9,
            )

        # ------------------------------------------------------------ assembly

        def construct(self) -> None:
            self.heading = None
            self.caption = None
            steps = trace.steps
            grid_step = next((s for s in steps if s.kind == "grid"), None)
            sampling_step = next((s for s in steps if s.kind == "sampling"), None)

            # Intro: the image this whole video is about. It never leaves.
            first = steps[0].image if steps else Image.new("RGB", (2, 2))
            title = manim.VGroup(
                manim.Text("SpriteGrid", font_size=54, weight="BOLD"),
                manim.Text("watch one image find its true pixels", font_size=26),
            ).arrange(manim.DOWN, buff=0.2).to_edge(manim.UP, buff=0.7)
            self.picture = self._to_mobject(first, 4.2).shift(manim.DOWN * 0.9)
            self.play(manim.FadeIn(self.picture, shift=manim.UP * 0.3),
                      manim.Write(title))
            self.wait(0.9)
            self.play(
                manim.FadeOut(title),
                self.picture.animate.move_to(manim.LEFT * 4.05 + manim.UP * 0.55).set(
                    height=3.3
                ),
            )

            for step in steps:
                if step is grid_step:
                    analysis: GridDetectionAnalysis = step.details["analysis"]
                    signals = analysis.signals
                    if signals is None:
                        raise ValueError(
                            "Image walkthrough requires detector signal diagnostics"
                        )
                    self._stage_signal(signals)
                    self._stage_collapse(signals, analysis)
                    self._stage_peaks(analysis)
                    accepted = self._stage_confidence(analysis)
                    if accepted:
                        self._stage_phase(step, analysis, step.image)
                    else:
                        # Rejected: restore the original colours in place and
                        # clear the working annotations; downstream steps (if
                        # any) continue on the same picture.
                        self._swap_picture(
                            self._to_mobject(step.image, self.picture.height),
                            manim.FadeOut(self.charts),
                            manim.FadeOut(self.peak_dots),
                            manim.FadeOut(self.tally),
                            manim.FadeOut(self.gates),
                        )
                        self.play(
                            self.picture.animate.move_to(manim.DOWN * 0.15).set(
                                height=4.6
                            ),
                            self._recaption(
                                "No trustworthy lattice -> SpriteGrid refuses to "
                                "resample and passes the image through untouched."
                            ),
                        )
                        self.wait(1.2)
                elif step is sampling_step:
                    self._stage_sampling(step)
                elif steps and step is steps[0]:
                    continue  # the intro already presented the loaded image
                else:
                    # Post-processing (quantize, symmetry, crop, res, ...):
                    # each is one more in-place morph of the same picture.
                    self._swap_picture(
                        self._to_mobject(step.image, self.picture.height),
                        self._retitle(step.title),
                        self._recaption(
                            step.details.get(
                                "caption",
                                f"{step.image.width} x {step.image.height}  |  "
                                f"{step.image.mode}",
                            )
                        ),
                        run_time=1.1,
                    )
                    self.wait(0.7)

            outro = manim.Text(
                "Same pipeline. Same pixels. One image, start to finish.",
                font_size=26,
            ).to_edge(manim.DOWN, buff=0.9)
            end_anims = [manim.FadeIn(outro)]
            if self.caption is not None:
                end_anims.append(manim.FadeOut(self.caption))
                self.caption = None
            if self.heading is not None:
                end_anims.append(
                    self._retitle("The walkthrough only ever observed snapshots")
                )
            self.play(*end_anims)
            self.wait(1.4)

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
