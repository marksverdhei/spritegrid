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
    profile_h: np.ndarray,
    profile_v: np.ndarray,
    grid_w: int,
    grid_h: int,
    detected_offset_x: int,
    detected_offset_y: int,
    applied_offset_x: int,
    applied_offset_y: int,
) -> None:
    """Record the exact detector inputs and selected grid."""

    trace.add(
        "grid",
        "Discover grid and local candidates",
        image,
        profile_h=np.asarray(profile_h, dtype=float).copy(),
        profile_v=np.asarray(profile_v, dtype=float).copy(),
        grid_w=grid_w,
        grid_h=grid_h,
        detected_offset_x=detected_offset_x,
        detected_offset_y=detected_offset_y,
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
            heading = self._heading(step.title)
            picture = self._image(step.image, height=4.4).shift(manim.LEFT * 2.2)
            grid_w = int(step.details["grid_w"])
            grid_h = int(step.details["grid_h"])
            overlay = manim.VGroup()
            if grid_w > 0 and grid_h > 0:
                left, right = picture.get_left()[0], picture.get_right()[0]
                bottom, top = picture.get_bottom()[1], picture.get_top()[1]
                phase_x = int(step.details["detected_offset_x"]) % grid_w
                phase_y = int(step.details["detected_offset_y"]) % grid_h
                for x in range(phase_x or grid_w, step.image.width, grid_w):
                    sx = left + (x / step.image.width) * (right - left)
                    overlay.add(
                        manim.Line(
                            [sx, bottom, 0],
                            [sx, top, 0],
                            color=manim.RED,
                            stroke_width=1,
                        )
                    )
                for y in range(phase_y or grid_h, step.image.height, grid_h):
                    sy = top - (y / step.image.height) * (top - bottom)
                    overlay.add(
                        manim.Line(
                            [left, sy, 0],
                            [right, sy, 0],
                            color=manim.RED,
                            stroke_width=1,
                        )
                    )

            profile_h = np.asarray(step.details["profile_h"])
            profile_v = np.asarray(step.details["profile_v"])
            h_spark = (
                manim.Sparkle()
                if profile_h.size == 0
                else self._sparkline(profile_h, 3.6, 1.1)
            )
            v_spark = (
                manim.Sparkle()
                if profile_v.size == 0
                else self._sparkline(profile_v, 3.6, 1.1)
            )
            charts = (
                manim.VGroup(
                    manim.Text("horizontal edge profile", font_size=20),
                    h_spark,
                    manim.Text("vertical edge profile", font_size=20),
                    v_spark,
                )
                .arrange(manim.DOWN, buff=0.2)
                .shift(manim.RIGHT * 3.0)
            )
            if grid_w and grid_h:
                detected = (
                    step.details["detected_offset_x"],
                    step.details["detected_offset_y"],
                )
                applied = (
                    step.details["applied_offset_x"],
                    step.details["applied_offset_y"],
                )
                result_text = (
                    f"selected cell: {grid_w} x {grid_h}px   "
                    f"detected phase: {detected}   sampling offset: {applied}"
                )
            else:
                result_text = "no reliable repeating grid selected"
            result = self._caption(result_text)
            self.play(manim.Write(heading), manim.FadeIn(picture), manim.Create(charts))
            if len(overlay) > 0:
                self.play(manim.Create(overlay))
            self.play(manim.FadeIn(result))
            self.wait(1.0)
            self.play(
                *[manim.FadeOut(m) for m in (heading, picture, overlay, charts, result)]
            )

        def _sparkline(self, values: np.ndarray, width: float, height: float):
            values = np.asarray(values, dtype=float)
            span = float(np.ptp(values))
            normalized = (values - float(values.min())) / (span or 1.0)
            points = [
                np.array(
                    [
                        -width / 2 + width * i / max(1, len(values) - 1),
                        -height / 2 + height * y,
                        0,
                    ]
                )
                for i, y in enumerate(normalized)
            ]
            line = manim.VMobject(color=manim.BLUE_C)
            line.set_points_as_corners(points)
            return line

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
            caption = self._caption(f"local 3 x 3 median -> output colour {color}")
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
