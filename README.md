<div align="center">
  <a href="https://github.com/marksverdhei/spritegrid">
    <img alt="spritegrid" height="128px" src="https://raw.githubusercontent.com/marksverdhei/spritegrid/main/assets/logo/336x336.png">
  </a>

  <h1>SpriteGrid</h1>

  <p><strong>Turn AI-generated pixel art into real pixel art.</strong></p>

  <p>
    <a href="https://pypi.org/project/spritegrid/"><img src="https://img.shields.io/pypi/v/spritegrid?color=%2334D058&label=PyPI" alt="PyPI"></a>
    <a href="https://pypi.org/project/spritegrid/"><img src="https://img.shields.io/pypi/dm/spritegrid?color=%2334D058" alt="Downloads"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"></a>
  </p>
</div>

---

AI image models generate pixel art at high resolution with **misaligned pixels**, **grainy colors**, and **wrong resolution**. SpriteGrid detects the implicit pixel grid and downsamples to clean, single-color pixel art at its true resolution.

<div align="center">
  <img alt="Before and after SpriteGrid processing" height="400px" src="https://raw.githubusercontent.com/marksverdhei/spritegrid/main/assets/docs/comparison.png">
</div>

## Quick Start

```bash
pip install spritegrid
spritegrid ai_pixelart.png -o clean_sprite.png
```

---

## ComfyUI Node

SpriteGrid ships as a ComfyUI custom node, so you can plug it directly into your image generation workflows.

### Installation

```bash
# Option 1: Symlink (recommended for development)
ln -s /path/to/spritegrid/src/spritegrid/comfyui \
  /path/to/ComfyUI/custom_nodes/spritegrid

# Option 2: Copy
cp -r /path/to/spritegrid/src/spritegrid/comfyui \
  /path/to/ComfyUI/custom_nodes/spritegrid
```

Restart ComfyUI after installing. The **SpriteGrid** node appears under `image/sprite`.

### Node Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_grid` | 4 | Minimum grid cell size to detect (1-32) |
| `quantize` | 8 | Color quantization bits (4-8). Lower = tighter palette |
| `remove_background` | none | Remove background: `none`, `before`, or `after` detection |
| `crop` | false | Crop to non-transparent content after processing |

### Workflow

Connect any image generation output to the SpriteGrid node. It will:

1. Detect the pixel grid in the AI output
2. Downsample to the true pixel art resolution
3. Optionally remove background and crop

The node includes a **pixelated preview extension** that sets `image-rendering: pixelated` on all ComfyUI preview images, so your pixel art stays crisp instead of getting blurred by browser interpolation.

### Idempotence

SpriteGrid is idempotent &mdash; processing an already-clean image returns it unchanged. You can safely leave it in your workflow without worrying about double-processing.

---

## CLI

### Grid Detection & Downsampling

Detect the pixel grid in AI-generated art and downsample to clean pixels:

```bash
# Basic cleanup
spritegrid ai_pixelart.png -o clean_sprite.png

# Remove background + crop to content
spritegrid ai_pixelart.png -b -c -o sprite.png

# Debug mode: visualize detected grid overlay
spritegrid ai_pixelart.png -d -o debug.png
```

**Options:**

| Flag | Description |
|------|-------------|
| `-o, --output FILE` | Save output to file |
| `-b` | Remove background before processing |
| `-c` | Crop to content after processing |
| `-d` | Show debug grid overlay |
| `-i` | Display output image |
| `-a, --ascii SCALE` | Output as ANSI art |
| `-m, --min-grid N` | Minimum grid size (default: 4) |
| `-q, --quantize N` | Color bits: 4-8 (default: 8) |

### Sprite Extraction

Convert AI images into sprites at a target resolution:

```bash
# 32x32 sprite
spritegrid-crop ai_image.png -o sprite.png -s 32

# 64x48 with padding
spritegrid-crop ai_image.png -o sprite.png -s 64x48 -p 5

# Centered on exact canvas
spritegrid-crop ai_image.png -o sprite.png -s 64 --center
```

---

## Python API

```python
from spritegrid import process_sprite, crop_and_scale, crop_and_scale_centered
from PIL import Image

# Full pipeline: remove background + crop + scale
img = Image.open("ai_generated.png")
sprite = process_sprite(img, size=32, remove_bg=True)
sprite.save("sprite_32x32.png")

# Crop and scale preserving aspect ratio
sprite = crop_and_scale(img, target_size=64, padding=5)

# Center on exact canvas size
sprite = crop_and_scale_centered(img, target_size=64)

# Batch process a directory
from spritegrid import batch_process
batch_process("input_dir/", "output_dir/", size=32)
```

---

## How It Works

SpriteGrid uses signal processing to recover the true pixel grid from upscaled AI art:

1. **Gradient Analysis** &mdash; Computes horizontal and vertical gradient profiles across the image
2. **Peak Detection** &mdash; Finds dominant spacing using SciPy peak detection with confidence scoring
3. **Grid Validation** &mdash; Checks aspect ratio and confidence thresholds to reject false grids
4. **Idempotence Check** &mdash; If output dimensions match input, the image is already clean
5. **Geometric Median Sampling** &mdash; Each grid cell is downsampled to one pixel using Weiszfeld's algorithm for robust color selection
6. **Quantization** &mdash; Optional color depth reduction for tighter palettes

---

## Scaling Up for Display

SpriteGrid outputs pixel art at its true resolution (often very small). To display or use the output:

```bash
# ImageMagick (nearest-neighbor to preserve sharp pixels)
convert sprite.png -filter point -resize 800% sprite_large.png
```

```python
img = Image.open("sprite.png")
big = img.resize((img.width * 8, img.height * 8), Image.NEAREST)
```

---

## Tips for AI Pixel Art Generation

- **Flux** produces the cleanest grids and works best with SpriteGrid
- **SDXL/SD 1.5** need stronger negative prompts: `blurry, gradient, anti-aliased, smooth`
- Generate at **512x512 or lower** for more detectable grids
- Request **transparent backgrounds** when possible for easier sprite extraction
- Specify the target sprite size in your prompt: `"32x32 pixel art character"`

---

## Contributing

Contributions welcome. The codebase is structured as:

```
src/spritegrid/
  detection.py       # Grid detection (gradient analysis + peak detection)
  main.py            # Orchestration and output handling
  crop_and_scale.py  # Sprite extraction pipeline
  segmentation.py    # Background removal (DBSCAN clustering)
  utils.py           # Geometric median, ASCII conversion
  comfyui/           # ComfyUI custom node + pixelated preview extension
```

```bash
# Run tests
uv run pytest tests/

# Install for development
uv sync
```

---

## License

MIT
