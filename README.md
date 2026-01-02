<div align="center">
  <a href="https://github.com/marksverdhei/spritegrid">
    <img alt="spritegrid" height="200px" src="https://raw.githubusercontent.com/marksverdhei/spritegrid/main/assets/logo/336x336.png">
  </a>
</div>


# spritegrid  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Spritegrid is an image postprocessor for generative art. When general image generation models attempt to make pixel art, they often generate high-resolution images with janky pixels and grainy pixel colors. 

<img alt="example showing janky and grainy pixels" height="200px" src="https://raw.githubusercontent.com/marksverdhei/spritegrid/main/assets/docs/visualization.png">

1. Pixels can be janky and pixels can be incorrectly aligned (half-pixels etc).
2. Pixels are grainy and don't contain a single color.
spritegrid divides 

Spritegrid converts these images into a grid and generates the pixel art in its appropriate resolution:


<img alt="comparison before and after postprocessing" height="400px" src="https://raw.githubusercontent.com/marksverdhei/spritegrid/main/assets/docs/comparison.png">

As you can see, it works but it is not yet flawless. If you would like to contribute, hurry before I add some lame contribution guidelines!

---


## Installation

```bash
pip install spritegrid
```

## Usage

### Grid Detection & Downsampling

Use `spritegrid` to detect pixel grids in AI-generated pixel art and downsample to clean sprites:

Basic usage:
```bash
spritegrid ai_pixelart.png -o clean_sprite.png
```

With background removal:
```bash
spritegrid ai_pixelart.png -b -o clean_sprite.png
```

You can resize the output afterwards with imagemagick:
```bash
convert pixel-art.png -filter point -resize 400% pixel-art-large.png
```

### AI Image to Sprite Conversion

Use `spritegrid-crop` to convert AI-generated images (from Imagen, Flux, SDXL, etc.) into clean pixel art sprites:

**Command Line:**
```bash
# Convert to 32x32 sprite
spritegrid-crop ai_image.png -o sprite.png -s 32

# Convert to 64x48 sprite with padding
spritegrid-crop ai_image.png -o sprite.png -s 64x48 -p 5

# Center on exact canvas size
spritegrid-crop ai_image.png -o sprite.png -s 64 --center
```

**Python API:**
```python
from spritegrid import process_sprite, crop_and_scale
from PIL import Image

# Method 1: Full pipeline with AI background removal
img = Image.open("ai_generated.png")
sprite = process_sprite(img, size=32, remove_bg=True)
sprite.save("sprite_32x32.png")

# Method 2: Crop and scale without background removal
img = Image.open("character.png")
sprite = crop_and_scale(img, target_size=64, padding=5)
sprite.save("character_64.png")

# Method 3: Centered on canvas
from spritegrid import crop_and_scale_centered
sprite = crop_and_scale_centered(img, target_size=64)
sprite.save("centered_sprite.png")
```

**Options:**
- `-s, --size`: Target size (e.g., `32` for 32x32, or `64x48` for non-square)
- `-p, --padding`: Padding around content before scaling
- `--center`: Center sprite on exact canvas size
- `--no-aspect`: Force exact size without maintaining aspect ratio
- `--alpha-threshold`: Alpha threshold for transparency detection (default: 0)