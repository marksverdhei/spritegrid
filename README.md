# SpriteGrid

This project is vibe-coded using Gemini-2.5 pro.

See VIBELOG.md for chat links.  

---
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SpriteGrid is a Python tool designed to analyze AI-generated "pixel art" images. These images often mimic the pixel art style but lack a true, consistent grid structure due to the nature of generative models. Edges can be uneven ("janky"), and areas that should be a single color might be noisy or grainy.

SpriteGrid attempts to detect the *intended* underlying grid dimensions (the width and height of the conceptual pixels) within these imperfect images. It can then generate a "cleaned", downsampled image based on this detected grid or provide a visual overlay for debugging purposes.

## Features

* Detects grid dimensions (cell width & height) from AI-generated pixel art.
* Handles common image formats (PNG, JPG, BMP, etc.) via Pillow.
* Accepts input images from local file paths or URLs.
* Generates a downsampled output image by sampling the center pixel of each detected grid cell.
* Provides a debug mode to visualize the detected grid lines overlaid on the original image.
* Command-line interface for easy use.
* Tunable parameters for the detection algorithm.

## Installation

Prerequisites: `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```  

```bash
git clone https://github.com/marksverdhei/spritegrid.git
```

```bash
uv sync
source .venv/bin/activate
```


## Usage

Run the tool from your terminal using the `main.py` script.

```bash
python main.py [OPTIONS] <image_source>
```