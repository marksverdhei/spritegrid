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

1.  **Prerequisites:**
    * Python 3.7 or higher is recommended.
    * `pip` or `uv` for installing packages.

2.  **Get the Code:** Clone this repository or download the source files (`main.py`, `detection.py`, etc.).
    ```bash
    git clone <repository-url> # Or download ZIP
    cd spritegrid # Navigate to the project directory
    ```

3.  **Install Dependencies:** Install the required Python libraries using pip or uv:
    ```bash
    # Using pip
    pip install Pillow requests numpy scipy

    # Or using uv
    uv pip install Pillow requests numpy scipy
    ```
    *(Alternatively, if a `requirements.txt` or configured `pyproject.toml` is provided, use that).*

## Usage

Run the tool from your terminal using the `main.py` script.

```bash
python main.py [OPTIONS] <image_source>
```