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

Basic
```bash
spritegrid assets/examples/centurion.png -o centurion.png
```

With background removal
```bash
spritegrid assets/examples/centurion.png -b -o centurion.png
```

You can resize the image afterwards with, e.g. imagemagick
```bash
convert pixel-art.png -filter point -resize 400% pixel-art-large.png
```