"""
ComfyUI custom nodes for SpriteGrid.

Install by symlinking this directory to ComfyUI's custom_nodes/:
    ln -s /path/to/spritegrid/src/spritegrid/comfyui ~/.comfyui/custom_nodes/spritegrid

Or copy this directory to custom_nodes/spritegrid.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
