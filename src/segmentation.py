from PIL import Image
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

import numpy as np
import matplotlib.pyplot as plt  # Ensure plt is imported


def generate_segment_masks(im_arr: np.ndarray) -> np.ndarray:
    h, w = im_arr.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    dataset = np.concatenate([im_arr.reshape(-1, 3), x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)

    # Use DBSCAN to cluster the pixels
    # clusterer = DBSCAN(eps=1, min_samples=100, metric="euclidean")
    # clusterer = KMeans(n_clusters=3)
    clusterer = GaussianMixture(n_components=3, covariance_type="full", random_state=0)
    labels = clusterer.fit_predict(dataset)

    label_mask = labels.reshape(h, w)

    if np.all(label_mask == -1):
        print("No background found.")
        return None

    return label_mask


def remove_background(image: Image.Image, debug=False) -> tuple[Image.Image, Image.Image | None]:
    im_arr = np.array(image)
    label_mask = generate_segment_masks(im_arr)

    if label_mask is None:
        print("No background found.")
        return image, None
    
    labels_flat = label_mask.flatten()
    # For now, let's assume that the background is the most common segment
    bg_id = np.bincount(labels_flat[labels_flat != -1]).argmax()  # Find the most common label (background)

    mask = (label_mask != bg_id).astype(np.uint8) * 255  # Create a mask for non-background pixels

    # Make image with alpha channel
    image = Image.fromarray(im_arr)
    alpha_channel = Image.fromarray(mask)
    image.putalpha(alpha_channel)

    debug_image = None
    if debug:
        clustered_mask = label_mask

        # Generate the debug visualization as a PIL image
        fig, ax = plt.subplots()
        ax.imshow(image.convert("RGBA"))  # Ensure the image is displayed correctly
        unique_labels = np.unique(label_mask)
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            mask = clustered_mask == label
            show_mask(mask, ax=ax, random_color=True)
        plt.axis("off")
        
        # Convert the matplotlib figure to a PIL image
        fig.canvas.draw()
        debug_image = Image.frombytes(
            'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_argb()
        )
        plt.close(fig)  # Close the figure to free memory

    return image, debug_image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, interpolation="none")  # Ensure proper rendering
