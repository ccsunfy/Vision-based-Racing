import torch as th
from dataclasses import dataclass
import numpy as np
import cv2
from PIL import Image, ImageDraw


def get_video_frames(video_path, indices):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    frames = []
    frame_id = 1
    while True:
        # Read a frame
        ret, frame = cap.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            break

        # Append the frame to the list
        if frame_id in indices:
            frames.append(frame)
        frame_id += 1

    # Release the video capture object
    cap.release()
    return frames

def get_masks(name, indices):
    masks = []
    for i in indices:
        mask = np.load(f"{name}/{str(i).zfill(5)}.npy")
        mask = cv2.resize(mask, (mask.shape[1]*2, mask.shape[0]*2), interpolation=cv2.INTER_NEAREST)
        masks.append(mask)

    return masks


@dataclass
class Config:
    start: int = 0
    interval: int = 1
    end: int = 10
    name: str = ""

def merge(cfg: Config):
    # Example logic to read a .npy file
    indices = np.arange(cfg.start, cfg.end, cfg.interval, dtype=int)
    frames = get_video_frames(cfg.name+".mp4", indices)
    masks = get_masks(cfg.name, indices=indices)
    img = merge_masked_images(frames, masks)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    combined_image = Image.fromarray(img, mode="RGBA")
    combined_image.save(f"{cfg.name}.png")
    combined_image.show()
    test = 1


def merge_masked_images(images, masks):
    """
    Merge masked parts of n images into the last image with transparency
    to illustrate movement (e.g., of a drone).

    Args:
        images (list of PIL.Image): List of n images in sequence.
        masks (list of PIL.Image): List of n binary masks (same size as images),
                                   where white (255) indicates the region to extract.

    Returns:
        PIL.Image: Combined image showing transparent masked areas.
    """
    """
    Merge masked parts of n images (NumPy arrays) into the last image, 
    making masked areas transparent, to illustrate movement (e.g., of a drone).

    Args:
        images (list of np.ndarray): List of n images as NumPy arrays with shape (H, W, 3 or 4).
        masks (list of np.ndarray): List of n masks as NumPy arrays with shape (H, W), 
                                    where values are 0 or 255 (binary mask).

    Returns:
        np.ndarray: Combined image as a NumPy array with shape (H, W, 4) (RGBA).
    """
    if len(images) != len(masks):
        raise ValueError("Number of images and masks must be the same")

    # Convert images to RGBA if they are RGB
    images = [np.dstack((img, np.full(img.shape[:2], 255))) if img.shape[-1] == 3 else img for img in images]

    # Use the last image as the base (convert to float for blending)
    combined_image = images[-1].astype(np.float32)
    background = images[-1].astype(np.float32)

    for idx in range(len(images) - 1):  # Process all images except the last one
        img = images[idx].astype(np.float32)
        mask = np.clip(masks[idx], None ,1) # Normalize mask to [0, 1]
        mask = np.expand_dims(mask, axis=-1)  # Add a channel dimension to match (H, W, 1)

        # Make the masked region transparent by reducing alpha channel
        # transparent_layer = img * mask
        # transparent_layer[..., 3] *= (idx + 1) / len(images)  # Gradual transparency
        trans = ((idx + 1) / len(images)) ** 1.4
        # Blend the transparent masked area onto the combined image
        combined_image = combined_image * (1 - mask) + background * mask * (1-trans) + img * mask * trans

    # Clip values to valid range and convert back to uint8
    combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)

    return combined_image


def main():
    # h = Config(start=1, interval=6, end=55, name="hovering")
    # t = Config(start=30, interval=10, end=350, name="tracking")
    # l = Config(start=20, interval=10, end=220, name="landing")
    # r1 = Config(start=1, interval=2, end=200, name="racing_circle3D")
    # r2 = Config(start=1, interval=2, end=200, name="racing_ellipse")
    r3 = Config(start=1, interval=2, end=118, name="examples/ete_racing_sim/cc_straight")

    # merge(h)
    # merge(t)
    # merge(l)
    merge(r3)

if __name__ == "__main__":
    main()