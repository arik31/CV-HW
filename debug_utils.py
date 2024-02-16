import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path


def plot_bbox_mask_and_ellipse(args, output_dir, box_dets=None, masks=None, ellipses=None, image=None):
    img_name = Path(args.image_path).name
    if image is None:
        image = cv2.imread(args.image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    if masks is not None:
        for mask in masks:
            plot_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    if ellipses is not None:
        for ellipse in ellipses:
            cx, cy, w, h, angle = ellipse["ellipse_params"]
            plot_ellipse(plt.gca(), cx, cy, w, h, angle)
    if box_dets is not None:
        for label, box, conf in box_dets:
            plot_box(box, plt.gca())
    plt.axis('off')
    suffix = 'bgmm' if args.classic_cv else 'dl'
    plt.savefig(f'{output_dir}/{img_name}_seg_{suffix}.png')
    plt.close()

def plot_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plot_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def plot_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def plot_ellipse(ax, cx, cy, w, h, angle):
    ellipse = Ellipse(xy=(cx, cy), width=w, height=h, angle=angle,
                      edgecolor='r', fc='None', lw=2)
    ax.add_patch(ellipse)