import json
import os
import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from pathlib import Path
from sklearn.mixture import BayesianGaussianMixture
from debug_utils import plot_bbox_mask_and_ellipse


def q1_DL(args):
    """Identifies each red blood cell using CST-YOLO,
    and returns its eccentricity and area. """

    # Bounding Box detection using CST-YOLO
    bboxs = bbox_detection(args.image_path, args.csp_yolo_weights)
    # Instance Segmentation using SAM with bboxs as prompts
    masks = segment(args.image_path, bboxs, args.sam_weights)
    # Fit ellipses
    ellipses = []
    for mask_idx in range(masks.shape[0]):
        I = np.zeros((480, 640), dtype=bool)
        M = np.array(masks[mask_idx, :, :, :].squeeze().cpu())
        I[M] = True
        ellipses.append(get_ellipse_params(I))
    if args.debug:
        os.makedirs(args.out_dir, exist_ok=True)
        plot_bbox_mask_and_ellipse(args, args.out_dir, box_dets=bboxs, masks=masks, ellipses=ellipses)
    return ellipses

def q1_bgmm(args):
    """Use Bayesian Gaussian Mixture Module to segment the blood cells"""
    orig_img = cv2.imread(args.image_path)
    H, W, _ = orig_img.shape
    # Down scale to get reasonable BGM runtime
    resize_factor = 4
    resized = cv2.resize(orig_img, (W // resize_factor, H // resize_factor))
    # IM -> gray -> thresh -> relevant pixels matrix
    img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, im_bw = cv2.threshold(img_gray, 100, 255, cv2.THRESH_OTSU)
    im_bw = im_bw.astype(bool)
    pixel_idxs = np.array(np.where(im_bw == 0)).T
    # Use BGM to cluster the pixels
    bgm = BayesianGaussianMixture(n_components=50, random_state=42).fit(pixel_idxs)
    segmented_labels = bgm.predict(pixel_idxs)

    # Fit ellipses
    ellipses = []
    unique_labels = np.unique(segmented_labels)
    for label in unique_labels:
        # Make a mask of the pixels with the current label. Fit an ellipse to it.
        label_mask = segmented_labels == label
        masked_pixels = pixel_idxs[label_mask, :]
        I = np.zeros_like(im_bw)
        I[masked_pixels[:, 0], masked_pixels[:, 1]] = True
        ellipses.append(get_ellipse_params(I, resize_factor))
    if args.debug:
        os.makedirs(args.out_dir, exist_ok=True)
        plot_bbox_mask_and_ellipse(args, args.out_dir, box_dets=None, masks=None, ellipses=ellipses)
    return ellipses


def get_ellipse_params(mask_img, resize_factor=1):
    points = np.column_stack(np.where(mask_img.T > 0))
    hull = cv2.convexHull(points)
    ((cx, cy), (w, h), angle) = cv2.fitEllipse(hull)
    a, b = w / 2, h / 2
    eccentricity = np.sqrt(1 - a ** 2 / b ** 2) if b > a else np.sqrt(1 - b ** 2 / a ** 2)
    area = np.sum(mask_img)
    result = {"ellipse_params": [cx * resize_factor, cy * resize_factor, w * resize_factor, h * resize_factor, angle],
              "eccentricity": eccentricity,
              "segmentation_area": area * (resize_factor ** 2)}
    return result


def bbox_detection(img_path, csp_yolo_ckp):
    """
    Use CST-YOLO to detect bboxs n the input image.
    As CST-YOLO is research-level project, I had to wrap it with a system call and write-read results from the disk
    """
    bbox_json_path = f"{os.getcwd()}/cst-yolo-output.json"
    os.chdir('./CST-YOLO')
    csp_yolo_ckp = f'../{csp_yolo_ckp}'
    os.system(f"python -m detect --weights {csp_yolo_ckp} --source {img_path} --nosave --json-path {bbox_json_path}")
    os.chdir("..")
    with open(bbox_json_path) as inf:
        bboxs = json.load(inf)
    os.remove(bbox_json_path)
    print(f'******* Done BBox Detection *******')
    return bboxs[Path(img_path).name]

def segment(img_path, bboxs_dets, sam_ckp):
    """segment the input image. If there are input boxes, use them as prompts"""
    # setup SAM predictor
    image = cv2.imread(img_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckp)
    sam.to(device=device)

    if bboxs_dets is not None:
        # Use boxes as prompts
        boxes = [det[1] for det in bboxs_dets]
        input_boxes = torch.tensor(boxes, device=device)
        predictor = SamPredictor(sam)
        predictor.set_image(image)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
    else:
        # Segment without prompts
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)
    print(f'******* Done Segmentation *******')
    return masks

def q1_standalone(image_path, debug=False, out_dir=None, classic_cv=False, csp_yolo_weights=None, sam_weights=None):
    """Runner for q1 logic as a standalone function"""
    class q1Args:
        def __init__(self):
            self.image_path = image_path
            self.debug = debug
            self.out_dir = out_dir
            self.csp_yolo_weights = csp_yolo_weights
            self.sam_weights = sam_weights
    args = q1Args()
    if classic_cv:
        return q1_bgmm(args)
    else:
        return q1_DL(args)
