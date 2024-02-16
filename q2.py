import cv2
import os


def q2(args):
    """Binarize input image using Otsu"""
    orig_img = cv2.imread(args.image_path)
    img_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    _, im_bw = cv2.threshold(img_gray, 100, 255, cv2.THRESH_OTSU)
    im_bw = im_bw.astype(bool)
    if args.debug:
        os.makedirs(args.out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(args.out_dir, 'binary.png'), im_bw.astype(int)*255)
    return im_bw


def q2_standalone(image_path, debug=False, out_dir=None):
    """Runner for q2 logic as a standalone function"""
    class q2Args:
        def __init__(self):
            self.image_path = image_path
            self.debug = debug
            self.out_dir = out_dir
    args = q2Args()
    return q2(args)
