import os
import cv2

def q3_deblurGAN_swin2sr(args):
    """Deblur the chita image using DeblurGAN and Swin2SR. Default option"""
    I = cv2.imread(args.image_path)
    # Downscale to X4 smaller
    H, W, _ = I.shape
    J = cv2.resize(I, (W // 4, H // 4))

    # Deblur using DeblurGAN
    os.chdir('./DeblurGAN-pytorch')
    os.makedirs('input', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    cv2.imwrite('./input/chita_blurred.jpg', J)
    os.system(f"python deblur_image.py --blurred ./input --deblurred ./output --resume {args.deblur_gan_weights}")
    deblur_small_image = cv2.imread('./output/deblurred chita_blurred.jpg')
    os.chdir("..")

    # Upscale using Swin2SR
    os.chdir('./swin2sr')
    os.makedirs('input', exist_ok=True)
    cv2.imwrite('./input/deblur_small.jpg', deblur_small_image)
    os.system(f"python main_test_swin2sr.py --task compressed_sr --scale 4 --training_patch_size 48 --model_path  {args.swin2sr_weights} --folder_lq ./input --save_img_only")
    result = cv2.imread('./results/swin2sr_compressed_sr_x4/deblur_small_Swin2SR.png')
    os.chdir("..")
    if args.debug:
        cv2.imwrite(os.path.join(args.out_dir, 'Deblurred.png'), result)
    return result


def q3_non_local_means(args):
    """Use opencv non-local means to deblur the chita"""
    I = cv2.imread(args.image_path)
    result = cv2.fastNlMeansDenoisingColored(I, None, 20, 20, 7, 21)
    if args.debug:
        cv2.imwrite(os.path.join(args.out_dir, 'Deblurred_nlm.png'), result)
    return result


def q3_standalone(image_path, debug=False, out_dir=None, classic_cv=False, deblur_gan_weights=None, swin2sr_weights=None):
    """Runner for q3 logic as a standalone function"""
    class q3Args:
        def __init__(self):
            self.image_path = image_path
            self.debug = debug
            self.out_dir = out_dir
            self.deblur_gan_weights = deblur_gan_weights
            self.swin2sr_weights = swin2sr_weights
    args = q3Args()
    if classic_cv:
        return q3_non_local_means(args)
    else:
        return q3_deblurGAN_swin2sr(args)
