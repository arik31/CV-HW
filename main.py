import argparse
from q1 import q1_bgmm, q1_DL
from q2 import q2
from q3 import q3_non_local_means, q3_deblurGAN_swin2sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, help="path to the image you want to process")
    parser.add_argument("--q1", action='store_true', help='Fit ellipses to blood cell and report eccentricities and areas')
    parser.add_argument("--q2", action='store_true', help='Binarize input image and save it to args.out_dir')
    parser.add_argument("--q3", action='store_true', help='Deblur the chita image, save result to args.out_dir')
    parser.add_argument("--classic-cv", action='store_true', help='Use classic algorithm instead of DL')
    parser.add_argument("--csp-yolo-weights", type=str, default="weights/cst-yolo-weights.pt", help="relative path to CSP-YOLO weigths")
    parser.add_argument("--sam-weights", type=str, default="weights/sam_vit_h_4b8939.pth", help="relative path to SAM weigths")
    parser.add_argument("--deblur-gan-weights", type=str, default="weights/deblur-gan.pth", help="relative path to deblur-gan weigths")
    parser.add_argument("--swin2sr-weights", type=str, default="weights/Swin2SR_CompressedSR_X4_48.pth", help="relative path to Swin2SR weigths")
    parser.add_argument("--debug", action='store_true', help='Write debug image to args.out_dir')
    parser.add_argument("--out-dir", type=str, default='./outputs', help='output dir path')
    args = parser.parse_args()

    return get_answer(args)


def get_answer(args):
    if args.q1:
        if args.classic_cv:
            q1_bgmm(args)
        else:
            q1_DL(args)
    if args.q2:
        return q2(args)
    if args.q3:
        if args.classic_cv:
            return q3_non_local_means(args)
        else:
            return q3_deblurGAN_swin2sr(args)


if __name__ == '__main__':
    main()
