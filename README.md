# HW solution

## General
This project is my solutions to the HW assignment.
### Project structure
#### Important part
q1.py - script with 2 solutions to q1. One is based on BGMM, and the other is based on CST-YOLO & SAM.  
q2.py - script with a solution to q2, using Otsu method.  
q3.py - script with 2 solutions to q3. One using Non-Local Means, and the other using DeblureGAN & Swin2SR.  
q4.jpg - detailed solution to q4  
main.py - runner script for all solutions. See usage section for further instructions.

#### Submodules
CST-YOLO  
DeblureGAN-pytorch  
swin2sr  

All the above are to be treated as black-boxes. I have made only the minimum changes in them to make them work.
#### Utilities
debug_utils - plotting functions  
weights - folder for all the relevant weight files  
output - all the debug output will be written here  
readme_images

## Usage
Implementations of my solutions to q1, q2 and q3 are in q1.py, q2.py and q3.py.  
main.py was build to run each logic using CL flags.  
Use the --debug flag to write results to ./outputs.
q1 and q3 both has 2 implemetations, a DL approach and a classic computer vision approach.  
The default is the DL approach. To use the classic one, use the --classic-cv flag.  
To use q1,q2 or q3 logics within your code, use the standalone runner. There is one in each qX.py file.  
Example usage for each question: 
```
python home_test.py --image-path BLOOD_CELL_PATH --q1 --debug
python home_test.py --image-path BLOOD_CELL_PATH --q1 --classic-cv --debug
```
```
python home_test.py --image-path BLOOD_CELL_PATH --q2 --debug
```
```
python home_test.py --image-path ./chita.jpg --q3 --debug
python home_test.py --image-path ./chita.jpg --q3 --classic-cv --debug
```
q4 is answered at the end of this file

## Setup
1. pip install -r requirments.txt
2. Download all 4 .pth files. If you choose to sae the files in a different place or with a different name,  
   You mast explicitly add thier path with the relevant flag, e.g. --sam-weights PATH_TO_SAM_WEIGHTS.  
### Download instructions per model

#### SAM
Download SAM weights from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  
Save as ```weights/sam_vit_h_4b8939.pth```

#### DeblurGAN
Download from https://drive.google.com/file/d/1DWtz9eVf4xWrdtTmwVPZhHHqwU7j5tGQ/view?usp=drive_link  
Save as ```weights/deblur-gan.pth```

#### Swin2SR
Download from https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_CompressedSR_X4_48.pth
Save as ```weights/Swin2SR_CompressedSR_X4_48.pth```

#### CST-YOLO
Download from  https://drive.google.com/file/d/1jKIVJ7Fp5U8j2ACPgTrNmQ3Oa24rl7fC/view?usp=drive_link
Save as ```weights/cst-yolo-weights.pt```

## Q1
BGMM is classic for this type of question, but as I was given here a labeled dataset I've also implemented a DL based solution.
The code support both implementations. Default is the DL solution, to run the BGMM, use the --classic-cv flag 

### BGMM
1. Downscale the input image X4 to speed up the BGMM runtime
2. Turn to grayscale
3. Threshold the image with Otsu
4. Relevant pixels are black ones. Arrange all their indexes in a N*2 matrix
5. Run BGMM using sklearn. Set max component number to 50
6. Use the model to cluster the pixel indexes
7. For each cluster:  
   8. Make a mask image of this cluster  
   9. Fit an ellipse to it using cv2.convexHull & cv2.FitEllipse

#### BGMM results
<img src="readme_images/BloodImage_00007.jpg_seg_gmm.png">

### DL Solution: CST-YOLO + SAM
#### CST-YOLO
From a brief search in the net I came across this CST-YOLO github project https://github.com/mkang315/CST-YOLO,
which looks like a perfect fit.  
I've chosen it due to its report on high mAP, and its user-friendly code.
No pretrained models were provided here, so I had to train it myself.  
I could only fit batch_size=8 on my GPU, so I got 0.5mAP of 0.904,  
a little lower than the 0.927 they report on the project's page, but still a great detector.
The code, almost unmodified is in ./CST-YOLO.

#### SAM
Using https://github.com/facebookresearch/segment-anything with the CST-YOLO results as prompts produced good segmentations.
I've chosen it because it does great out-of-the-box segmentation, and it's super user-friendly. 

#### Eliipses
Once I had instance segmentations of the blood cells, I took each instance mask as binary image,
and used cv2.convexHull & cv2.fitEllipse to find it's best fit.
The main issue I've found in this pipeline are some miss detections of CSV-YOLO.
If I had the time and resorces, I would improve it by:
1. Training it with a larger batch size
2. The dataset labels aren't perfect. I would iteratively use the trained model to spot issues in the train dataset, fix and re-train

### DL Solution: Examples from the test
#### BloodImage_00011
<img src="readme_images/BloodImage_00011.jpg_seg.png">

#### BloodImage_00015
<img src="readme_images/BloodImage_00015.jpg_seg.png">

## Q2

Using Otsu gave good binary images, so I didn't try anything extra.
### Original
<img src="readme_images/BloodImage_00007.jpg">

### Binary
<img src="readme_images/binary.png">

## Q3

Here I've tried Non-Local Means Denoising and got interesting result, but I wanted something better.  
I had some hard time with recent Deblur solutions, many don't have pretrained models or has an unstable code.    
I've ended up with a combination of DeblurGAN and Swin2SR.  
DeblurGAN is an old project which work well on small images but fails miserably with big ones.  
Swin2SR was an ECCV 2022 paper. It is build on Swin Transformer v2, and was trained to SR compressed images.  
I've combine those 2 as follows:
1. Downscale the input X4
2. Use DeblureGAN on the small image
3. Use Swin2SR to upscale it back  

The code support both implementations.  
The default is DeblurGAN + Swi2SR.  
To use Non-Local Means, run with --classic-cv flag
### DeblurGAN + Swin2SR
<img src="readme_images/deblur_small_Swin2SR.png">

### FastNlMeansDenoisingColored
<img src="readme_images/Deblurred.png">

### Original image
<img src="readme_images/chita.jpg">

## Q4

A is not a plane, but a line.  
A = (1+a, 2+a, -2-2a)  
B and C intersection vector = (-2t, 1+2t, t)   
point of A,B,C intersection is (0, 1, 0)  

See detailed proof in q4.jpg



