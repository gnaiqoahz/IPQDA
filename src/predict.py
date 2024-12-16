import argparse
from cellpose import models, io, utils
import os
from roifile import ImagejRoi, roiwrite
import time
import torch

"""
model_training_ori.py
Version: 3.0
Author: [Qianqian WANG]

Description:
    This script utilizes the Cellpose model to perform segmentation of cell images.
    It generates masks and Regions of Interest (ROIs) suitable for use in ImageJ.

Arguments:
    -t, --image_type  : Type of image to process, either 'cytoplasm' or 'nuclei'. Required.
    -mt, --model_type : Type of model to use for segmentation, either 'pretrain' or 'trained'. Required.
    Optional:
    -i, --input       : Path to the input image directory or file.
    -o, --output      : Path to the output directory where segmentation results will be saved.
    -m, --model       : Path to a specific model file to use for segmentation.

Input:
    The Cellpose model and the image that needed to be segmented
    
Output:   
    The segmentation masks files and Regions of Interest (ROIs) suitable for use in ImageJ.
    
Example usage:
    python predict.py -t nuclei -mt pretrain
"""

def parse_arguments():
    """
    Parse command line arguments for the Cellpose segmentation script.

    Arguments:
    -t, --image_type  : Type of image to process, either 'cytoplasm' or 'nuclei'. Required.
    -mt, --model_type : Type of model to use for segmentation, either 'pretrain' or 'trained'. Required.
    Optional:
    -i, --input       : Path to the input image directory or file.
    -o, --output      : Path to the output directory where segmentation results will be saved.
    -m, --model       : Path to a specific model file to use for segmentation.

    Description:
    This script utilizes the Cellpose model to perform segmentation of cell images.
    It generates masks and Regions of Interest (ROIs) suitable for use in ImageJ.

    Example usage:
    python predict_ori.py -t nuclei -mt pretrain
    """
    parser = argparse.ArgumentParser(description="This script utilizes the Cellpose model to perform segmentation of "
                                                 "cell images. It generates masks and Regions of Interest (ROIs) "
                                                 "suitable for use in ImageJ.")
    # add parameters
    parser.add_argument('-t', '--image_type', choices=['cytoplasm', 'nuclei'], required=True,
                        help="Type of image to process (must be 'cytoplasm' or 'nuclei')")
    parser.add_argument('-mt', '--model_type', choices=['pretrain', 'trained'], required=True,
                        help="Type of model to process (must be 'pretrain' or 'trained')")

    parser.add_argument('-i', '--input', dest='input_file_dir', help='Path to the input file')
    parser.add_argument('-o', '--output', dest='output_file_dir', help='Path to the output file')
    parser.add_argument('-m', '--model', type=str,help='Path to the model file')
    parser.add_argument('-gpu', dest='gpu', default=False, type=bool,
                        help='Flag indicating whether to use GPU for prediction')
    args = parser.parse_args()

    # check whether the computer have gpu.
    if args.gpu:
        if not torch.cuda.is_available():
            device = torch.device('cpu')
            args.gpu = False

    return args


def new_save_rois(masks, file_name):
    """
    This function override the original save_rois function in Cellpose.io, to deal with the ValueError: zero-size
    array to reduction operation minimum which has no identity.

    Input:
    masks: the mask file from the model prediction.
    file_name: the name of image, which will used in the output file name.
    """
    outlines = utils.outlines_list(masks)
    valid_outlines = [outline for outline in outlines if outline.size > 0]
    rois = [ImagejRoi.frompoints(outline) for outline in valid_outlines]
    file_name = os.path.splitext(file_name)[0] + "_rois.zip"
    if os.path.exists(file_name):
        os.remove(file_name)
    roiwrite(file_name, rois)


def predict(args):
    """
    This function is the main function to perform segmentation of cell images.

    Input:
    args： the parameters from command line
    """

    # Get the input files
    if args.input_file_dir:
        folder_path = args.input_file_dir
    else:
        folder_path = '../data/splited_image/single_image/' + args.image_type
    sub_folder_path = os.path.join(folder_path, 'control')
    files = [f for f in os.listdir(sub_folder_path) if f.endswith('.tif')]
    sub_folder_path = os.path.join(folder_path, 'sample')
    files.extend([f for f in os.listdir(sub_folder_path) if f.endswith('.tif')])

    # Get the output path
    if args.output_file_dir:
        output_path = args.output_file_dir
    else:
        output_path = '../results/' + args.model_type + '_models_results/' + args.image_type


    # choose correct models
    if args.model:
        model = models.CellposeModel(gpu=args.gpu, model_type=args.model)
    else:
        if args.model_type == 'pretrain':
            if args.image_type == 'nuclei':
                model = models.CellposeModel(gpu=args.gpu, model_type='nuclei')
            elif args.image_type == 'cytoplasm':
                model = models.CellposeModel(gpu=args.gpu, model_type='cyto2')
            else:
                print('model is unclear, please check image_type and model_type')
        elif 'trained' in args.model_type:
            model_name = '../results/models/trained_' + args.image_type
            model = models.CellposeModel(gpu=args.gpu, model_type=model_name)
        else:
            print('model is unclear, please check image_type and model_type')

    # Use model to perform segmentation of cell images
    index = 0
    for file_name in files:
        # Show which image is current working on
        index = index + 1
        print(f'deal with image {index}: {file_name}')
        # Get the image path and the output file path
        file_path = os.path.join(folder_path, file_name)
        output_name = os.path.join(output_path, file_name)
        # load the input image
        images = io.imread(file_path)
        # perform segmentation of cell images
        if args.model_type == 'trained':
            masks, flows, styles = model.eval(images, channels=[0, 0])
        else:
            if args.image_type == 'cytoplasm':
                masks, flows, styles = model.eval(images, channels=[0, 0], diameter=100)
            else:
                masks, flows, styles = model.eval(images, channels=[0, 0], diameter=90)
        # save the mask files
        io.save_masks(images, masks, flows, file_names=output_name, png=True)
        # save the rois files
        try:
            io.save_rois(masks, file_name=output_name)
        except:
            new_save_rois(masks, file_name=output_name)


if __name__ == "__main__":
    parameters = parse_arguments()
    # record the running time
    start_time = time.time()
    predict(parameters)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Finish the image segmentation. Running time is {elapsed_time} seconds')