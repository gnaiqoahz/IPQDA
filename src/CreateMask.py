'''
CreateMask.py
Version: 1.0
Author: Qiang Zhao [q.zhao2@student.vu.nl]
Date: 30 Jun, 2024

This script will create measurement masks based on the segmented results from the trained cellpose model.

Input:
- input_directory: The directory containing the segmented images in two subfolders (cytoplasm and nuclei).

Output:
- Output_directory: The directory will contain three subfolders containing measurement masks for three compartments.

Time taken: ~2 minutes
'''


import os
import numpy as np
import cv2
import time
import argparse
from skimage import io, segmentation
from skimage.morphology import disk, erosion, dilation
from matplotlib import pyplot as plt

def visualize_masks(raw_nuclear, raw_cyto, nucleus, cytoplasm, cell_boundary):
    plt.figure(figsize=(15, 3))
    titles = ['Raw Nuclear Mask', 'Raw Cyto/Cell Mask', 'Processed Nucleus', 'Processed Cytoplasm',
              'Processed Cell Boundary']
    masks = [raw_nuclear, raw_cyto, nucleus, cytoplasm, cell_boundary]
    for i, (mask, title) in enumerate(zip(masks, titles), 1):
        plt.subplot(1, 5, i)
        plt.imshow(mask,cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def process_masks(nuclear_mask, cell_mask):

    # remove edge label
    nuclear_mask_clear = segmentation.clear_border(nuclear_mask)
    cell_mask_clear = segmentation.clear_border(cell_mask)

    # link nuclear and cell and only keep both positive signal
    cell_label = np.unique(cell_mask_clear)
    for i in cell_label:
        if i !=0:
            mapping_signal = nuclear_mask_clear[np.array(np.where(cell_mask_clear == i, cell_mask_clear, 0), dtype=bool)]
            if np.sum(mapping_signal) == 0:
                cell_mask_clear = np.where(cell_mask_clear != i, cell_mask_clear, 0)

    nuclear_label = np.unique(nuclear_mask_clear)
    for i in nuclear_label:
        if i !=0:
            mapping_signal = cell_mask_clear[np.array(np.where(nuclear_mask_clear == i, nuclear_mask_clear, 0), dtype=bool)]
            if np.sum(mapping_signal) == 0:
                nuclear_mask_clear = np.where(nuclear_mask_clear != i, nuclear_mask_clear, 0)

    #visualize_masks(nuclear_mask_clear,nuclear_mask,cell_mask_clear,cell_mask,cell_mask)

    selem = disk(4) #Optional

    # 01-nuclear
    eroded_nucleus = erosion(nuclear_mask_clear, selem)
    eroded_nucleus = eroded_nucleus > 0

    # 02-cytoplasm
    dilated_nucleus = dilation(nuclear_mask_clear, selem)
    dilated_nucleus_mask = dilated_nucleus > 0
    eroded_cell = erosion(cell_mask_clear, selem)
    eroded_cell_mask = eroded_cell > 0
    cytoplasm = np.logical_and(eroded_cell_mask, ~dilated_nucleus_mask)

    # 03-cell boundary
    cell_boundary = eroded_cell - cell_mask_clear
    cell_boundary = cell_boundary > 0

    # Retain the value for identifying cells
    return np.where(eroded_nucleus,nuclear_mask_clear,0), np.where(cytoplasm,cell_mask_clear,0), np.where(cell_boundary,cell_mask_clear,0)

def save_masks_as_png(n_mask, cyto_mask, cb_mask, directory, sample_id):

    subfolders = {'nucleus': n_mask,
                  'cytoplasm': cyto_mask,
                  'cell_boundary': cb_mask}

    if not os.path.exists(directory):
        os.makedirs(directory)

    for folder, mask in subfolders.items():
        subfolder_path = os.path.join(directory, folder)

        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        mask_uint8 = (mask.astype(np.uint8) * 255)
        file_path = os.path.join(subfolder_path,sample_id+'.png')
        cv2.imwrite(file_path, mask_uint8)
        #cv2.imwrite(file_path,mask)
        #image = Image.fromarray(mask)
        #image.save(file_path)

def run(nuc_dir, cell_dir, output_dir):

    nuc_files = [f for f in os.listdir(nuc_dir) if f.endswith('.png')]
    nuc_files.sort()
    cell_files = [f for f in os.listdir(cell_dir) if f.endswith('.png')]
    cell_files.sort()

    for nuc_file, cell_file in zip(nuc_files, cell_files):

        sample_id = nuc_file.replace('_c02', '').replace('_cp_masks.png', '')
        #print(sample_id)

        sample_id_cell = cell_file.replace('_c01', '').replace('_cp_masks.png', '')
        #print(sample_id_cell)
        # load mask
        nuclear_mask = io.imread(os.path.join(nuc_dir, nuc_file))
        cell_mask = io.imread(os.path.join(cell_dir, cell_file))

        '''
        nuclear_mask = io.imread('/Users/crystal_zhao/Desktop/IADA/groupproject/pre_model_output_selected/c02/Exp1_HAP1_Pos03_Control_c02_t1_cp_masks.png')
        cell_mask = io.imread('/Users/crystal_zhao/Desktop/IADA/groupproject/pre_model_output_selected/c01/Exp1_HAP1_Pos03_Control_c01_t1_cp_masks.png')
        output_directory = '/Users/crystal_zhao/Desktop/IADA/groupproject/splited_data/measure'
        '''
        # create measurement mask
        nucleus, cytoplasm, cell_boundary = process_masks(nuclear_mask, cell_mask)

        # plotting (for double check)
        # visualize_masks(nuclear_mask, cell_mask, nucleus, cytoplasm, cell_boundary)

        # save mask as png file
        save_masks_as_png(nucleus, cytoplasm, cell_boundary, output_directory, sample_id)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create Measurement masks')
    parser.add_argument('--input_directory', '-i', type=str, default='../results/trained_models_results/',  ###？
                        help='The directory of input segmented images')
    parser.add_argument('--output_directory', '-o', type=str, default='../data/measure/',
                        help='The directory for saving output mask pngs')
    args = parser.parse_args()

    start_time = time.time()

    #Sample
    cell_directory = args.input_directory + '/sample/cytoplasm/'
    nucleus_directory = args.input_directory + '/sample/nuclei/'
    output_directory = args.output_directory + '/sample/'
    run(nucleus_directory, cell_directory, output_directory)

    #Control
    cell_directory = args.input_directory + '/control/cytoplasm/'
    nucleus_directory = args.input_directory + '/control/nuclei/'
    output_directory = args.output_directory + '/control/'
    run(nucleus_directory, cell_directory, output_directory)

    #Timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time >= 60:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(f"Running Time: {minutes} mins {seconds:.2f} secs")
    else:
        print(f"Running Time: {elapsed_time:.2f} secs")
