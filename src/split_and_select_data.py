"""
split_and_select_data.py
Version: 1.0
Author: [Xiaomin Zhang]

This Python script splits raw hyper-stacks into stacks or single images as needed and selects 8 images from both control and sample datasets, totaling 16 for cells and 16 for nuclei for training. Additionally, it creates output directories if they do not already exist.

Input: raw data
- Exp1_HAP1_Pos03_Control.tif
- Exp1_HAP1_Pos06_Sample_CHIR.tif

Output:
- Stacks or single images that are separated or selected from the raw data.
- The 8 images from each dataset are selected by keeping the first and last timepoints’ image and maintaining a consistent interval of 13 between selected images.

Time taken: about 5 seconds.
"""

import os
import tifffile as tif
import time
import json

# Parameters
params = {
    "inputdir": "../data/rawdata",
    "outputdir":"../data",
    "interval": 13,
    "num_images": 8
}

# Start the timer
start = time.perf_counter()

# Define directories
input_folder = params['inputdir']
control_split_folder = os.path.join(params['outputdir'], 'splited_image', 'control')
sample_split_folder = os.path.join(params['outputdir'], 'splited_image', 'sample')
single_image_c01_control = os.path.join(params['outputdir'], 'splited_image', 'single_image','cytoplasm', 'control')
single_image_c01_sample = os.path.join(params['outputdir'], 'splited_image', 'single_image', 'cytoplasm', 'sample')
single_image_c02_control = os.path.join(params['outputdir'], 'splited_image', 'single_image','nucleus', 'control')
single_image_c02_sample = os.path.join(params['outputdir'], 'splited_image', 'single_image', 'nucleus','sample')
selected_image_c01 = os.path.join(params['outputdir'], 'selected_image', 'cytoplasm')
selected_image_c02 = os.path.join(params['outputdir'], 'selected_image', 'nucleus')

# Create output directories if they do not exist
for folder in [control_split_folder, sample_split_folder, single_image_c01_control, single_image_c01_sample,
               single_image_c02_control, single_image_c02_sample, selected_image_c01, selected_image_c02]:
    os.makedirs(folder, exist_ok=True)


def create_output_dirs(base_dir):
    """Create necessary output directories if they do not exist."""
    cells_dir = os.path.join(base_dir, "cells")
    nuclei_dir = os.path.join(base_dir, "nuclei")
    os.makedirs(cells_dir, exist_ok=True)
    os.makedirs(nuclei_dir, exist_ok=True)
    return cells_dir, nuclei_dir


def save_selected_images(channel, base_output_path, num_images=params['num_images']):
    """Split hyper-stack into individual images and select specific images based on interval."""
    total_images = channel.shape[0]
    interval = total_images // (num_images - 1)
    selected_indices = [i * interval for i in range(num_images - 1)] + [total_images - 1]

    for idx in selected_indices:
        output_path = base_output_path.replace('.tif', f'_t{idx + 1}.tif')
        selected_output_path = os.path.join(selected_image_c01 if 'c01' in output_path else selected_image_c02,
                                            os.path.basename(output_path))
        tif.imwrite(selected_output_path, channel[idx])


def process_files(input_folder):
    """Process each file in the input folder."""
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            filepath = os.path.join(input_folder, filename)
            image = tif.imread(filepath)

            # Split the channels
            cell_channel = image[:, 0, :, :]
            nucleus_channel = image[:, 1, :, :]

            # Determine the output filenames and folder based on whether 'Control' is in the filename
            is_control = 'Control' in filename
            output_folder = control_split_folder if is_control else sample_split_folder
            cell_output_path = os.path.join(output_folder, filename.replace('.tif', '_c01.tif'))
            nucleus_output_path = os.path.join(output_folder, filename.replace('.tif', '_c02.tif'))

            # Save the channels
            tif.imwrite(cell_output_path, cell_channel)
            tif.imwrite(nucleus_output_path, nucleus_channel)

            # Save the channels as sequences of images
            single_image_c01_folder = single_image_c01_control if is_control else single_image_c01_sample
            single_image_c02_folder = single_image_c02_control if is_control else single_image_c02_sample

            for idx in range(cell_channel.shape[0]):
                cell_image_path = os.path.join(single_image_c01_folder,
                                               filename.replace('.tif', f'_c01_t{idx + 1}.tif'))
                nucleus_image_path = os.path.join(single_image_c02_folder,
                                                  filename.replace('.tif', f'_c02_t{idx + 1}.tif'))
                tif.imwrite(cell_image_path, cell_channel[idx])
                tif.imwrite(nucleus_image_path, nucleus_channel[idx])

            # Save selected images from each channel
            save_selected_images(cell_channel, cell_output_path)
            save_selected_images(nucleus_channel, nucleus_output_path)


def main(params):
    """Main function to execute the script tasks."""
    process_files(input_folder)
    # End the timer
    end = time.perf_counter()
    elapsed = end - start
    print(f"Time taken: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main(params)