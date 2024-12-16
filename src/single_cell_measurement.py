"""
single_cell_measurement.py
Version: 1.0
Author: [Xiaomin Zhang]

This Python script measures and plots the mean intensity of individual cells over time. It can generate plots for both normalized mean intensity and raw mean intensity over time.

Input:
These two files are the raw data of the cell channel of the sample and control. They are used to provide the original intensity.
- Exp1_HAP1_Pos06_Sample_CHIR_c01.tif
- Exp1_HAP1_Pos03_Control_c01.tif

These two files below are used to plot the whole cell’s intensity over time:
- CorrectedTracking_control_cell.tif
- CorrectedTracking_sample_cell.tif

These two files below are used to plot the cytoplasm’s intensity over time:
- labelled_control_cytoplasm.tif
- labelled_sample_cytoplasm.tif

These two files below are used to plot the nucleus’s intensity over time:
- labelled_sample_nucleus.tif
- labelled_control_nucleus.tif

Output:
- CSV files with mean intensity measurements.
- Plots for normalized mean intensity and raw mean intensity over time.

Time taken: about 9 seconds
"""

import numpy as np
import tifffile as tiff
from skimage.measure import regionprops_table
import matplotlib.pyplot as plt
import pandas as pd
import os
import time


# Parameters
inputdir1 = "../data/splited_image"
inputdir2= "../data/tracked_masks"
inputdir3 = "../data/labeled_masks"
output_dir = "../result/single_cell_measure"
params = {
    "sample_image_stack_path": os.path.join(inputdir1, "sample/Exp1_HAP1_Pos06_Sample_CHIR_c01.tif"),
    "control_image_stack_path": os.path.join(inputdir1, "control/Exp1_HAP1_Pos03_Control_c01.tif"),
    "label_stack_paths": [
        (os.path.join(inputdir2, "sample/CorrectedTracking_sample_cell.tif"),
         os.path.join(inputdir2, "control/CorrectedTracking_control_cell.tif"), "whole_cell"),
        (os.path.join(inputdir3, "sample/labelled_sample_nucleus.tif"),
         os.path.join(inputdir3, "control/labelled_control_nucleus.tif"), "nucleus"),
        (os.path.join(inputdir3, "sample/labelled_sample_cytoplasm.tif"),
         os.path.join(inputdir3, "control/labelled_control_cytoplasm.tif"), "cytoplasm")
    ],
"output_dir": output_dir
}


# Start the timer
start = time.perf_counter()

def measure_mask(image_stack_path, label_stack_path):
    """Measure the mean intensity of individual cells over time."""
    image_stack = tiff.imread(image_stack_path)
    label_stack = tiff.imread(label_stack_path).astype(np.uint32)

    if image_stack.shape[0] != label_stack.shape[0]:
        raise ValueError("Image stack and label stack must have the same number of frames")

    intensity_dict = {}
    num_frames = image_stack.shape[0]

    for frame in range(num_frames):
        image = image_stack[frame]
        label_image = label_stack[frame]

        props = regionprops_table(label_image, intensity_image=image, properties=('label', 'intensity_mean'))
        for label, intensity in zip(props['label'], props['intensity_mean']):
            if label not in intensity_dict:
                intensity_dict[label] = []
            intensity_dict[label].append(intensity)

    return {k: v for k, v in intensity_dict.items() if len(v) == num_frames}

def save_to_csv(intensity_dict, filename):
    """Save intensity measurements to a CSV file."""
    data = [[label, i, intensity] for label, intensities in intensity_dict.items() for i, intensity in enumerate(intensities)]
    pd.DataFrame(data, columns=['Label', 'Time', 'Mean Intensity']).to_csv(filename, index=False)

def plot_combined_intensities(sample_dict, control_dict, title, output_path, ylabel, normalized=False):
    """Plot combined intensities for sample and control groups."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for label, intensities in sample_dict.items():
        ax.plot(range(len(intensities)), [i / intensities[0] for i in intensities] if normalized else intensities, 'r')
    for label, intensities in control_dict.items():
        ax.plot(range(len(intensities)), [i / intensities[0] for i in intensities] if normalized else intensities, 'gray')
    ax.plot([], [], 'r', label='Sample')
    ax.plot([], [], 'gray', label='Control')
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.legend(loc='upper left', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(output_path)
    plt.close()

def process_and_plot(sample_image_stack_path, control_image_stack_path, label_stack_paths, output_dir):
    """Process the stacks and plot the mean intensities."""
    for sample_label_path, control_label_path, label_type in label_stack_paths:
        sample_dict = measure_mask(sample_image_stack_path, sample_label_path)
        control_dict = measure_mask(control_image_stack_path, control_label_path)

        save_to_csv(sample_dict, os.path.join(output_dir, f'{label_type}_sample_intensities.csv'))
        save_to_csv(control_dict, os.path.join(output_dir, f'{label_type}_control_intensities.csv'))

        plot_combined_intensities(sample_dict, control_dict, f"Normalized Mean Intensity Over Time ({label_type})",
                                  os.path.join(output_dir, f'{label_type}_normalized_intensities.png'), "Normalized Mean Intensity", normalized=True)
        plot_combined_intensities(sample_dict, control_dict, f"Mean Intensity Over Time ({label_type})",
                                  os.path.join(output_dir, f'{label_type}_raw_intensities.png'), "Mean Intensity", normalized=False)

def main(params):
    """Main function to run the process and plot functions."""
    process_and_plot(params["sample_image_stack_path"], params["control_image_stack_path"], params["label_stack_paths"], params["output_dir"])
    print(f"Time taken: {time.perf_counter() - start:.2f} seconds")

if __name__ == "__main__":
    main(params)
