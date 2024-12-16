'''
MeasureAndPlot.py
Version: 1.0
Author: Qiang Zhao [q.zhao2@student.vu.nl]
Date: 30 Jun, 2024

This script will perform measurement analysis on the fluorescence data of the CTNNB1 protein.

Input:
- ratio_plot: A bool value to decide whether to plot normalized ratio or un-normalized mean intensity.
-
Output:
- output_dir: The directory for saving output CSV and png.

Time taken: around 2 mins
'''

import os
import re
import time
import argparse
import pandas as pd
from scipy.stats import t
import numpy as np
import tifffile as tiff
from matplotlib import pyplot as plt
from skimage import io,measure

def visualize_masksimages(raw_image, nucleus, cytoplasm, cell_boundary):
    plt.figure(figsize=(15, 3))
    titles = ['Raw Image', 'With Nucleus Mask', 'With Cyto/Cell Mask', 'With Cell Boundary Mask']
    masks = [raw_image, nucleus, cytoplasm, cell_boundary]
    for i, (mask, title) in enumerate(zip(masks, titles), 1):
        plt.subplot(1, 4, i)
        plt.imshow(mask)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def extract_number(s):
    match = re.search(r't(\d+)\.', s)
    return int(match.group(1)) if match else 0

def cal_intensity(mask,image): #Deleted
    props = measure.regionprops_table(
        mask,
        intensity_image=image,
        properties=('label', 'area', 'intensity_mean'),
    )
    return props['area'] * props['intensity_mean']

def ci_cell(in_image,in_mask):
    n = len(np.unique(in_mask))
    std = np.std([np.mean(in_image[np.array(np.where(in_mask == k, in_mask, 0), dtype=bool)])
                  for k in np.unique(in_mask) if k != 0])
    t_ = t.ppf(1 - 0.05 / 2, n-1)
    ci = t_ * (std / np.sqrt(n))
    return ci

def measure_mask(image_dir,mask_dir):

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    image_files = sorted(image_files, key=extract_number)
    mask_files = [f for f in os.listdir(mask_dir+'nucleus') if f.endswith('.png')]
    mask_files = sorted(mask_files, key=extract_number)

    intensity_n = []
    intensity_cyto = []
    intensity_cb = []

    ci_n = []
    ci_cyto = []
    ci_cb = []

    for i,m in zip(image_files,mask_files):
        print(m)

        image = tiff.imread(os.path.join(image_dir, i))
        n_mask = io.imread(os.path.join(mask_dir+'nucleus', m))
        cyto_mask = io.imread(os.path.join(mask_dir+'cytoplasm', m))
        cb_mask = io.imread(os.path.join(mask_dir+'cell_boundary', m))

        intensity_n.append(np.mean(image[np.array(n_mask, dtype=bool)]))
        intensity_cyto.append(np.mean(image[np.array(cyto_mask, dtype=bool)]))
        intensity_cb.append(np.mean(image[np.array(cb_mask, dtype=bool)]))

        ci_n.append(ci_cell(image,n_mask))
        ci_cyto.append(ci_cell(image,cyto_mask))
        ci_cb.append(ci_cell(image,cb_mask))

    #Double Checking: Visualization for the image in the final timespot
    visualize_masksimages(image,
                          np.where(np.array(n_mask, dtype=bool), image, 0),
                          np.where(np.array(cyto_mask, dtype=bool), image, 0),
                          np.where(np.array(cb_mask, dtype=bool), image, 0))

    return [intensity_n,intensity_cyto,intensity_cb,ci_n,ci_cyto,ci_cb]

def plot_TimeVsIntensity(control_list,sample_list,save_dir,ratio_plot=False):

    x = list(range(1, 97))
    plt.figure(figsize=(10, 8))

    color_lists = ['blue','orange','red','green','purple','gray']
    plot_list = control_list[:3]+sample_list[:3]
    ci_list = control_list[3:]+sample_list[3:]
    title = ['control_nucleus','control_cytoplasm','control_cell boundary','sample_nucleus','sample_cytoplasm','sample_cell boundary',]

    df = pd.DataFrame()
    df['timepoint'] = x
    for i in range(len(plot_list)):
        if ratio_plot:
            mean_intensities =  plot_list[i] / plot_list[i][0]
            plt.plot(x, mean_intensities, label=title[i], color=color_lists[i])
        else:
            mean_intensities = plot_list[i]
            ci = ci_list[i]
            ci_upper = [m + ci for m, ci in zip(mean_intensities, ci)]
            ci_lower = [m - ci for m, ci in zip(mean_intensities, ci)]

            df[f'{title[i]}'] = mean_intensities
            df[f'ci_L_{title[i]}'] = ci_lower
            df[f'ci_U_{title[i]}'] = ci_upper

            plt.plot(x, mean_intensities, label=title[i], color=color_lists[i])
            plt.fill_between(x, ci_lower, ci_upper, color=color_lists[i], alpha=0.3)

    df = df.round(3)
    plt.title('Intensity VS Time', fontsize=20)
    plt.xlabel('Time', fontsize=20)

    if ratio_plot:
        plt.ylabel('Normalized ratio', fontsize=20)
        csv_filename = save_dir + 'normalized_ratio.csv'
        image_filename = save_dir + 'normalized_ratio.png'
    else:
        plt.ylabel('Mean Intensity', fontsize=20)
        csv_filename = save_dir + 'un-normalized_MeanIntensity.csv'
        image_filename = save_dir + 'un-normalized_MeanIntensity.png'

    df.to_csv(csv_filename, index=False)
    plt.legend(loc='upper left')
    plt.savefig(image_filename)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Measure And Plot')
    parser.add_argument('--ratio_plot','-n', default=False, type=bool,
                        help='If output normalized ratio or un-normalized mean intensity')
    parser.add_argument('--output_dir', '-o', type=str, default='../data/csvAndpng/',
                        help='The directory for saving output csv and png')
    args = parser.parse_args()

    start_time = time.time()

    # Load image and measured masks
    s_image_dir = '../data/splited_image/single_image/sample/cytoplasm/'
    s_mask_dir = '../data/measure/sample/'

    c_image_dir = '../data/splited_image/single_image/control/cytoplasm/'
    c_mask_dir = '../data/measure/control/'

    s_list = measure_mask(s_image_dir,s_mask_dir)
    c_list = measure_mask(c_image_dir,c_mask_dir)

    if args.ratio_plot:
        plot_TimeVsIntensity(c_list,s_list,ratio_plot= args.ratio_plot,save_dir= args.output_dir)
    else:
        plot_TimeVsIntensity(c_list,s_list,save_dir= args.output_dir)

    #Timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time >= 60:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(f"Running Time: {minutes} mins {seconds:.2f} secs")
    else:
        print(f"Running Time: {elapsed_time:.2f} secs")
