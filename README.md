# IPQDA-24-A3
## :microscope: Tracking CTNNB1 Fluorescence Signals Using Deep Learning Models

## 1. Introduction
The objective of this project is to develop a Python pipeline for single-cell analysis in microscopy images, focusing on the localization and quantification of β-catenin and nuclear markers. This pipeline uses deep learning tools like Cellpose for image segmentation, skimage for quantitative analysis and TrackMate for single-cell tracking.

<div align="center">
  <img src="/imgs/Group_Plan.png" width="100%">
</div>

## 2. Data Preparation
### 2.1 Creation of Folder Structure
The folder structure is shown in Figure 1.
Here the data folder stored some of the input data, like the raw images, splited images, and manually corrected images with their regions of interest (ROIs) (*_rois.zip)  and masks (*_mask.tif).

The raw images have a time series and two channels. To make them more easily to be used by models, we separate the image from stack to single images. The image name is in the form of ‘Exp1_HAP1_Pos03_Control_c01_t1.tif’. The ‘Exp1_HAP1_Pos03’ means which dataset we used. ‘Control’ means this image belongs to the control group, it will be ‘Sample’ if this image belongs to the sample group. The ‘c01’ means this image is from channel 1, which is the cytoplasm, and for nuclei, it will be ‘c02’. The ‘t1’ means this image is in the time point 1. There are in total 96 time points in our dataset, so the time point is from ‘t1’ to ‘t96’.

The src folder stored all the scripts we used in the project.

The results folder stored all the models, the predicted results from models and the measure results. The predicted results include the mask files (*_cp_mask.png) and the ROIs files (*_rois.zip).

The docs folder stored some log files, like the output while training the model, the final best parameter from cross-validation and the program running time record.

The environment file is stored in the root folder with a suffix as .yml. 

* Figure.1 The folder structure:
<div align="center">
  <img src="/imgs/structure.png" width="20%">
</div>

To record time we used different ways for different scripts. For the model_training scripts, since it will print the current time during training, we just stored the output and did not use a time recorder. For other scripts, we used the Python time package to record. We try to optimise the running time in several ways and we will introduce them below in the corresponding part. 

### 2.2 Split the raw data and select data for model training
This Python script splits raw hyper-stacks into stacks or single images as needed and selects 8 images from both control and sample datasets, totalling 16 for cells and 16 for nuclei for training. The 8 images from each dataset are selected by keeping the first and last time points’ image and maintaining a consistent interval of 13 between selected images.

my_project/
- data/
  - rawdata/
   - Exp1_HAP1_Pos03_Control.tif
   - Exp1_HAP1_Pos06_Sample_CHIR.tif
- src/
  - split_and_select_data.py
To automatically split the raw hyper-stacks and select the training data, you can use the following command:
```bash
python split_and_select_data.py 
```

## 3. Image Segmentation Using pretrain models
Using the Cellpose API, we performed initial segmentation using pre-trained models (‘cyto2’ model for cytoplasm, ‘nuclei’ model for nuclei). Segmentation results (the mask files (*_cp_mask.png) and the ROIs files (*_rois.zip)) were saved in the results/pretrain_models_results/ folder. 

We first use the Cellpose GUI to evaluate the diameter of our image and choose the diam_mean = 100 for the cytoplasm image and diam_mean = 90 for the nuclei image.

To do the segmentation by pretrain model as we did, you can use the command below:
```bash
python predict.py -t nuclei -mt pretrain
```
The arguments include:

  -t, --image_type  : Type of image to process, either 'cytoplasm' or 'nuclei'. Required.
  
  -mt, --model_type : Type of model to use for segmentation, either 'pretrain' or 'trained'. Required.
  
  Optional:
  
  -i, --input       : Path to the input image directory or file.
  
  -o, --output      : Path to the output directory where segmentation results will be saved.
  
  -m, --model       : Path to a specific model file to use for segmentation.

Figure 2 shows part of the results, where the first column is the original image, the second column is the mask predicted by the pretrain models, and the third column is the predicted outline on the original image shown in imageJ. Although most of the cells and nucleus were predicted successfully, for nuclei there are still some dark nucleus that cannot be recognized, and for cytoplasm, some places that do not contain the cell (background) were misrecognized as cell, and for cell that have a unclear edge, the model also cannot predict perfectly. 

We randomly choose one image for nuclei (Exp1_HAP1_Pos03_Control_c02_t10.tif) and cytoplasm (Exp1_HAP1_Pos06_Sample_CHIR_c01_t36.tif) to correct the label manually, which will not be used in the model training below, but to evaluate the model performance by Jaccard index ( the intersection area/ the union area). The Jaccard index is 0.916 for cytoplasm and 0.614 for nuclei.

* Figure 2 Part of the results from pre-train models:
<div align="center">
  <img src="/imgs/pretrain.png" width="70%">
</div>

## 4. Training Data Preparation
We selected 16 images for training (8 from each dataset). To ensure that the images were representative and evenly distributed over time, we chose them at fixed intervals: t1, t14, t27, t40, t53, t66, t79, and t96. 
We used the predicted ROIs from the pretrain model as the basic and manually corrected them using ImageJ. We used the macros ‘rois2labels.ijm’ to transfer the ROIs into mask files in ImageJ. The ‘rois2labels.ijm’ is saved in the src folder, which was modified to suit the images without stack and not the original version from the teacher. The corrected ROIs and masks are stored in the data/selected_image folder, separated by cytoplasm and nuclei.


## 5. Model Training
We use the corrected image masks to train our own models. To find the parameters that have the best performance, we use grid-search and 3-cross-validation (3-CV) with the Jaccard index as the metric. The grid-search parameter list is as follows:

n_epochs_list = [75, 100, 250]

learning_rates = [0.2, 0.1, 0.05, 0.01]

weight_decay_list = [1e-5, 1e-4, 1e-3]

momentum_list = [0.8, 0.9, 1]

The best parameters for the trained_cytoplasm model are: n_epochs: 100, learning_rate: 0.2, weight_decay: 1e-05, momentum: 0.8. The best parameters for the trained_nuclei model are: n_epochs: 250, learning_rate: 0.1, weight_decay: 1e-03, momentum: 0.8. 

The cross-validation needs a lot of time to run since for each parameter combination, it needs to run three times and there are in total 108 combinations. To make the process more quickly, we tried to view the original code in the Cellpose train.train_seg function, and found that before each time training, the algorithm will transfer the masks into flows for future training, which will take a lot of time. For cross-validation, each mask will be used many times, so we first computed all the flows in advance so that the algorithm can directly use the flows without wasting time to repeatedly compute images into flows.
The final models used the best parameters from 3-CV with all 16 manually correct images as training sets, which were stored in the results/models folder. The process output was stored at docs/log_nuclei and docs/log_cytoplasm.

You can run the same process by the command line below:
```base
python model_training.py -t nuclei -n 3
```
The arguments include:

  -t, --image_type  : Type of image to process, either 'cytoplasm' or 'nuclei'. Required.
  
  Optional:
  
  -i, --input       : Path to the input image directory or file.
  
  -o, --output      : Path to the output directory where the model will be saved.
  
  -m, --model       : Path to a specific model file to use for segmentation.
  
  -n, --ncv         : The number of folds used in cross validate. The default number is 3. It must be int and smaller than the dataset.



## 6. Application of Trained Models
We used the trained model to segment all the images in our datasets and stored the masks and ROIs in the results/trained_models_results folder.
Figure 3 shows some of the results, where the first column is the original image, the second column is the mask predicted by the trained models, and the third column is the predicted outline on the original image shown in ImageJ. Compared to the pretrain models, the trained_nuclei model can successfully predict the dark nucleus, and the trained_cytoplasm model can successfully recognize the area that does not contain a cell, and the unclear cell edge. We calculated the Jaccard index for the trained models using the same randomly chosen image that was previously used for the pre-trained models. The results were 0.880 for nuclei and 0.927 for cytoplasm, both higher than the pre-trained models' indices of 0.614 for nuclei and 0.916 for cytoplasm, which indicated that the trained models have better performance than the pretrain models. 

* Figure 3 Part of the results from trained models.
<div align="center">
  <img src="/imgs/trained.png" width="70%">
</div>





## 7. Measurement Mask Creation
### 7.1 Generating Measurement Masks
The following script will perform mask calculation using methods such as skimage to obtain three different types of masks. The resulting files will be stored in the 'control' or 'sample' directories within the 'measure' folder, each automatically generated with three subfolders.

```bash
python CreateMask.py -i <input directory> -o <output directory>
```
### Example of outputs:

- measure/
  - control/
    - cell_boundary/
    - cytoplasm/
    - nucleus/
  - sample/
    - cell_boundary/
    - cytoplasm/
    - nucleus/
      
* Examples of Raw image and Measurement mask:
<div align="center">
  <img src="/imgs/DemoMeasureMask.png" width="100%">
</div>


## 8. Quantitative Analysis
### 8.1 Measuring Fluorescence Intensity
Using the measurement masks, we calculated the average fluorescence intensity for each compartment and plotted the IntensityVsTime figure simultaneously. Results were exported as CSV files.

```bash
python MeasureAndPlot.py -n <If output normalized ratio or unnormalized mean intensity> -o <output_directory>
```
### Example of outputs:

- csvAndpng/
  - normalized_ratio.csv
  - normalized_ratio.png
  - un-normalized_MeanIntensity.csv
  - un-normalized_MeanIntensity.png
    
*Top rows of MeanIntensity csv file*:
| timepoint | control_nucleus | ci_L_control_nucleus | ci_U_control_nucleus | control_cytoplasm | ci_L_control_cytoplasm | ci_U_control_cytoplasm | control_cell boundary | ci_L_control_cell boundary | ci_U_control_cell boundary | sample_nucleus | ci_L_sample_nucleus | ci_U_sample_nucleus | sample_cytoplasm | ci_L_sample_cytoplasm | ci_U_sample_cytoplasm | sample_cell boundary | ci_L_sample_cell boundary | ci_U_sample_cell boundary |
|-----------|-----------------|----------------------|----------------------|-------------------|------------------------|------------------------|-----------------------|----------------------------|----------------------------|----------------|--------------------|--------------------|-----------------|----------------------|----------------------|---------------------|--------------------------|--------------------------|
| 1         | 41.795          | 39.709               | 43.881               | 163.801           | 150.71                 | 176.893                | 334.037               | 304.395                    | 363.68                    | 42.72          | 38.374            | 47.066            | 153.39           | 139.226              | 167.554              | 310.837             | 283.421                  | 338.254                  |
| 2         | 43.347          | 40.02                | 46.674               | 164.144           | 153.067                | 175.221                | 341.542               | 316.978                    | 366.105                    | 43.43          | 39.341            | 47.519            | 153.402          | 139.468              | 167.336              | 311.583             | 283.098                  | 340.068                  |
| 3         | 45.46           | 42.045               | 48.875               | 164.734           | 151.464                | 178.005                | 335.332               | 309.01                     | 361.654                    | 42.426         | 39.235            | 45.617            | 155.189          | 142.104              | 168.274              | 318.407             | 287.337                  | 349.477                  |
| 4         | 43.445          | 37.156               | 49.735               | 166.094           | 153.217                | 178.972                | 356.308               | 330.076                    | 382.539                    | 45.087         | 39.835            | 50.339            | 163.326          | 146.974              | 179.679              | 314.558             | 282.175                  | 346.941                  |
| 5         | 45.178          | 36.192               | 54.163               | 163.392           | 151.453                | 175.33                 | 342.864               | 313                        | 372.728                    | 43.745         | 40.148            | 47.343            | 152.659          | 138.971              | 166.346              | 303.135             | 279.014                  | 327.256                  |


### 8.2 Results
Matplotlib was used to generate plots of fluorescence intensity over time, comparing control and sample conditions.

* Results of Mean Intensity:
<div align="center">
  <img src="/imgs/MeanIntensity.png" width="70%">
</div>

* Results of normalized ratio:
<div align="center">
  <img src="/imgs/NormalizedRatio.png" width="70%">
</div>


This above plot indicates fluorescence intensity versus time for control and sample groups across three cellular compartments: nucleus, cytoplasm, and cell boundary. Key observations include:

(1) The cell membrane exhibits the highest fluorescence intensity in both sample and control groups.

(2) In the treated sample group, the fluorescence intensity in all three regions starts increasing after 1 hour, while the control group remains stable.

(3) For the sample group, the cell membrane fluorescence intensity decreases after approximately 6 hours, unlike the cytoplasm and nucleus, suggesting CTNNB1 molecules are gradually transferring to nuclear regions, which agrees with the previous studies[1]. 

## 9. Single Cell Tracking
### 9.1 Manual Correction of Masks
The cell masks for both the control and sample group, generated by our trained model, were manually checked and corrected to ensure accurate tracking. The corrected masks are stored in data/corrected_masks, separated by control stack and sample stack.


### 9.2 Applying Cell Tracking
We used the TrackMate [2] plugin in Fiji to track the corrected cell masks. TracMate provides various detectors and trackers. After numerous trials with various detectors, trackers, and parameter settings, we determined that the ‘Label Image detector’ combined with the ‘LAP tracker’ performed optimally for our dataset. The other parameters we used are shown in the table below:
|parameters|Max distance (pixels)|
|----------|---------------------|
|Frame to frame linking|70|
|Track segment gap closing|30|
|Max frame gap|5|
|Track segment spliting|30|
|Track segment merging|30|

We then manually corrected the tracks in TrackScheme. After correction, we generated animations illustrating cell movement over time with the 'Capture Overlay' option and created labelled images with the 'Export Label Image' option, ensuring consistent labelling for cells in the same track. 

* Example of single-cell movement:
<table>
  <tr>
    <td>
      <div align="center">
        <img src="/imgs/TrackMate_control_cyto.gif" width="100%">
      </div>
    </td>
    <td>
      <div align="left">
        <img src="/imgs/TrackMate_selected_control_cyto.gif" width="50%">
      </div>
    </td>
  </tr>
</table>

## 10. Single Cell Analysis
### 10.1 Generating and Analyzing Tracked Measurement Masks

We used the assign_tracking_label.py script, stored in 'src' to create labelled nucleus and cytoplasm mask stacks with the same labels as the 'Export Label Image' option. After consulting with the teacher, we learned that separating the cell membrane in the single-cell analysis was unnecessary, so we did not include this part. These labelled masks were stored in the 'labeled_masks' folder and will be used for single-cell analysis.

You can use the following command to generate the labelled nucleus and cytoplasm mask stacks:

```bash
python assign_tracking_label.py 
```

* Example of labelled nucleus and cytoplasm mask:
  
<div align="center">
  <img src="/imgs/Control_Cytoplasm_Frame_0.png" width="45%">
  <img src="/imgs/Control_Nucleus_Frame_0.png" width="45%">
</div>
 
We analyzed and plotted the intensity of single cells over time using the ‘single_cell_measurement.py’ script stored in the 'src' directory. We generated plots for traces across all 96-time points for the whole cell, cytoplasm, and nucleus. By normalizing each time point using the first time point as a reference, we obtained ratios that clearly demonstrated changes in density over time. Although we also plotted the raw mean intensity of single cells over time, the figures with normalized data were more illustrative. Therefore, we chose the normalized results as our final result.
You can use the following command to analyze and plot the single-cell intensity over time. 
```bash
python single_cell_measurement.py
```
The figures below illustrate the normalized and raw single-cell intensity for both control and sample groups. In the control group, the intensity line remains flat, whereas in the sample group, the intensity line trends upward. This observation is consistent with the experimental findings. Additionally, the varied trends observed among cells within the same group highlight the heterogeneity of the cells.

* Results of normalized single-cell measurement: 
<div align="center">
  <img src="/imgs/whole_cell_normalized_intensities.png" width="30%">
  <img src="/imgs/nucleus_normalized_intensities.png" width="30%">
  <img src="/imgs/cytoplasm_normalized_intensities.png" width="30%">
</div>

* Results of raw single-cell measurement: 
<div align="center">
  <img src="/imgs/whole_cell_raw_intensities.png" width="30%">
  <img src="/imgs/nucleus_raw_intensities.png" width="30%">
  <img src="/imgs/cytoplasm_raw_intensities.png" width="30%">
</div>

## Reference

1. Saskia MA de Man, Gooitzen Zwanenburg, Tanne van der Wal, Mark A Hink, Renée van Amerongen (2021) Quantitative live-cell imaging and computational modelling shed new light on endogenous WNT/CTNNB1 signalling dynamics eLife 10:e66440

2. https://imagej.net/media/plugins/trackmate/trackmate-manual.pdf
   
## Contribution

Qianqian Wang: Responsible for the pre-train model, model training, and application of the trained model, including writing the corresponding scripts

Qiang Zhao: Responsible for creating measurement masks, measuring the masks, and writing the corresponding scripts. Additionally, manually corrected the single-cell tracking in the control group.

Xiaomin Zhang: Responsible for data splitting, training data selection, single-cell tracking, single-cell analysis, and writing the corresponding scripts.

The tasks related to the manual correction of the masks were evenly distributed among all members.
## License

[MIT](https://choosealicense.com/licenses/mit/)
