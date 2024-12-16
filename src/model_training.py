import argparse
import shutil
import numpy as np
from sklearn.model_selection import KFold
from cellpose import models, io, train, metrics, utils, dynamics
import os
import time
import torch

"""
model_training_ori.py
Version: 3.0
Author: [Qianqian WANG]

Description:
    This script performs cross-validation to optimize Cellpose model parameters using personal datasets.
    It then trains the model using the best parameters on the entire dataset, saving the trained model.

Arguments:
    -t, --image_type  : Type of image to process, either 'cytoplasm' or 'nuclei'. Required.
    Optional:
    -i, --input       : Path to the input image directory or file.
    -o, --output      : Path to the output directory where model will be saved.
    -m, --model       : Path to a specific model file to use for segmentation.
    -n, --ncv         : The number of fold used in cross validate. The default number is 3. It must be int and smaller than the dataset.

Input:
    The image and manually correct 
    
Output:
    The trained model using the best parameters from cross-validation.

Example usage:
    python model_training.py -t nuclei -n 3
"""


def parse_arguments():
    """
    Parse command line arguments for the Cellpose segmentation cross validation and final model training script.

    Arguments:
    -t, --image_type  : Type of image to process, either 'cytoplasm' or 'nuclei'. Required.
    Optional:
    -i, --input       : Path to the input image directory or file.
    -o, --output      : Path to the output directory where model will be saved.
    -m, --model       : Path to a specific model file to use for segmentation.
    -n, --ncv         : The number of fold used in cross validate. The default number is 3. It must be int and smaller than the dataset.

    Description:
    This script performs cross-validation to optimize Cellpose model parameters using personal datasets.
    It then trains the model using the best parameters on the entire dataset, saving the trained model.


    Example usage:
    python model_training_ori.py -t nuclei -n 3
    """
    parser = argparse.ArgumentParser(description="This script performs cross-validation to optimize Cellpose model "
                                                 "parameters using personal datasets. It then trains the model using "
                                                 "the best parameters on the entire dataset, saving the trained model.")
    # add parameters
    parser.add_argument('-t', '--image_type', choices=['cytoplasm', 'nuclei'], required=True,
                        help="Type of image to process (must be 'cytoplasm' or 'nuclei')")
    parser.add_argument('-i', '--input', dest='input_file_dir', help='Path to the input file')
    parser.add_argument('-o', '--output', dest='output_file_dir', help='Path to the output file')
    parser.add_argument('-m', '--model', type=str, help='Path to the model file')
    parser.add_argument('-gpu', dest='gpu', default=False, type=bool,
                        help='Flag indicating whether to use GPU for prediction')
    parser.add_argument('-n', '--ncv', type=int, default=3, help='The number of fold in the cross validation (must be '
                                                                 'int and smaller than the training dataset)')
    args = parser.parse_args()

    # check whether the computer have gpu.
    if args.gpu:
        if not torch.cuda.is_available():
            device = torch.device('cpu')
            args.gpu = False

    return args


def record_time(start_time, end_time, program_name):
    elapsed_time = end_time - start_time
    if elapsed_time < 60:
        unit = "seconds"
    elif elapsed_time < 3600:
        elapsed_time /= 60  # transfer to minutes
        unit = "minutes"
    else:
        elapsed_time /= 3600  # transfer to hours
        unit = "hours"
    output = f"{program_name} ran for {elapsed_time:.2f} {unit}\n"
    return output


def model_training_kcv(args):
    record_time_str = ''
    start_time = time.time()
    # Load training data dir
    if args.input:
        file_dir = args.input
    else:
        file_dir = '../data/selected_image/' + args.image_type
    # Load training data
    X, y, X_names = io.load_images_labels(file_dir, mask_filter="_masks")  # X is image, y is mask
    # Per-computing flows for labels
    dynamics.labels_to_flows(labels=y, files=X_names)
    X, y, X_names = io.load_images_labels(file_dir, mask_filter="_masks")  # X is image, y is mask
    # Load the original mask files to calculate the Jaccard index below
    y_names = [names.replace('.tif', '_masks.tif') for names in X_names]
    y_masks_files = [io.imread(y_name) for y_name in y_names]
    end_time = time.time()
    # store the running time
    record_time_str = record_time_str + record_time(start_time, end_time, 'Load training data')

    start_time = time.time()
    # The grid-search parameters list
    n_epochs_list = [75, 100, 250]
    learning_rates = [0.2, 0.1, 0.05, 0.01]
    weight_decay_list = [1e-5, 1e-4, 1e-3]
    momentum_list = [0.8, 0.9, 1]

    kf = KFold(n_splits=args.ncv)

    best_params = None
    best_score = 0
    io.logger_setup()

    for n_epochs in n_epochs_list:
        for learning_rate in learning_rates:
            for weight_decay in weight_decay_list:
                for momentum in momentum_list:
                    scores = []
                    for train_index, test_index in kf.split(X):
                        print(f'train_index:{train_index}, test_index:{test_index}')
                        X_train = [X[i] for i in train_index]
                        X_test = [X[i] for i in test_index]
                        y_train = [y[i] for i in train_index]
                        y_test = [y[i] for i in test_index]
                        y_test_mask_files = [y_masks_files[i] for i in test_index]
                        if args.image_type == 'cytoplasm':
                            model = models.CellposeModel(gpu=args.gpu, model_type='cyto2')
                        elif args.image_type == 'nuclei':
                            model = models.CellposeModel(gpu=args.gpu, model_type='nuclei')
                        else:
                            print('model is unclear, please check image_type')
                        model_path = train.train_seg(
                            net=model.net,
                            train_data=X_train,
                            train_labels=y_train,
                            test_data=X_test,
                            test_labels=y_test,
                            n_epochs=n_epochs,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            momentum=momentum,
                            SGD=True,
                            normalize=True,
                            channels=[0, 0]
                        )

                        # calculate the jaccard_index for test dataset
                        preds, _, _ = models.CellposeModel(gpu=args.gpu, model_type=model_path).eval(X_test,
                                                                                                     channels=[0, 0])
                        jaccard_index = metrics.aggregated_jaccard_index(masks_true=y_test_mask_files, masks_pred=preds)
                        print(f'jaccard_index = {jaccard_index}')
                        scores.append(np.mean(jaccard_index))

                    mean_score = np.mean(scores)
                    print(
                        f"Params: n_epochs={n_epochs}, learning_rate={learning_rate},weight_decay={weight_decay}, momentum={momentum} Mean Jaccard Index: {mean_score}")
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {'n_epochs': n_epochs, 'learning_rate': learning_rate,
                                       'weight_decay': weight_decay, 'momentum': momentum}

    print(f"Finish the cross validation: Best Parameters: {best_params}, Best Score: {best_score}")

    # save best parameter
    params_path = '../docs/best_params_' + args.image_type + '.txt'
    with open(params_path, 'w') as text_file:
        for key, value in best_params.items():
            text_file.write(f"{key}: {value}\n")

    # delete the middel models
    shutil.rmtree('/models')

    end_time = time.time()
    record_time_str = record_time_str + record_time(start_time, end_time, 'CV')

    start_time = time.time()

    # train final model
    if args.output_file_dir:
        model_name = os.path.join(args.output_file_dir, 'trained_' + args.image_type)
    else:
        model_name = "../../results/models/trained_" + args.image_type
    final_model_path = train.train_seg(model.net, train_data=X, train_labels=y, channels=[0, 0], normalize=True,
                                       weight_decay=best_params['weight_decay'], SGD=True,
                                       learning_rate=best_params['learning_rate'],
                                       momentum=best_params['momentum'],
                                       n_epochs=best_params['n_epochs'], model_name=model_name)
    end_time = time.time()
    record_time_str = record_time_str + record_time(start_time, end_time, 'Train final model')
    print('Finish final model training')
    return record_time_str


if __name__ == "__main__":
    parameters = parse_arguments()
    time_record = model_training_kcv(parameters)
    print(time_record)
