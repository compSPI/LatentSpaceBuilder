import sys
import os
from pathlib import Path
from random import random
import argparse
import json

import numpy as np
np.random.seed(13)

from PIL import Image
import h5py as h5

"""

Deeban Ramalingam (deebanr@slac.stanford.edu)

This script creates a mixed-hit synthetic dataset from single- and double-hit datasets.

python cspi_create_synthetic_dataset_mixed_hit.py --config cspi-create-synthetic-dataset-mixed-hit.json --dataset 3iyf-10K-mixed-hit-99

If you wish to use the Latent Space Visualizer to visualize the synthetic dataset, make the image output directory accessible to JupyterHub on PSWWW after running the script.

Example on how to make the image output directory accessible to the Latent Space Visualizer:

If img_dir is /reg/data/ana03/scratch/deebanr/3iyf-10K-mixed-hit-99/images then run the following command from the Terminal

ln -s /reg/data/ana03/scratch/deebanr/3iyf-10K-mixed-hit-99 /reg/neh/home/deebanr/3iyf-10K-mixed-hit-99

"""

def create_synthetic_dataset_mixed_hit(mixed_hits_h5_file, single_hits_h5_file, double_hits_h5_file, dataset_size, single_to_double_hits_mixture_ratio, img_dir):
    """
    
    :param mixed_hits_h5_file: Path to the HDF5 file containing the mixed-hit dataset
    :param single_hits_h5_file: Path to the HDF5 file containing the single-hit dataset
    :param double_hits_h5_file: Path to the HDF5 file containing the double-hit dataset
    :param dataset_size: Size of the dataset for mixed, single, and double hits
    :param single_to_double_hits_mixture_ratio: Ratio of single hits to double hits for the mixed-hits dataset
    """

    n_single_hits = int(round(dataset_size * single_to_double_hits_mixture_ratio))
    n_double_hits = int(round(dataset_size * (1.0 - single_to_double_hits_mixture_ratio)))

    assert n_single_hits + n_double_hits == dataset_size

    dataset_idx = np.arange(dataset_size)
    single_hits_idx = np.sort(np.random.choice(dataset_idx, n_single_hits, replace=False))
    double_hits_idx = np.sort(np.random.choice(dataset_idx, n_double_hits, replace=False))

    mixed_hits_h5_file_handle = h5.File(mixed_hits_h5_file, 'w')
    single_hits_h5_file_handle = h5.File(single_hits_h5_file, 'r')
    double_hits_h5_file_handle = h5.File(double_hits_h5_file, 'r')

    orientations_key = "orientations"
    diffraction_patterns_key = "diffraction_patterns"
    atomic_coordinates_key = "atomic_coordinates"
    single_hits_mask_key = "single_hits_mask"

    mixed_hits_h5_file_handle.create_dataset(orientations_key, (dataset_size, 4), dtype='f')
    mixed_hits_h5_file_handle.create_dataset(diffraction_patterns_key, (dataset_size, 1024, 1040), dtype='f')

    atomic_coordinates = single_hits_h5_file_handle[atomic_coordinates_key][:]
    atomic_coordinates_shape = atomic_coordinates.shape
    mixed_hits_h5_file_handle.create_dataset(atomic_coordinates_key, atomic_coordinates_shape, dtype='f', data=atomic_coordinates)
    
    single_hits_mask = np.zeros_like(dataset_idx, dtype=bool)
    single_hits_mask_shape = single_hits_mask.shape
    mixed_hits_h5_file_handle.create_dataset(single_hits_mask_key, single_hits_mask_shape, dtype='u1', data=single_hits_mask)

    i_single_hits = 0
    j_double_hits = 0
    k_mixed_hits = 0
    
    while i_single_hits < n_single_hits and j_double_hits < n_double_hits:
        
        if random() < single_to_double_hits_mixture_ratio:
            
            mixed_hits_h5_file_handle[orientations_key][k_mixed_hits] = single_hits_h5_file_handle[orientations_key][single_hits_idx[i_single_hits]]
            mixed_hits_h5_file_handle[diffraction_patterns_key][k_mixed_hits] = single_hits_h5_file_handle[diffraction_patterns_key][single_hits_idx[i_single_hits]]
            mixed_hits_h5_file_handle[single_hits_mask_key][k_mixed_hits] = True
            
            i_single_hits = i_single_hits + 1
        
        else:
            
            # Use the orientation of the particle centered at the origin
            mixed_hits_h5_file_handle[orientations_key][k_mixed_hits] = double_hits_h5_file_handle[orientations_key][double_hits_idx[j_double_hits]][0]
            mixed_hits_h5_file_handle[diffraction_patterns_key][k_mixed_hits] = double_hits_h5_file_handle[diffraction_patterns_key][double_hits_idx[j_double_hits]]
            mixed_hits_h5_file_handle[single_hits_mask_key][k_mixed_hits] = False
            
            j_double_hits = j_double_hits + 1
        
        save_diffraction_pattern_as_image(k_mixed_hits, img_dir, mixed_hits_h5_file_handle[diffraction_patterns_key][k_mixed_hits])
        
        k_mixed_hits = k_mixed_hits + 1

        if k_mixed_hits % 100 == 0 and k_mixed_hits > 0:
            log_status(k_mixed_hits, i_single_hits, j_double_hits)

    while i_single_hits < n_single_hits:

        mixed_hits_h5_file_handle[orientations_key][k_mixed_hits] = single_hits_h5_file_handle[orientations_key][single_hits_idx[i_single_hits]]
        mixed_hits_h5_file_handle[diffraction_patterns_key][k_mixed_hits] = single_hits_h5_file_handle[diffraction_patterns_key][single_hits_idx[i_single_hits]]
        mixed_hits_h5_file_handle[single_hits_mask_key][k_mixed_hits] = True
        
        save_diffraction_pattern_as_image(k_mixed_hits, img_dir, mixed_hits_h5_file_handle[diffraction_patterns_key][k_mixed_hits])

        i_single_hits = i_single_hits + 1
        k_mixed_hits = k_mixed_hits + 1

        if k_mixed_hits % 100 == 0 and k_mixed_hits > 0:
            log_status(k_mixed_hits, i_single_hits, j_double_hits)

    while j_double_hits < n_double_hits:

        # Use the orientation of the particle centered at the origin
        mixed_hits_h5_file_handle[orientations_key][k_mixed_hits] = double_hits_h5_file_handle[orientations_key][double_hits_idx[j_double_hits]][0]
        mixed_hits_h5_file_handle[diffraction_patterns_key][k_mixed_hits] = double_hits_h5_file_handle[diffraction_patterns_key][double_hits_idx[j_double_hits]]
        mixed_hits_h5_file_handle[single_hits_mask_key][k_mixed_hits] = False
        
        save_diffraction_pattern_as_image(k_mixed_hits, img_dir, mixed_hits_h5_file_handle[diffraction_patterns_key][k_mixed_hits])

        j_double_hits = j_double_hits + 1
        k_mixed_hits = k_mixed_hits + 1

        if k_mixed_hits % 100 == 0 and k_mixed_hits > 0:
            log_status(k_mixed_hits, i_single_hits, j_double_hits)
    
    mixed_hits_h5_file_handle.close()
    single_hits_h5_file_handle.close()
    double_hits_h5_file_handle.close()

def parse_input_arguments(args):
    del args[0]
    parse = argparse.ArgumentParser()
    parse.add_argument('-c', '--config', type=str, help='JSON Config file')
    parse.add_argument('-d', '--dataset', type=str, help='Dataset name: 3iyf-10K-mixed-hit')

    # convert argparse to dict
    return vars(parse.parse_args(args))

def log_status(k_mixed_hits, i_single_hits, j_double_hits):
    print("Processed {} mixed hits, {} single hits, and {} double hits".format(k_mixed_hits, i_single_hits, j_double_hits))

    if j_double_hits > 0:
        print("Current mixture of single hits to double hits: {:.2f}".format((i_single_hits * 1.0) / (i_single_hits + j_double_hits)))

def save_diffraction_pattern_as_image(data_index, img_dir, diffraction_pattern):
    """
    Saves diffraction_pattern as a PNG image in img_dir.
    """
    img_file = 'diffraction-pattern-{}.png'.format(data_index)
    img_path = os.path.join(img_dir, img_file)
    
    im = gnp2im(diffraction_pattern)
    im.save(img_path, format='png')

def gnp2im(image_np):
    """
    Converts an image stored as a 2-D grayscale Numpy array into a PIL image.
    """
    rescaled = (255.0 / image_np.max() * (image_np - image_np.min())).astype(np.uint8)
    im = Image.fromarray(rescaled, mode='L')
    return im
   
if __name__ == '__main__':
    user_input = parse_input_arguments(sys.argv)
    config_file = user_input["config"]
    dataset_name = user_input["dataset"]
    
    with open(config_file) as config_file:
        config_params = json.load(config_file)

    if dataset_name not in config_params:
        raise Exception("Dataset {} not in Config file.".format(dataset_name))
    
    dataset_params = config_params[dataset_name]
    
    mixed_hits_h5_file = dataset_params["mixedHitsH5File"]
    single_hits_h5_file = dataset_params["singleHitsH5File"]
    double_hits_h5_file = dataset_params["doubleHitsH5File"]
    dataset_size = dataset_params["datasetSize"]
    single_to_double_hits_mixture_ratio = dataset_params["singleToDoubleHitsMixtureRatio"]
    img_dir = dataset_params["imgDir"]

    mixed_hits_h5_file_parent_directory = str(Path(mixed_hits_h5_file).parent)
    if not os.path.exists(mixed_hits_h5_file_parent_directory):
        os.makedirs(mixed_hits_h5_file_parent_directory)

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    print("Creating mixed-hit synthetic dataset: {}".format(mixed_hits_h5_file))
    print("Single- to double-hit mixture ratio: {:.2f}%".format(single_to_double_hits_mixture_ratio * 100.0))
    create_synthetic_dataset_mixed_hit(mixed_hits_h5_file, single_hits_h5_file, double_hits_h5_file, dataset_size, single_to_double_hits_mixture_ratio, img_dir)
