# MPI parameters
from mpi4py import MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
N_RANKS = COMM.size

if RANK == 0:
    assert N_RANKS >= 2, "This script is planned for at least 2 ranks."

MASTER_RANK = 0

import time
import os

# Unlock parallel but non-MPI HDF5
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

from os.path import dirname, abspath

import sys
import argparse
import json

import numpy as np
np.random.seed(13)

from tqdm import tqdm

import h5py as h5


"""
Deeban Ramalingam (deebanr@slac.stanford.edu)

The goal of this script is to efficiently downsample diffraction patterns using the MPI Communication Model.

Run instructions:

mpiexec -n 16 python incremental_pca_downsampling_featurization_mpi.py --config incremental-pca-downsampling-featurization-mpi.json --dataset 3iyf-10K-mixed-hit-99-single-hits-labeled

Algorithm for downsampling diffraction patterns using the MPI Communication Model:

1. Master creates empty set of downsampled diffraction patterns
2. Slave i asks Master for a data index
3. Master provides Slave i with data index k
4. Slave i uses data index k to get diffraction pattern k
5. Slave i downsamples diffraction pattern k
6. Slave i adds downsampled diffraction pattern k to the set of downsampled diffraction patterns created by Master
7. Repeat steps 2-6 for all k

Note that Master and Slaves work in parallel.
"""

def main():
    user_input = parse_input_arguments(sys.argv)
    config_file = user_input["config"]
    dataset_name = user_input["dataset"]
    
    with open(config_file) as config_file:
        config_params = json.load(config_file)

    if dataset_name not in config_params:
        raise Exception("Dataset {} not in Config file.".format(dataset_name))
    
    dataset_params = config_params[dataset_name]
    
    diffraction_patterns_h5_file = dataset_params["diffractionPatternsH5File"]
    downsampled_diffraction_patterns_h5_file = dataset_params["downsampledDiffractionPatternsH5File"]
    downsampled_diffraction_pattern_target_height = dataset_params["downsampledDiffractionPatternTargetHeight"]
    downsampled_diffraction_pattern_target_width = dataset_params["downsampledDiffractionPatternTargetWidth"]
    num_diffraction_patterns_to_downsample = dataset_params["numDiffractionPatternsToDownsample"]
    
    diffraction_patterns_h5_key = "diffraction_patterns"
    downsampled_diffraction_patterns_h5_key = "downsampled_diffraction_patterns"

    if RANK == MASTER_RANK:        
        
        downsampled_diffraction_patterns_h5_file_parent_directory = dirname(abspath(downsampled_diffraction_patterns_h5_file))
        print("\n(Master) Create parent directory for downsampled data: {}.".format(downsampled_diffraction_patterns_h5_file_parent_directory))
        
        if not os.path.exists(downsampled_diffraction_patterns_h5_file_parent_directory):
            print("Create downsampled data h5 file parent directory: {}.".format(downsampled_diffraction_patterns_h5_file_parent_directory))
            os.makedirs(downsampled_diffraction_patterns_h5_file_parent_directory)
        
        print("\n(Master) Create H5 file to save downsampled images: {}.".format(downsampled_diffraction_patterns_h5_file))
        
        downsampled_diffraction_patterns_h5_file_handle = h5.File(downsampled_diffraction_patterns_h5_file, 'w')
        downsampled_diffraction_patterns_h5_file_handle.create_dataset(downsampled_diffraction_patterns_h5_key, (num_diffraction_patterns_to_downsample, downsampled_diffraction_pattern_target_height, downsampled_diffraction_pattern_target_width), dtype='f')
        
        downsampled_diffraction_patterns_h5_file_handle.close()
    
    sys.stdout.flush()
    COMM.barrier()
    
    if RANK == MASTER_RANK:
        
        print("\n(Master) Start receiving requests for data from Slaves.")
        
        for data_k in tqdm(range(num_diffraction_patterns_to_downsample)):
            
            slave_i = COMM.recv(source=MPI.ANY_SOURCE)
            COMM.send(data_k, dest=slave_i)
        
        n_slaves = N_RANKS - 1
        for _ in range(n_slaves):
            
            # Send one "None" to each rank as final flag to stop asking for more data
            slave_i = COMM.recv(source=MPI.ANY_SOURCE)
            COMM.send(None, dest=slave_i)
    
    else:
        
        diffraction_patterns_h5_file_handle = h5.File(diffraction_patterns_h5_file, 'r')
        downsampled_diffraction_patterns_h5_file_handle = h5.File(downsampled_diffraction_patterns_h5_file, 'r+')
                
        while True:
            
            COMM.send(RANK, dest=MASTER_RANK)
            data_k = COMM.recv(source=MASTER_RANK)
            
            if data_k is None:
                print("\n(Slave {}) Receive final flag from Master to stop asking for more data.".format(RANK))
                break
            
            diffraction_pattern_k = diffraction_patterns_h5_file_handle[diffraction_patterns_h5_key][data_k]

            downsampled_diffraction_pattern_k = downsample(diffraction_pattern_k, downsampled_diffraction_pattern_target_height, downsampled_diffraction_pattern_target_width)
            
            downsampled_diffraction_patterns_h5_file_handle[downsampled_diffraction_patterns_h5_key][data_k] = downsampled_diffraction_pattern_k

        diffraction_patterns_h5_file_handle.close()
        downsampled_diffraction_patterns_h5_file_handle.close()

    sys.stdout.flush()
    COMM.barrier()

def parse_input_arguments(args):
    del args[0]
    
    parse = argparse.ArgumentParser()
    parse.add_argument('-c', '--config', type=str, help='JSON Config file')
    parse.add_argument('-d', '--dataset', type=str, help='Dataset name: 3iyf-10K-mixed-hit-99-single-hits-labeled')

    return vars(parse.parse_args(args))

def downsample(img, downsampled_image_target_height, downsampled_image_target_width):
    fft_img = np.fft.fft2(img)
    fft_img_fft_shifted = np.fft.fftshift(fft_img)

    fft_img_fft_shifted_high_frequencies_truncated = truncate_high_frequencies(fft_img_fft_shifted, downsampled_image_target_height, downsampled_image_target_width)

    fft_img_high_frequencies_truncated = np.fft.fftshift(fft_img_fft_shifted_high_frequencies_truncated)
    img_high_frequencies_truncated = np.real(np.fft.ifft2(fft_img_high_frequencies_truncated))
    
    return img_high_frequencies_truncated

def truncate_high_frequencies(img_fft, downsampled_image_target_height, downsampled_image_target_width):   
    img_fft_height = img_fft.shape[0]
    img_fft_width = img_fft.shape[1]
    
    assert downsampled_image_target_height < img_fft_height
    assert downsampled_image_target_width < img_fft_width

    return crop_center(img_fft, downsampled_image_target_width, downsampled_image_target_height)

# https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    return img[starty : starty + cropy, startx : startx + cropx]

if __name__ == '__main__':
    main()
