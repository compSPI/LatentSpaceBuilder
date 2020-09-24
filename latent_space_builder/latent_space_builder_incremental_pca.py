import sys
import os
from pathlib import Path
import argparse
import json

from tqdm import tqdm
import numpy as np
from sklearn.decomposition import IncrementalPCA
import h5py as h5
import deepdish as dd

"""

Deeban Ramalingam (deebanr@slac.stanford.edu)

This script builds the latent space using Incremental PCA.

python latent_space_builder_incremental_pca.py --config latent-space-builder-incremental-pca.json --dataset 3iyf-10K-mixed-hit-80

"""

def build_latent_space_using_incremental_pca(h5_file, dataset_size, batch_size, latent_dim, metadata_dir, n_train_samples_to_save_metadata):
    """
    
    :param h5_file: Path to the HDF5 file containing the dataset
    :param dataset_size: Size of the dataset
    :param batch_size: Size of each batch
    :param latent_dim: Number of dimensions of the latent space
    :param metadata_dir: Path to a directory to save snapshots of the model during training
    """

    assert dataset_size % batch_size == 0
    n_batches = dataset_size // batch_size

    h5_file_handle = h5.File(h5_file, 'a')
        
    image_type_key = "diffraction_patterns"
    latent_method_key = "incremental_principal_component_analysis"

    images = h5_file_handle[image_type_key]
    
    if latent_method_key in h5_file_handle:
        del h5_file_handle[latent_method_key]
    
    latent_vectors_shape = (dataset_size, latent_dim)
    latent_space = h5_file_handle.create_dataset(latent_method_key, latent_vectors_shape, dtype='f')

    incremental_pca = IncrementalPCA(n_components=latent_dim)
    
    for i_batch in tqdm(range(n_batches)):
        
        image_batch = images[i_batch * batch_size : (i_batch + 1) * batch_size]
        
        _, image_height, image_width = image_batch.shape
        image_batch_vectors = image_batch.reshape(batch_size, image_height * image_width)
        
        incremental_pca.partial_fit(image_batch_vectors)
        
        if (i_batch + 1) * batch_size % n_train_samples_to_save_metadata == 0:
            incremental_pca_metadata = {
                "components": incremental_pca.components_,
                "explained_variance": incremental_pca.explained_variance_,
                "singular_values": incremental_pca.singular_values_,
                "mean": incremental_pca.mean_,
                "var": incremental_pca.var_,
                "noise_variance": incremental_pca.noise_variance_,
                "n_samples_seen": incremental_pca.n_samples_seen_
            }

            metadata_file = "incremental_pca_batch_number={}_batch_size={}.h5".format(i_batch + 1, batch_size)
            metadata_path = os.path.join(metadata_dir, metadata_file)
        
            print("Saving metadata for Incremental PCA (batch_number={}, batch_size={}) at:".format(i_batch + 1, batch_size, metadata_path))
            dd.io.save(metadata_path, incremental_pca_metadata)
    
    for i_batch in tqdm(range(n_batches)):
        
        image_batch = images[i_batch * batch_size : (i_batch + 1) * batch_size]
        
        _, image_height, image_width = image_batch.shape
        image_batch_vectors = image_batch.reshape(batch_size, image_height * image_width)
        
        latent_vectors = incremental_pca.transform(image_batch_vectors)
        
        latent_space[i_batch * batch_size : (i_batch + 1) * batch_size] = latent_vectors
    
    h5_file_handle.close()

def parse_input_arguments(args):
    del args[0]
    parse = argparse.ArgumentParser()
    parse.add_argument('-c', '--config', type=str, help='JSON Config file')
    parse.add_argument('-d', '--dataset', type=str, help='Dataset name(s): 3iyf-10K-mixed-hit-80 3iyf-10K-mixed-hit-90 3iyf-10K-mixed-hit-95 3iyf-10K-mixed-hit-99')

    # convert argparse to dict
    return vars(parse.parse_args(args))

if __name__ == '__main__':
    user_input = parse_input_arguments(sys.argv)
    config_file = user_input["config"]
    dataset_name = user_input["dataset"]
    
    with open(config_file) as config_file:
        config_params = json.load(config_file)

    if dataset_name not in config_params:
        raise Exception("Dataset {} not in Config file.".format(dataset_name))
    
    dataset_params = config_params[dataset_name]
    
    h5_file = dataset_params["h5File"]
    dataset_size = dataset_params["datasetSize"]
    batch_size = dataset_params["batchSize"]
    latent_dim = dataset_params["latentDim"]
    metadata_dir = dataset_params["metadataDir"]
    n_train_samples_to_save_metadata = dataset_params["numTrainSamplesToSaveMetadata"]
    
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    print("Creating latent vectors in: {}".format(h5_file))
    print("Dataset size: {}".format(dataset_size))
    print("Batch size: {}".format(batch_size))
    print("Latent dimension: {}".format(latent_dim))
    print("Metadata directory: {}".format(metadata_dir))
    print("Choosing to save model after training on {} data points".format(n_train_samples_to_save_metadata))
    
    build_latent_space_using_incremental_pca(h5_file, dataset_size, batch_size, latent_dim, metadata_dir, n_train_samples_to_save_metadata)