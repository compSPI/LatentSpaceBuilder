import time

import numpy as np

from sklearn import preprocessing

from sklearn.decomposition import PCA
from pydiffmap.diffusion_map import DiffusionMap

import h5py as h5


def principal_component_analysis(data, latent_dim):
    pca = PCA(n_components=latent_dim)
    tic = time.time()
    latent_vectors = pca.fit_transform(data)
    toc = time.time()
    print("It takes {:.2f} seconds for PCA to complete.".format(toc-tic))
    return latent_vectors

def diffusion_map(data, latent_dim):
    dmap = DiffusionMap.from_sklearn(n_evecs=latent_dim)
    tic = time.time()
    latent_vectors = dmap.fit_transform(data)
    toc = time.time()
    print("It takes {:.2f} seconds for diffusion map to complete.".format(toc-tic))
    return latent_vectors

def build_latent_space(dataset_file, image_type, target_type, latent_method, latent_dim, dataset_size, random_sample_size=None): 
    if random_sample_size is not None:
        # Consider a random sample of the image data
        random_idx = np.random.choice(dataset_size, random_sample_size, replace=False)
        random_idx_sorted = np.sort(random_idx)

        # Read the image data from the HDF5 file
        with h5.File(dataset_file, 'r') as dataset_file_handle:
            images = dataset_file_handle[image_type][random_idx_sorted]
    
    else:
        # Read the image data from the HDF5 file
        with h5.File(dataset_file, 'r') as dataset_file_handle:
            images = dataset_file_handle[image_type][:]
    
    # Get the shape of the image data
    n, h, w = images.shape
    
    # Normalize image data to be between 0 and 1
    scaled_image_vectors = preprocessing.minmax_scale(images.reshape((n, h * w)), feature_range=(0, 1))
    
    # Specify the method used to build the latent space
    if latent_method == "principal_component_analysis":
        latent_vectors = principal_component_analysis(scaled_image_vectors, latent_dim)
    elif latent_method == "diffusion_map":
        latent_vectors = diffusion_map(scaled_image_vectors, latent_dim)
    else:
        raise Exception("Unrecognized latent method. Please choose from: principal_component_analysis, diffusion_map")
    
    # Write the latent vectors to the HDF5 file
    latent_vectors_shape = latent_vectors.shape
    with h5.File(dataset_file, 'a') as dataset_file_handle:
        if latent_method in dataset_file_handle:
            del dataset_file_handle[latent_method]
        
        dataset_file_handle.create_dataset(latent_method, latent_vectors_shape, dtype='f', data=latent_vectors)
