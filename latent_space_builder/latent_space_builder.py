from sklearn import preprocessing

from sklearn.decomposition import PCA
from pydiffmap.diffusion_map import DiffusionMap

import h5py as h5


def principal_component_analysis(data, latent_dim):
    pca = PCA(n_components=latent_dim)
    latent = pca.fit_transform(data)
    return latent

def diffusion_map(data, latent_dim):
    dmap = DiffusionMap.from_sklearn(n_evecs=latent_dim)
    latent = dmap.fit_transform(data)
    return latent

def build_latent_space(dataset_file, image_type, target_type, latent_method, latent_dim):    
    with h5.File(dataset_file, 'r') as dataset_file_handle:
        print(dataset_file_handle.keys())
        images = dataset_file_handle[image_type][:]
        targets = dataset_file_handle[target_type][:]
        
    n, h, w = images.shape    
    
    scaled_image_vectors = preprocessing.minmax_scale(images.reshape((n, h * w)), feature_range=(0, 1))
    
    if latent_method == "principal_component_analysis":
        latent = principal_component_analysis(scaled_image_vectors, latent_dim)
    elif latent_method == "diffusion_map":
        latent = diffusion_map(scaled_image_vectors, latent_dim)
    else:
        raise Exception("Unrecognized latent method. Please choose from: principal_component_analysis, diffusion_map")
    
    latent_shape = latent.shape
    with h5.File(dataset_file, 'a') as dataset_file_handle:
        if latent_method in dataset_file_handle:
            dset_latent = dataset_file_handle[latent_method]
        else:
            dset_latent = dataset_file_handle.create_dataset(latent_method, latent_shape, dtype='f')
        
        dset_latent[...] = latent
