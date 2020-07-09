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

# def diffusion_map(dataset_file, image_type, latent_dim):
#     diffraction_patterns_dataset_name = "diffraction_patterns"
#     orientations_dataset_name = "orientations"
    
#     with h5.File(dataset_file, "r") as cspi_synthetic_dataset_file_handle:
#         print(cspi_synthetic_dataset_file_handle.keys())
#         diffraction_patterns = cspi_synthetic_dataset_file_handle[image_type][:]
#         orientations = cspi_synthetic_dataset_file_handle[orientations_dataset_name][:]
        
#     n, h, w = diffraction_patterns.shape    
    
#     scaled_image_vectors = preprocessing.minmax_scale(diffraction_patterns.reshape((n, h * w)), feature_range=(0, 1))
    
#     dmap = DiffusionMap.from_sklearn(n_evecs=latent_dim)
#     latent = dmap.fit_transform(scaled_image_vectors)
    
#     dm_dataset_name = "diffusion_map"
#     dm_dataset_shape = latent.shape
#     with h5.File(cspi_synthetic_dataset_file, "a") as cspi_synthetic_dataset_file_handle:
#         if dm_dataset_name in cspi_synthetic_dataset_file_handle:
#             dset_latent = cspi_synthetic_dataset_file_handle[dm_dataset_name]
#         else:
#             dset_latent = cspi_synthetic_dataset_file_handle.create_dataset(dm_dataset_name, dm_dataset_shape, dtype='f')
        
#         dset_latent[...] = latent
                

# def principal_component_analysis(cspi_synthetic_dataset_file, latent_dim):
#     diffraction_patterns_dataset_name = "diffraction_patterns"
#     orientations_dataset_name = "orientations"
    
#     with h5.File(cspi_synthetic_dataset_file, "r") as cspi_synthetic_dataset_file_handle:
#         print(cspi_synthetic_dataset_file_handle.keys())
#         diffraction_patterns = cspi_synthetic_dataset_file_handle[diffraction_patterns_dataset_name][:]
#         orientations = cspi_synthetic_dataset_file_handle[orientations_dataset_name][:]
        
#     n, h, w = diffraction_patterns.shape    
    
#     scaled_image_vectors = preprocessing.minmax_scale(diffraction_patterns.reshape((n, h * w)), feature_range=(0, 1))
    
#     pca = PCA(n_components=latent_dim)
#     latent = pca.fit_transform(scaled_image_vectors)
    
#     pca_dataset_name = "principal_component_analysis"
#     pca_dataset_shape = latent.shape
#     with h5.File(cspi_synthetic_dataset_file, "a") as cspi_synthetic_dataset_file_handle:
#         if pca_dataset_name in cspi_synthetic_dataset_file_handle:
#             dset_latent = cspi_synthetic_dataset_file_handle[pca_dataset_name]
#         else:
#             dset_latent = cspi_synthetic_dataset_file_handle.create_dataset(pca_dataset_name, pca_dataset_shape, dtype='f')
        
#         dset_latent[...] = latent

# def diffusion_map(cspi_synthetic_dataset_file, latent_dim):
#     diffraction_patterns_dataset_name = "diffraction_patterns"
#     orientations_dataset_name = "orientations"
    
#     with h5.File(cspi_synthetic_dataset_file, "r") as cspi_synthetic_dataset_file_handle:
#         print(cspi_synthetic_dataset_file_handle.keys())
#         diffraction_patterns = cspi_synthetic_dataset_file_handle[diffraction_patterns_dataset_name][:]
#         orientations = cspi_synthetic_dataset_file_handle[orientations_dataset_name][:]
        
#     n, h, w = diffraction_patterns.shape    
    
#     scaled_image_vectors = preprocessing.minmax_scale(diffraction_patterns.reshape((n, h * w)), feature_range=(0, 1))
    
#     dmap = DiffusionMap.from_sklearn(n_evecs=latent_dim)
#     latent = dmap.fit_transform(scaled_image_vectors)
    
#     dm_dataset_name = "diffusion_map"
#     dm_dataset_shape = latent.shape
#     with h5.File(cspi_synthetic_dataset_file, "a") as cspi_synthetic_dataset_file_handle:
#         if dm_dataset_name in cspi_synthetic_dataset_file_handle:
#             dset_latent = cspi_synthetic_dataset_file_handle[dm_dataset_name]
#         else:
#             dset_latent = cspi_synthetic_dataset_file_handle.create_dataset(dm_dataset_name, dm_dataset_shape, dtype='f')
        
#         dset_latent[...] = latent
                

# def diffusion_map(cspi_synthetic_dataset_file, latent_dim, k=200, epsilon='bgh', alpha=1.0, neighbor_params={'n_jobs': -1, 'algorithm': 'ball_tree'}):
#     diffraction_patterns_dataset_name = "diffraction_patterns"
#     orientations_dataset_name = "orientations"
    
#     with h5.File(cspi_synthetic_dataset_file, "r") as cspi_synthetic_dataset_file_handle:
#         print(cspi_synthetic_dataset_file_handle.keys())
#         diffraction_patterns = cspi_synthetic_dataset_file_handle[diffraction_patterns_dataset_name][:]
#         orientations = cspi_synthetic_dataset_file_handle[orientations_dataset_name][:]
        
#     n, h, w = diffraction_patterns.shape    
    
#     scaled_image_vectors = preprocessing.minmax_scale(diffraction_patterns.reshape((n, h * w)), feature_range=(0, 1))
    
#     dmap = DiffusionMap.from_sklearn(n_evecs=latent_dim, k=k, epsilon=epsilon, alpha=alpha, neighbor_params=neighbor_params)
#     latent = dmap.fit_transform(scaled_image_vectors)
    
#     dm_dataset_name = "diffusion_map"
#     dm_dataset_shape = latent.shape
#     with h5.File(cspi_synthetic_dataset_file, "a") as cspi_synthetic_dataset_file_handle:
#         if dm_dataset_name in cspi_synthetic_dataset_file_handle:
#             dset_latent = cspi_synthetic_dataset_file_handle[dm_dataset_name]
#         else:
#             dset_latent = cspi_synthetic_dataset_file_handle.create_dataset(dm_dataset_name, dm_dataset_shape, dtype='f')
        
#         dset_latent[...] = latent
        