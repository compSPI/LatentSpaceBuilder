import time

import numpy as np

from sklearn import preprocessing

from sklearn.decomposition import PCA, IncrementalPCA
from pydiffmap.diffusion_map import DiffusionMap

import h5py as h5


def build_latent_space(dataset_file, image_type, target_type, latent_method, latent_dim, dataset_size, training_set_size, batch_size):
    # Check if batch_size divides dataset_size
    if dataset_size % batch_size != 0:
        raise Exception("batch_size {} must divide dataset_size {}".format(batch_size, dataset_size))
    
    n_batches = dataset_size // batch_size
    
    # Read the image data from the dataset
    with h5.File(dataset_file, 'a') as dataset_file_handle:
        # Get handle to images
        images_handle = dataset_file_handle[image_type]
        
        # Get shape of images
        h, w = images_handle[0].shape

        # Define a min-max rescaler to scale images between 0 and 1
        # Adapted from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html
        max_value = np.finfo(np.float).min
        min_value = np.finfo(np.float).max
        
        for i in range(n_batches):
            image_batch = images_handle[i * batch_size : (i + 1) * batch_size]
            max_value = max(max_value, np.max(image_batch))
            min_value = min(min_value, np.min(image_batch))
        
        def minmax_rescale(x):
            return (x - min_value) / (max_value - min_value)

        # Define the training set
        training_set_idx = np.sort(np.random.choice(dataset_size, training_set_size, replace=False))
        training_set = images_handle[training_set_idx]

        # Vectorize the training set
        training_set_vectors = training_set.reshape(training_set_size, h * w)
        
        # Rescale the training set
        training_set_rescaled = minmax_rescale(training_set_vectors)
        
        # Fit the latent (dimensionality reduction) method to the training set
        latent_model = None
        if latent_method == "principal_component_analysis":
            # Fit PCA to the rescaled training set
            latent_model = PCA(n_components=latent_dim)
            tic = time.time()
            latent_model.fit(training_set_rescaled)
            toc = time.time()
            print("It takes {:.2f} seconds for PCA to fit to the training set of shape {}.".format(toc-tic, training_set_rescaled.shape))
        
        elif latent_method == "diffusion_map":
            # Fit Diffusion Map to the rescaled training set
            latent_model = DiffusionMap.from_sklearn(n_evecs=latent_dim)
            tic = time.time()
            latent_model.fit(training_set_rescaled)
            toc = time.time()
            print("It takes {:.2f} seconds for Diffusion Map to fit to the training set of shape {}.".format(toc-tic, training_set_rescaled.shape))
        
        elif latent_method == "incremental_principal_component_analysis":
            # Check if batch_size divides training_set_size
            if training_set_size % batch_size != 0:
                raise Exception("batch_size {} must divide training_set_size {}".format(batch_size, training_set_size))

            n_training_batches = training_set_size // batch_size

            # https://stackoverflow.com/questions/31428581/incremental-pca-on-big-data
            # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
            latent_model = IncrementalPCA(n_components=latent_dim)
            total_time = 0
            for i in range(n_training_batches):
                # Get a training batch of rescaled image vectors
                training_batch_rescaled = training_set_rescaled[i * batch_size : (i + 1) * batch_size]
                
                # Partial fit Incremental PCA to the rescaled training batch
                tic = time.time()
                latent_model.partial_fit(training_batch_rescaled)
                toc = time.time()
                print("It takes {:.2f} seconds for Incremental PCA to fit to a training batch of shape {}.".format(toc-tic, training_batch_rescaled.shape))
                total_time += toc - tic
            
            print("It takes {:.2f} seconds total for Incremental PCA to fit to the training set of shape {}.".format(total_time, training_set_rescaled.shape))
        
        else:
            raise Exception("Unsupported latent method. Please provide one of the following: principal_component_analysis, diffusion_map")
        
        # Define shape of latent vectors
        latent_vectors_shape = (dataset_size, latent_dim)
        
        # Delete dataset in HDF5 file if dataset with same key already exists
        if latent_method in dataset_file_handle:
            del dataset_file_handle[latent_method]

        # Create a new dataset for the latent vectors
        latent_vectors_handle = dataset_file_handle.create_dataset(latent_method, latent_vectors_shape, dtype='f')

        # Apply latent method to batches of images in the dataset and add the resulting latent vectors into the dataset
        total_time = 0
        for i in range(n_batches):
            # Get a batch of images
            image_batch = images_handle[i * batch_size : (i + 1) * batch_size]
            
            # Reshape into image vectors
            image_batch_vectors = image_batch.reshape(batch_size, h * w)
            
            # Rescale between 0 and 1
            image_batch_vectors_rescaled = minmax_rescale(image_batch_vectors)
            
            # Transform the rescaled image vectors using the fit latent method into latent vectors
            tic = time.time()
            latent_vectors = latent_model.transform(image_batch_vectors_rescaled)
            toc = time.time()
            print("It takes {:.2f} seconds for transforming a batch of {} image vectors.".format(toc-tic, image_batch_vectors_rescaled.shape))
            total_time += toc - tic
            
            # Add the latent vectors into the dataset
            latent_vectors_handle[i * batch_size : (i + 1) * batch_size] = latent_vectors
        
        print("It takes {:.2f} seconds total to transform the dataset of shape {}.".format(total_time, (dataset_size, h, w)))
