"""
Deeban Ramalingam (deebanr@slac.stanford.edu)

The goal of this script is to efficiently build the latent space using Ensemble PCA using the MPI Communication Model.

This script implements Ensemble PCA using the MPI Communication Model.

Run instructions:

mpiexec -n 2 python ensemble_pca_mpi.py --config ensemble-pca-mpi.json --dataset 3iyf-10K
mpiexec -n 3 python ensemble_pca_mpi.py --config ensemble-pca-mpi.json --dataset 3iyf-10K

Description of the algorithm:

The algorithm is split into two phases.

Phase I: Fitting the PCA models

In the first phase, we fit the PCA base models of Ensemble PCA to batches of the training dataset. Using the MPI Communication Model, Master divides up the training dataset into batches and assigns each batch to a Slave. Each Slave will then fit a PCA model to the batch. At the end of this phase, all of the training data will have been fit by a PCA model owned by some Slave.

More formally:
    
1. Slave i asks Master for training data batch j
2. Master provides Slave i with batch j
3. Slave i defines PCA model ij and fits to batch j
4. Slave i adds PCA ij to PCA sub-ensemble i, where PCA sub-ensemble is a subset of the ensemble of PCA base models

Phase II: Building the Latent Space using the PCA models

In the second phase, we use the fit PCA models to transform batches of the entire dataset. The tranformations are then averaged to build the the latent space. Using the MPI Communication Model, Master divides up the dataset into batches and assigns each batch to a Slave. Each Slave then transforms each batch using their collection of PCA models that were fit in the previous phase. The Slaves averages the resulting transformations to build the latent space. At the end of this phase, the latent space should be fully populated with the transformations.

More formally:

1. Slave i asks Master for data batch k
2. Master provides Slave i with batch k
3. Slave i uses its j PCA models ij to transform batch k into transformation ik
4. Slave i adds the transformation ik to the latent space
"""

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

import sys
import argparse
import json

import numpy as np
from sklearn.decomposition import PCA

from tqdm import tqdm

import h5py as h5

# Set random seed
np.random.seed(7)


def main():
    # Parse user input for config file and dataset name
    user_input = parse_input_arguments(sys.argv)
    config_file = user_input['config']
    dataset_name = user_input['dataset'] 
    
    # Get the Config file parameters
    with open(config_file) as config_file:
        config_params = json.load(config_file)

    # Check if dataset in Config file
    if dataset_name not in config_params:
        raise Exception("Dataset {} not in Config file.".format(dataset_name))
    
    # Get the dataset parameters from Config file parameters
    dataset_params = config_params[dataset_name]
    h5_file = dataset_params["h5File"]
    dataset_key = dataset_params["datasetKey"]
    latent_space_key = dataset_params["latentSpaceKey"]
    dataset_size = dataset_params["datasetSize"]
    training_set_size = dataset_params["trainingSetSize"]
    training_batch_size = dataset_params["trainingBatchSize"]
    testing_batch_size = dataset_params["testingBatchSize"]
    latent_dim = dataset_params["latentDim"]
    n_shuffles = dataset_params["nShuffles"]
     
    # Define indices for the entire dataset
    dataset_idx = np.arange(dataset_size)
        
    # Define indices for the training set
    training_set_idx = np.sort(np.random.choice(dataset_idx, training_set_size, replace=False))
    
    if RANK == MASTER_RANK:
        # Define a key for a training set mask in the HDF5 file, specifying which data points are in the training set, True if in training, False o/w
        training_set_mask_key = "training_set_mask"
        
        print("\n(Master) Create/update a training set mask {} in HDF5 file {}.".format(training_set_mask_key, h5_file))
        
        # Get the HDF5 file
        h5_file_handle = h5.File(h5_file, 'a')

        # Define the shape for the training set mask
        training_set_mask_shape = (dataset_size,)
        
        # Define the data for the training set mask
        training_set_mask = np.zeros(training_set_mask_shape, dtype=np.bool)
        training_set_mask[training_set_idx] = True
        
        # Delete any existing training set mask in HDF5 file
        if training_set_mask_key in h5_file_handle:
            del h5_file_handle[training_set_mask_key]
        
        # Create a new dataset for the boolean array of training labels
        h5_file_handle.create_dataset(training_set_mask_key, training_set_mask_shape, dtype='u1', data=training_set_mask)
        
        # Close the HDF5 file
        h5_file_handle.close()
    
    sys.stdout.flush()

    # Make sure HDF5 file is closed created before others open it
    COMM.barrier()
    
    # Resample training set randomly into n_shuffles batches
    training_set_idx_resampled = np.zeros(n_shuffles * training_batch_size)
    for training_batch_j in range(n_shuffles):
        # Define the batch start and end offsets
        batch_start = training_batch_j * training_batch_size
        batch_end = (training_batch_j + 1) * training_batch_size
        
        # Resample a batch from the training set
        batch_resample_idx = np.random.choice(training_set_idx, training_batch_size, replace=False)
        training_set_idx_resampled[batch_start:batch_end] = np.sort(batch_resample_idx)
    
    # Define indices for the testing set
    #testing_set_idx = np.delete(dataset_set_idx, training_set_idx)
    testing_set_idx = dataset_idx
    
    # Check if the training dataset can be split into m batches
    if training_set_size % training_batch_size != 0:
        raise Exception("training_set_size {} needs to be divisible by training_batch_size {}".format(training_set_size, training_batch_size))
    
    # Check if the testing dataset can be split into p batches
    #testing_set_size = dataset_size - training_set_size
    testing_set_size = dataset_size
    if testing_set_size % testing_batch_size != 0:
        raise Exception("testing_set_size {} needs to be divisible by testing_batch_size {}".format(testing_set_size, testing_batch_size))
    
    # Split the training dataset into training batches
    m_training_batches = training_set_size // training_batch_size
    
    # Split the testing dataset into testing batches
    p_testing_batches = testing_set_size // testing_batch_size
    
    # There is one Master and the rest of the ranks are Slaves
    n_slaves = N_RANKS - 1
    
    """
    PREPROCESSING
    """
    
#     # Initialize the min and max to their respective limits before processing the training dataset
#     max_value = np.finfo(np.float).min
#     min_value = np.finfo(np.float).max
#     if RANK == MASTER_RANK:
#         # Get the HDF5 file
#         h5_file_handle = h5.File(h5_file, "r+")
        
#         # Read dataset
#         h5_dataset = h5_file_handle[dataset_key]
        
#         # Update the min and max as we go through the training dataset
#         print("(Master) Preprocessing m_training_batches={} of training_batch_size={}".format(m_training_batches, training_batch_size))
#         for batch_j in tqdm(range(m_training_batches)):
            
#             # Define the batch start and end offsets
#             batch_start = batch_j * training_batch_size
#             batch_end = (batch_j + 1) * training_batch_size
            
#             # Define the training batch
#             training_batch_j = h5_dataset[training_set_idx[batch_start:batch_end]]
            
#             # Update the current min and max
#             max_value = max(max_value, np.max(training_batch_j))
#             min_value = min(min_value, np.min(training_batch_j))
        
#         # Close the HDF5 file
#         h5_file_handle.close()
    
#     sys.stdout.flush()

#     # Wait for Master to find min and max
#     COMM.barrier()
    
    # Broadcasting a numpy array as [max, min] to Slavess
#     buffer = np.array([max_value.item(), min_value.item()])
#     COMM.Bcast(buffer, root=MASTER_RANK)
#     max_value, min_value = buffer
    
    """
    TRAINING
    """
    
    if RANK == MASTER_RANK:
        #print("\n(Master) Training on m_training_batches={} of size training_batch_size={}".format(m_training_batches, training_batch_size))
        
        print("\n(Master) Training on n_shuffles={} of size training_batch_size={}".format(n_shuffles, training_batch_size))

        # Send batch numbers j=1, 2, ..., m_training_batches to Slaves i=1, 2, ..., n_slaves
        #for batch_j in tqdm(range(m_training_batches)):
        for batch_j in tqdm(range(n_shuffles)):
            
            # Receive query for batch j from a Slave i
            slave_i = COMM.recv(source=MPI.ANY_SOURCE)
            
            # Send batch j to Slave i
            COMM.send(batch_j, dest=slave_i)
        
        # Tell Slave ranks to stop asking for more data since there are no more batch numbers to process
        for _ in range(n_slaves):
            # Send one "None" to each rank as final flag
            slave_i = COMM.recv(source=MPI.ANY_SOURCE)
            COMM.send(None, dest=slave_i)

    else: 
        # Get the HDF5 file
        h5_file_handler = h5.File(h5_file, "r+")

        # Get the dataset from the HDF5 file
        h5_dataset = h5_file_handler[dataset_key]
        
        # Each Slave owns a subset of the ensemble of PCA base models, call it PCA sub-ensemble i
        pca_subensemble_i = []
        
        while True:
            # Ask for batch j from Master
            COMM.send(RANK, dest=MASTER_RANK)

            # Receive batch j from Master
            batch_j = COMM.recv(source=MASTER_RANK)

            # If batch j is final flag, stop
            if batch_j is None:
                break

            # Define the batch start and end offsets
            batch_start = batch_j * training_batch_size
            batch_end = (batch_j + 1) * training_batch_size

            # Get training batch j
            training_batch_j_idx = training_set_idx_resampled[batch_start:batch_end]
            training_batch_j = h5_dataset[training_batch_j_idx]
                        
            # Vectorize the training batch
            training_batch_j_vectors = training_batch_j.reshape((training_batch_size, -1))

            # Rescale the training batch
#             def minmax_rescale(x):
#                 return (x - min_value) / (max_value - min_value)
            
#             training_batch_j_rescaled = minmax_rescale(training_batch_j_vectors)
            
            # Define PCA model ij
            pca_model_ij = PCA(n_components=latent_dim)
            
            # Fit PCA ij to batch j
            #pca_model_ij.fit(training_batch_j_rescaled)
            pca_model_ij.fit(training_batch_j_vectors)
            
            # Add PCA ij to subset i
            pca_subensemble_i.append(pca_model_ij)
            
        # Close the HDF5 file
        h5_file_handler.close()

    sys.stdout.flush()

    # Wait for ranks to finish
    COMM.barrier()
    
    """
    TESTING
    """
    
    if RANK == MASTER_RANK:
        print("\n(Master) Create/update empty latent space {} in HDF5 file {}.".format(latent_space_key, h5_file))
        
        # Get the HDF5 file
        h5_file_handle = h5.File(h5_file, 'a')
        
        # Define the latent space shape
        latent_space_shape = (testing_set_size, latent_dim)
        
        # Delete any existing latent space in HDF5 file
        if latent_space_key in h5_file_handle:
            del h5_file_handle[latent_space_key]

        # Create a new dataset for the latent vectors
        h5_file_handle.create_dataset(latent_space_key, latent_space_shape, dtype='f')
        
        # Close the HDF5 file
        h5_file_handle.close()
    
    sys.stdout.flush()

    # Make sure HDF5 file is closed created before others open it
    COMM.barrier()
    
    if RANK == MASTER_RANK:
        print("\n(Master) Testing on p_testing_batches={} of size testing_batch_size={}".format(p_testing_batches, testing_batch_size))
        
        # Send batch numbers k=1, 2, ..., p_testing_batches to Slaves i=1, 2, ..., n_slaves
        for batch_k in tqdm(range(p_testing_batches)):
            
            # Receive query for batch k from a Slave i
            slave_i = COMM.recv(source=MPI.ANY_SOURCE)
            
            # Send batch k to Slave i
            COMM.send(batch_k, dest=slave_i)

        # Tell Slave ranks to stop asking for more data since there are no more batch numbers to process
        for _ in range(n_slaves):
            # Send one "None" to each rank as final flag
            slave_i = COMM.recv(source=MPI.ANY_SOURCE)
            COMM.send(None, dest=slave_i)
    else:
        # Get the HDF5 file
        h5_file_handle = h5.File(h5_file, "r+")

        # Get the dataset from the HDF5 file
        h5_dataset = h5_file_handle[dataset_key]
        
        # Get the latent space from the HDF5 file
        h5_latent_space = h5_file_handle[latent_space_key]
        
        while True:
            # Ask for batch k from Master
            COMM.send(RANK, dest=MASTER_RANK)

            # Receive batch k from Master
            batch_k = COMM.recv(source=MASTER_RANK)

            # If batch k is final flag, stop
            if batch_k is None:
                break

            # Define the batch start and end offsets
            batch_start = batch_k * testing_batch_size
            batch_end = (batch_k + 1) * testing_batch_size   
            
            # Define the testing batch
            testing_batch_k = h5_dataset[testing_set_idx[batch_start:batch_end]]

            # Vectorize the training batch
            testing_batch_k_vectors = testing_batch_k.reshape((testing_batch_size, -1))

            # Rescale the training batch
#             def minmax_rescale(x):
#                 return (x - min_value) / (max_value - min_value)
            
#             testing_batch_k_rescaled = minmax_rescale(testing_batch_k_vectors)
            
#             V = get_projection_matrix_for_top_two_dissimilar_eigenvectors(pca_subensemble_i)
#             M = np.mean(testing_batch_k_vectors.T, axis=1)
#             C = testing_batch_k_vectors - M
        
#             transformation_ik = np.matmul(V, C.T).T
            
#             h5_latent_space[testing_set_idx[batch_start:batch_end]] = transformation_ik
    
            # Define a zero Numpy array to accumulate the transformations produced by Slave's j PCA models ij
            transformation_ik = np.zeros((testing_batch_size, latent_dim))
            
            # Use this Slave's j PCA models ij to transform batch k into transformation ik
            for pca_model_ij in pca_subensemble_i:
                transformation_ik += pca_model_ij.transform(testing_batch_k_vectors)
                #transformation_ijk = pca_model_ij.transform(testing_batch_k_vectors)
                
                # align transformation ik according to principal axis
                #transformation_ijk_aligned = center_and_align_atom_positions_according_to_principal_axes(transformation_ijk)
                
                # Average aligned transformations
                #transformation_ik += transformation_ijk_aligned
            
            # Add the transformation to the latent space
            h5_latent_space[testing_set_idx[batch_start:batch_end]] = transformation_ik
            
            #h5_latent_space[testing_set_idx[batch_start:batch_end]] += center_and_align_atom_positions_according_to_principal_axes(transformation_ik)
            
        # Close the HDF5 file
        h5_file_handle.close()

    sys.stdout.flush()

    # Wait for ranks to finish
    COMM.barrier()

def parse_input_arguments(args):
    del args[0]
    parse = argparse.ArgumentParser()
    parse.add_argument('-c', '--config', type=str, help='JSON Config file')
    parse.add_argument('-d', '--dataset', type=str, help='Dataset name: 3iyf-10K')

    # convert argparse to dict
    return vars(parse.parse_args(args))

# def build_inertia_matrix(r):
#     x = r[:, 0]
#     y = r[:, 1]
#     z = r[:, 2]
#     I = np.zeros((3, 3))
#     I[0, 0] = -np.sum(np.square(y) + np.square(z))
#     I[0, 1] = np.sum(x * y)
#     I[0, 2] = np.sum(x * z)
#     I[1, 0] = np.sum(x * y)
#     I[1, 1] = -np.sum(np.square(x) + np.square(z))
#     I[1, 2] = np.sum(y * z)
#     I[2, 0] = np.sum(x * z)
#     I[2, 1] = np.sum(y * z)
#     I[2, 2] = -np.sum(np.square(x) + np.square(y))
#     return I
    
# def center_and_align_atom_positions_according_to_principal_axes(r):
#     # https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_matrix_in_different_reference_frames
    
#     # Compute its center of mass
#     r_C = np.mean(r, axis=0)

#     # Center the atom positions
#     dr = r - r_C

#     # Compute the body frame inertia matrix
#     IcB = build_inertia_matrix(dr)

#     # Find the principal moments of inertia and the rotation matrix that defines the directions of the principal axes of the particle
#     Ixyz, Q = np.linalg.eig(IcB)

#     # Align the centered atom positions in the body frame to the inertial frame
#     dr_rotated = np.matmul(dr, Q)
    
#     return dr_rotated

# def get_projection_matrix_for_top_two_dissimilar_eigenvectors(pca_subensemble_i):
#     v = np.zeros((len(pca_subensemble_i), latent_dim))
#     for i in range(len(pca_subensemble_i)):
#         for j in range(latent_dim):
#             v[i, j] = pca_subensemble_i[i].components_[j]

#     scalar_product_matrix = np.zeros((len(pca_subensemble_i), latent_dim, len(pca_subensemble_i), latent_dim))
#     for i in range(len(pca_subensemble_i)):
#         for j in range(latent_dim):
#             for i_ in range(len(pca_subensemble_i)):
#                 for j_ in range(latent_dim):
#                     scalar_product_matrix[i, j, i_, j_] = np.abs(v[i, j].dot(v[i_, j_]))
                    
#     min_value = np.finfo(np.float).max
#     for i in range(len(pca_subensemble_i)):
#         for j in range(latent_dim):
#             for i_ in range(len(pca_subensemble_i)):
#                 for j_ in range(latent_dim):
#                     if i != i_ and min_value > scalar_product_matrix[i, j, i_, j_]:
#                         min_value = scalar_product_matrix[i, j, i_, j_]
#                         scalar_product_matrix[i, j, i_, j_] = np.finfo(np.float).max
#                         v1 = v[i, j]
#                         v2 = v[i_, j_]
    
#     return np.concatenate((v1[np.newaxis], v2[np.newaxis]))


if __name__ == '__main__':
    main()
