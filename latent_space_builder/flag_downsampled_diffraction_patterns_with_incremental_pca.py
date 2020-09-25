import os
import sys
import time
import argparse
import json

import pickle

import numpy as np
np.random.seed(13)

from sklearn.decomposition import IncrementalPCA

from sklearn.neighbors import KernelDensity
from scipy.spatial import distance

from sklearn.covariance import EllipticEnvelope

import h5py as h5

from tqdm import tqdm

"""
Deeban Ramalingam (deebanr@slac.stanford.edu)

python flag_downsampled_diffraction_patterns_with_incremental_pca.py --config flag-downsampled-diffraction-patterns-with-incremental-pca.json --dataset 3iyf-10K-mixed-hit-99-single-hits-labeled
"""

def main():
    user_input = parse_input_arguments(sys.argv)
    incremental_pca_config_file = user_input["config"]
    dataset_name = user_input["dataset"]
    
    with open(incremental_pca_config_file) as incremental_pca_config_file_handle:
        incremental_pca_config_params = json.load(incremental_pca_config_file_handle)

    if dataset_name not in incremental_pca_config_params:
        raise Exception("Dataset {} not in Config file.".format(dataset_name))
    
    dataset_params = incremental_pca_config_params[dataset_name]
    
    downsampled_diffraction_patterns_h5_file = dataset_params["downsampledDiffractionPatternsH5File"]
    downsampled_diffraction_patterns_h5_file_handle = h5.File(downsampled_diffraction_patterns_h5_file, 'r')
    downsampled_diffraction_patterns = downsampled_diffraction_patterns_h5_file_handle["downsampled_diffraction_patterns"]

    downsampled_diffraction_pattern_height = dataset_params["downsampledDiffractionPatternHeight"]
    downsampled_diffraction_pattern_width = dataset_params["downsampledDiffractionPatternWidth"]

    num_downsampled_diffraction_patterns = dataset_params["numDiffractionPatterns"]    

    num_downsampled_diffraction_patterns_to_fit_per_batch = dataset_params["numDownsampledDiffractionPatternsToFitPerBatch"]
    assert num_downsampled_diffraction_patterns % num_downsampled_diffraction_patterns_to_fit_per_batch == 0
    num_batches_of_downsampled_diffraction_patterns = num_downsampled_diffraction_patterns // num_downsampled_diffraction_patterns_to_fit_per_batch 

    incremental_pca_results_dir = dataset_params["incrementalPcaResultsDir"]
    if not os.path.exists(incremental_pca_results_dir):
        os.makedirs(incremental_pca_results_dir)

    save_incremental_pca_model = dataset_params["saveIncrementalPcaModel"]
    num_iters_to_save_incremental_pca_model = dataset_params["numItersToSaveIncrementalPcaModel"]

    project_downsampled_diffraction_patterns_seen_thus_far = dataset_params["projectDownsampledDiffractionPatternsSeenThusFar"]
    num_iters_to_project_downsampled_diffraction_patterns_seen_thus_far = dataset_params["numItersToProjectDownsampledDiffractionPatternsSeenThusFar"]
    
    measure_convergence_for_downsampled_diffraction_patterns_seen_thus_far = dataset_params["measureConvergenceForDownsampledDiffractionPatternsSeenThusFar"]
    num_iters_to_measure_convergence_for_downsampled_diffraction_patterns_seen_thus_far = dataset_params["numItersToMeasureConvergenceForDownsampledDiffractionPatternsSeenThusFar"]

    option_for_measuring_convergence = dataset_params["optionForMeasuringConvergence"]

    if option_for_measuring_convergence == "meanSquaredReconstructionErrorForDiffractionPatternsSeenThusFarUsingLastNIters":
        num_previous_downsampled_diffraction_pattern_batches_to_consider_for_mean_square_reconstruction_error_convergence_measure = dataset_params["numPreviousDownsampledDiffractionPatternBatchesToConsiderForMeanSquareReconstructionErrorConvergenceMeasure"]

    batch_number_converged = None
    convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far = None    

    minimum_convergence_measure_to_start_flagging_downsampled_diffraction_pattern_outliers = dataset_params["minimumConvergenceMeasureToStartFlaggingDownsampledDiffractionPatternOutliers"]

    num_iters_to_update_latent_space_after_convergence = dataset_params["numItersToUpdateLatentSpaceAfterConvergence"]

    num_latent_dims = dataset_params["numLatentDims"]

    verbose_output = dataset_params["verboseOutput"]

    if not verbose_output:
        progress_bar_for_num_batches_of_downsampled_diffraction_patterns_processed = tqdm(total = num_batches_of_downsampled_diffraction_patterns)

    new_batch_number_to_load_incremental_pca_model = dataset_params["newBatchNumberToLoadIncrementalPcaModel"]
    if new_batch_number_to_load_incremental_pca_model is not None:

        assert new_batch_number_to_load_incremental_pca_model > 0
        
        incremental_pca_model_file = os.path.join(incremental_pca_results_dir, "incremental-pca-incremental-pca-model-dataset_name={}-downsampled_shape={}x{}-num_diffraction_patterns={}-batch_size={}-batch_number={}.pkl".format(dataset_name, downsampled_diffraction_pattern_height, downsampled_diffraction_pattern_width, num_downsampled_diffraction_patterns, num_downsampled_diffraction_patterns_to_fit_per_batch, new_batch_number_to_load_incremental_pca_model))
            
        with open(incremental_pca_model_file, 'rb') as incremental_pca_model_file_handle:
            incremental_pca = pickle.load(incremental_pca_model_file_handle)

        if verbose_output:
            print("Load incremental_pca_model from: {}".format(incremental_pca_model_file))    

        if measure_convergence_for_downsampled_diffraction_patterns_seen_thus_far and new_batch_number_to_load_incremental_pca_model % num_iters_to_measure_convergence_for_downsampled_diffraction_patterns_seen_thus_far == 0:

            all_downsampled_diffraction_patterns_seen_thus_far = downsampled_diffraction_patterns[:new_batch_number_to_load_incremental_pca_model * num_downsampled_diffraction_patterns_to_fit_per_batch].reshape((new_batch_number_to_load_incremental_pca_model * num_downsampled_diffraction_patterns_to_fit_per_batch, -1))
            
            previous_incremental_pca_mean_for_diffraction_patterns_seen_thus_far = incremental_pca.mean_
            previous_incremental_pca_components_for_diffraction_patterns_seen_thus_far = incremental_pca.components_
            
            previous_projection_for_all_downsampled_diffraction_patterns_seen_thus_far = np.dot(all_downsampled_diffraction_patterns_seen_thus_far - previous_incremental_pca_mean_for_diffraction_patterns_seen_thus_far, previous_incremental_pca_components_for_diffraction_patterns_seen_thus_far.T)
            
            previous_kernel_density_estimate_for_all_downsampled_diffraction_patterns_seen_thus_far = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(previous_projection_for_all_downsampled_diffraction_patterns_seen_thus_far)

        convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far_file = os.path.join(incremental_pca_results_dir, "incremental-pca-convergence-measure-for-all-downsampled-diffraction-patterns-seen-dataset_name={}-downsampled_shape={}x{}-num_diffraction_patterns={}-batch_size={}-batch_number={}.npy".format(dataset_name, downsampled_diffraction_pattern_height, downsampled_diffraction_pattern_width, num_downsampled_diffraction_patterns, num_downsampled_diffraction_patterns_to_fit_per_batch, new_batch_number_to_load_incremental_pca_model))
        
        if os.path.exists(convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far_file):
            convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far = np.load(convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far_file)   

            if verbose_output:
                print("Load convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far={:.4f} from: {}".format(np.asscalar(convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far), convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far_file))
        
        new_batch_number = new_batch_number_to_load_incremental_pca_model

        if not verbose_output:
            if convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far is not None:
                progress_bar_for_num_batches_of_downsampled_diffraction_patterns_processed.set_postfix_str("convergence measure={:.4f}".format(np.asscalar(convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far)), refresh=False)
            
            progress_bar_for_num_batches_of_downsampled_diffraction_patterns_processed.update(new_batch_number)

    else:
        
        incremental_pca = IncrementalPCA(n_components=num_latent_dims)
        new_batch_number = 0
        convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far = np.asarray(np.inf)
    
    while new_batch_number < num_batches_of_downsampled_diffraction_patterns and convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far >= minimum_convergence_measure_to_start_flagging_downsampled_diffraction_pattern_outliers:

        new_downsampled_diffraction_patterns_batch_to_fit = downsampled_diffraction_patterns[new_batch_number * num_downsampled_diffraction_patterns_to_fit_per_batch : (new_batch_number + 1) * num_downsampled_diffraction_patterns_to_fit_per_batch].reshape((num_downsampled_diffraction_patterns_to_fit_per_batch, -1))

        tic = time.time()
        incremental_pca.partial_fit(new_downsampled_diffraction_patterns_batch_to_fit)
        toc = time.time()
        time_taken_to_update_incremental_pca_with_new_batch = toc - tic
        
        time_taken_to_update_incremental_pca_with_new_batch_file = os.path.join(incremental_pca_results_dir, "incremental-pca-time-taken-to-update-incremental-pca-with-new-batch-dataset_name={}-downsampled_shape={}x{}-num_diffraction_patterns={}-batch_size={}-batch_number={}.npy".format(dataset_name, downsampled_diffraction_pattern_height, downsampled_diffraction_pattern_width, num_downsampled_diffraction_patterns, num_downsampled_diffraction_patterns_to_fit_per_batch, new_batch_number + 1))
        np.save(time_taken_to_update_incremental_pca_with_new_batch_file, time_taken_to_update_incremental_pca_with_new_batch)
        
        if verbose_output:
            print("Save time_taken_to_update_incremental_pca_with_new_batch={:.4f} to: {}".format(time_taken_to_update_incremental_pca_with_new_batch, time_taken_to_update_incremental_pca_with_new_batch_file))

        if not verbose_output:
            if convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far is not None:
                progress_bar_for_num_batches_of_downsampled_diffraction_patterns_processed.set_postfix_str("convergence measure={:.4f}".format(np.asscalar(convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far)), refresh=False)

            progress_bar_for_num_batches_of_downsampled_diffraction_patterns_processed.update(1)

        if save_incremental_pca_model and (new_batch_number + 1) % num_iters_to_save_incremental_pca_model == 0:

            incremental_pca_model_file = os.path.join(incremental_pca_results_dir, "incremental-pca-incremental-pca-model-dataset_name={}-downsampled_shape={}x{}-num_diffraction_patterns={}-batch_size={}-batch_number={}.pkl".format(dataset_name, downsampled_diffraction_pattern_height, downsampled_diffraction_pattern_width, num_downsampled_diffraction_patterns, num_downsampled_diffraction_patterns_to_fit_per_batch, new_batch_number + 1))

            with open(incremental_pca_model_file, 'wb') as incremental_pca_model_file_handle:
                pickle.dump(incremental_pca, incremental_pca_model_file_handle)

            if verbose_output:
                print("Saved incremental_pca_model to: {}".format(incremental_pca_model_file))

        if project_downsampled_diffraction_patterns_seen_thus_far and (new_batch_number + 1) % num_iters_to_project_downsampled_diffraction_patterns_seen_thus_far == 0:

            num_downsampled_diffraction_patterns_seen_thus_far = (new_batch_number + 1) * num_downsampled_diffraction_patterns_to_fit_per_batch
            downsampled_diffraction_patterns_seen_thus_far = downsampled_diffraction_patterns[:num_downsampled_diffraction_patterns_seen_thus_far].reshape(num_downsampled_diffraction_patterns_seen_thus_far, -1)

            incremental_pca_mean_for_diffraction_patterns_seen_thus_far = incremental_pca.mean_
            incremental_pca_components_for_diffraction_patterns_seen_thus_far = incremental_pca.components_

            tic = time.time()
            latent_space_projection_updated_with_downsampled_diffraction_patterns_seen_thus_far = np.dot(downsampled_diffraction_patterns_seen_thus_far - incremental_pca_mean_for_diffraction_patterns_seen_thus_far, incremental_pca_components_for_diffraction_patterns_seen_thus_far.T)
            toc = time.time()
            time_taken_to_project_downsampled_diffraction_patterns_seen_thus_far = toc - tic
            
            time_taken_to_project_downsampled_diffraction_patterns_seen_thus_far_file = os.path.join(incremental_pca_results_dir, "incremental-pca-time-taken-to-project-downsampled-diffraction-patterns-seen-thus-far-dataset_name={}-downsampled_shape={}x{}-num_diffraction_patterns={}-batch_size={}-batch_number={}.npy".format(dataset_name, downsampled_diffraction_pattern_height, downsampled_diffraction_pattern_width, num_downsampled_diffraction_patterns, num_downsampled_diffraction_patterns_to_fit_per_batch, new_batch_number + 1))
            np.save(time_taken_to_project_downsampled_diffraction_patterns_seen_thus_far_file, time_taken_to_project_downsampled_diffraction_patterns_seen_thus_far)
            
            if verbose_output:
                print("Saved time_taken_to_project_downsampled_diffraction_patterns_seen_thus_far={:.4f} to: {}".format(time_taken_to_project_downsampled_diffraction_patterns_seen_thus_far, time_taken_to_project_downsampled_diffraction_patterns_seen_thus_far_file))

        if measure_convergence_for_downsampled_diffraction_patterns_seen_thus_far:

            if option_for_measuring_convergence == "meanSquaredReconstructionErrorForDiffractionPatternsSeenThusFarUsingLastNIters":

                if (new_batch_number + 1) % num_iters_to_measure_convergence_for_downsampled_diffraction_patterns_seen_thus_far == 0 and (new_batch_number + 1) >= num_previous_downsampled_diffraction_pattern_batches_to_consider_for_mean_square_reconstruction_error_convergence_measure:

                    num_downsampled_diffraction_patterns_seen_thus_far = (new_batch_number + 1) * num_downsampled_diffraction_patterns_to_fit_per_batch

                    all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters = downsampled_diffraction_patterns[num_downsampled_diffraction_patterns_seen_thus_far - num_previous_downsampled_diffraction_pattern_batches_to_consider_for_mean_square_reconstruction_error_convergence_measure * num_downsampled_diffraction_patterns_to_fit_per_batch:num_downsampled_diffraction_patterns_seen_thus_far].reshape(num_previous_downsampled_diffraction_pattern_batches_to_consider_for_mean_square_reconstruction_error_convergence_measure * num_downsampled_diffraction_patterns_to_fit_per_batch, -1)

                    incremental_pca_mean_for_diffraction_patterns_seen_thus_far = incremental_pca.mean_
                    incremental_pca_components_for_diffraction_patterns_seen_thus_far = incremental_pca.components_

                    tic = time.time()

                    projection_for_all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters = np.dot(all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters - incremental_pca_mean_for_diffraction_patterns_seen_thus_far, incremental_pca_components_for_diffraction_patterns_seen_thus_far.T)

                    # convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far = 0.0
                    # for all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters_index in range(num_previous_downsampled_diffraction_pattern_batches_to_consider_for_mean_square_reconstruction_error_convergence_measure * num_downsampled_diffraction_patterns_to_fit_per_batch):
                    #     incremental_pca_reconstruction_for_downsampled_diffraction_pattern_i = np.dot(projection_for_all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters[all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters_index, :], incremental_pca_components_for_diffraction_patterns_seen_thus_far) + incremental_pca_mean_for_diffraction_patterns_seen_thus_far
                    #     convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far += ((incremental_pca_reconstruction_for_downsampled_diffraction_pattern_i - all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters[all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters_index, :]) ** 2).sum()

                    # convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far /= num_previous_downsampled_diffraction_pattern_batches_to_consider_for_mean_square_reconstruction_error_convergence_measure * num_downsampled_diffraction_patterns_to_fit_per_batch * downsampled_diffraction_pattern_height * downsampled_diffraction_pattern_width

                    incremental_pca_reconstruction_for_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters = np.dot(projection_for_all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters, incremental_pca_components_for_diffraction_patterns_seen_thus_far) + incremental_pca_mean_for_diffraction_patterns_seen_thus_far

                    convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far = ((incremental_pca_reconstruction_for_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters - all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters) ** 2).mean()

                    toc = time.time()
                    time_taken_to_compute_incremental_pca_mean_squared_reconstruction_error_for_all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters = toc - tic
                    time_taken_to_compute_incremental_pca_mean_squared_reconstruction_error_for_all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters_file = os.path.join(incremental_pca_results_dir, "time-taken-to-compute-reconstruction-mse-for-downsampled-diffraction-patterns-seen-usings-last-n-iters-dataset_name={}-downsampled_shape={}x{}-num_diffraction_patterns={}-batch_size={}-batch_number={}.npy".format(dataset_name, downsampled_diffraction_pattern_height, downsampled_diffraction_pattern_width, num_downsampled_diffraction_patterns, num_downsampled_diffraction_patterns_to_fit_per_batch, new_batch_number + 1))
                    np.save(time_taken_to_compute_incremental_pca_mean_squared_reconstruction_error_for_all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters_file, time_taken_to_compute_incremental_pca_mean_squared_reconstruction_error_for_all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters)

                    if verbose_output:
                        print("Saved time_taken_to_compute_incremental_pca_mean_squared_reconstruction_error_for_all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters={:.4f} to: {}".format(time_taken_to_compute_incremental_pca_mean_squared_reconstruction_error_for_all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters, time_taken_to_compute_incremental_pca_mean_squared_reconstruction_error_for_all_downsampled_diffraction_patterns_seen_thus_far_for_last_n_iters_file))

                    incremental_pca_mean_squared_reconstruction_error_for_diffraction_patterns_seen_thus_far_using_last_n_iters_file = os.path.join(incremental_pca_results_dir, "incremental-pca-reconstruction-mse-for-downsampled-diffraction-patterns-seen-using-last-n-iters-dataset_name={}-downsampled_shape={}x{}-num_diffraction_patterns={}-batch_size={}-batch_number={}.npy".format(dataset_name, downsampled_diffraction_pattern_height, downsampled_diffraction_pattern_width, num_downsampled_diffraction_patterns, num_downsampled_diffraction_patterns_to_fit_per_batch, new_batch_number + 1))
                    np.save(incremental_pca_mean_squared_reconstruction_error_for_diffraction_patterns_seen_thus_far_using_last_n_iters_file, convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far)

                    if verbose_output:
                        print("Saved incremental_pca_mean_squared_reconstruction_error_for_diffraction_patterns_seen_thus_far_using_last_n_iters={:.4f} to: {}".format(convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far, incremental_pca_mean_squared_reconstruction_error_for_diffraction_patterns_seen_thus_far_using_last_n_iters_file))

            elif option_for_measuring_convergence == "jensenShannonDivergenceForDiffractionPatternsSeenThusFar":

                if (new_batch_number + 1) == num_iters_to_measure_convergence_for_downsampled_diffraction_patterns_seen_thus_far:

                    all_downsampled_diffraction_patterns_seen_thus_far = downsampled_diffraction_patterns[:(new_batch_number + 1) * num_downsampled_diffraction_patterns_to_fit_per_batch].reshape(((new_batch_number + 1) * num_downsampled_diffraction_patterns_to_fit_per_batch, -1))
                    
                    previous_incremental_pca_mean_for_diffraction_patterns_seen_thus_far = incremental_pca.mean_
                    previous_incremental_pca_components_for_diffraction_patterns_seen_thus_far = incremental_pca.components_
                    
                    previous_projection_for_all_downsampled_diffraction_patterns_seen_thus_far = np.dot(all_downsampled_diffraction_patterns_seen_thus_far - previous_incremental_pca_mean_for_diffraction_patterns_seen_thus_far, previous_incremental_pca_components_for_diffraction_patterns_seen_thus_far.T)
 
                    previous_kernel_density_estimate_for_all_downsampled_diffraction_patterns_seen_thus_far = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(previous_projection_for_all_downsampled_diffraction_patterns_seen_thus_far)

                elif (new_batch_number + 1) % num_iters_to_measure_convergence_for_downsampled_diffraction_patterns_seen_thus_far == 0:

                    all_downsampled_diffraction_patterns_seen_thus_far = downsampled_diffraction_patterns[:(new_batch_number + 1) * num_downsampled_diffraction_patterns_to_fit_per_batch].reshape(((new_batch_number + 1) * num_downsampled_diffraction_patterns_to_fit_per_batch, -1))
                    
                    current_incremental_pca_mean_for_diffraction_patterns_seen_thus_far = incremental_pca.mean_
                    current_incremental_pca_components_for_diffraction_patterns_seen_thus_far = incremental_pca.components_

                    current_projection_for_all_downsampled_diffraction_patterns_seen_thus_far = np.dot(all_downsampled_diffraction_patterns_seen_thus_far - current_incremental_pca_mean_for_diffraction_patterns_seen_thus_far, current_incremental_pca_components_for_diffraction_patterns_seen_thus_far.T)
                    
                    tic = time.time()

                    current_kernel_density_estimate_for_all_downsampled_diffraction_patterns_seen_thus_far = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(current_projection_for_all_downsampled_diffraction_patterns_seen_thus_far)
                    
                    current_probability_density_estimate_for_all_downsampled_diffraction_patterns_seen_thus_far = np.exp(current_kernel_density_estimate_for_all_downsampled_diffraction_patterns_seen_thus_far.score_samples(current_projection_for_all_downsampled_diffraction_patterns_seen_thus_far))
                    previous_probability_density_estimate_for_all_downsampled_diffraction_patterns_seen_thus_far = np.exp(previous_kernel_density_estimate_for_all_downsampled_diffraction_patterns_seen_thus_far.score_samples(current_projection_for_all_downsampled_diffraction_patterns_seen_thus_far))

                    convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far = distance.jensenshannon(previous_probability_density_estimate_for_all_downsampled_diffraction_patterns_seen_thus_far, current_probability_density_estimate_for_all_downsampled_diffraction_patterns_seen_thus_far)
                    
                    toc = time.time()
                    time_taken_to_compute_convergence_measure_using_all_downsampled_diffraction_patterns_seen_thus_far = toc - tic
                    time_taken_to_compute_convergence_measure_using_all_downsampled_diffraction_patterns_seen_thus_far_file = os.path.join(incremental_pca_results_dir, "time-taken-to-compute-convergence-measure-using-all-downsampled-diffraction-patterns-seen-dataset_name={}-downsampled_shape={}x{}-num_diffraction_patterns={}-batch_size={}-batch_number={}.npy".format(dataset_name, downsampled_diffraction_pattern_height, downsampled_diffraction_pattern_width, num_downsampled_diffraction_patterns, num_downsampled_diffraction_patterns_to_fit_per_batch, new_batch_number + 1))
                    np.save(time_taken_to_compute_convergence_measure_using_all_downsampled_diffraction_patterns_seen_thus_far_file, time_taken_to_compute_convergence_measure_using_all_downsampled_diffraction_patterns_seen_thus_far)
                    
                    if verbose_output:
                        print("Saved time_taken_to_compute_convergence_measure_using_all_downsampled_diffraction_patterns_seen_thus_far={:.4f} to: {}".format(time_taken_to_compute_convergence_measure_using_all_downsampled_diffraction_patterns_seen_thus_far, time_taken_to_compute_convergence_measure_using_all_downsampled_diffraction_patterns_seen_thus_far_file))

                    convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far_file = os.path.join(incremental_pca_results_dir, "incremental-pca-convergence-measure-for-all-downsampled-diffraction-patterns-seen-dataset_name={}-downsampled_shape={}x{}-num_diffraction_patterns={}-batch_size={}-batch_number={}.npy".format(dataset_name, downsampled_diffraction_pattern_height, downsampled_diffraction_pattern_width, num_downsampled_diffraction_patterns, num_downsampled_diffraction_patterns_to_fit_per_batch, new_batch_number + 1))
                    np.save(convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far_file, convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far)
                    
                    if verbose_output:
                        print("Saved convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far={:.4f} to: {}".format(convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far, convergence_measure_for_downsampled_diffraction_patterns_seen_thus_far_file))

                    previous_incremental_pca_mean_for_diffraction_patterns_seen_thus_far = current_incremental_pca_mean_for_diffraction_patterns_seen_thus_far
                    previous_incremental_pca_components_for_diffraction_patterns_seen_thus_far = current_incremental_pca_components_for_diffraction_patterns_seen_thus_far

                    previous_projection_for_all_downsampled_diffraction_patterns_seen_thus_far = current_projection_for_all_downsampled_diffraction_patterns_seen_thus_far
                    
                    previous_kernel_density_estimate_for_all_downsampled_diffraction_patterns_seen_thus_far = current_kernel_density_estimate_for_all_downsampled_diffraction_patterns_seen_thus_far


            else:

                raise Exception("[optionForMeasuringConvergence] not recognized or defined. Please use one of the following: [meanSquaredReconstructionErrorForDiffractionPatternsSeenThusFarUsingLastNIters] or [jensenShannonDivergenceForDiffractionPatternsSeenThusFar]")

        new_batch_number += 1

    batch_number_converged = new_batch_number - 1

    num_downsampled_diffraction_patterns_seen_thus_far = new_batch_number * num_downsampled_diffraction_patterns_to_fit_per_batch

    downsampled_diffraction_patterns_seen_thus_far = downsampled_diffraction_patterns[:num_downsampled_diffraction_patterns_seen_thus_far].reshape(num_downsampled_diffraction_patterns_seen_thus_far, -1)

    incremental_pca_mean_for_diffraction_patterns_seen_thus_far = incremental_pca.mean_
    incremental_pca_components_for_diffraction_patterns_seen_thus_far = incremental_pca.components_
    
    latent_space_projection_updated_with_downsampled_diffraction_patterns_seen_thus_far = np.dot(downsampled_diffraction_patterns_seen_thus_far - incremental_pca_mean_for_diffraction_patterns_seen_thus_far, incremental_pca_components_for_diffraction_patterns_seen_thus_far.T)

    latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file = os.path.join(incremental_pca_results_dir, "latent-space-projection-for-downsampled-diffraction-patterns-seen-thus-far-dataset_name={}-downsampled_shape={}x{}-num_diffraction_patterns={}-batch_size={}-batch_number_converged={}.hdf5".format(dataset_name, downsampled_diffraction_pattern_height, downsampled_diffraction_pattern_width, num_downsampled_diffraction_patterns, num_downsampled_diffraction_patterns_to_fit_per_batch, batch_number_converged + 1))
    latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_key = "latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far"

    latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle = h5.File(latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file, 'w')
    latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle.create_dataset(latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_key, (num_downsampled_diffraction_patterns_seen_thus_far, num_latent_dims), dtype='f', data=latent_space_projection_updated_with_downsampled_diffraction_patterns_seen_thus_far)
    latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle.close()

    elliptic_envelope_for_outlier_prediction = EllipticEnvelope(random_state=13).fit(latent_space_projection_updated_with_downsampled_diffraction_patterns_seen_thus_far)
    elliptic_envelope_outlier_predictions_for_downsampled_diffraction_patterns_seen_thus_far = elliptic_envelope_for_outlier_prediction.predict(latent_space_projection_updated_with_downsampled_diffraction_patterns_seen_thus_far)

    elliptic_envelope_outlier_predictions_for_downsampled_diffraction_patterns_seen_thus_far_with_labels_adjusted_to_binary = (elliptic_envelope_outlier_predictions_for_downsampled_diffraction_patterns_seen_thus_far + 1) // 2

    elliptic_envelope_single_hit_indices_for_downsampled_diffraction_patterns_seen_thus_far = np.where(elliptic_envelope_outlier_predictions_for_downsampled_diffraction_patterns_seen_thus_far_with_labels_adjusted_to_binary == 1)[0]
    elliptic_envelope_outlier_indices_for_downsampled_diffraction_patterns_seen_thus_far = np.where(elliptic_envelope_outlier_predictions_for_downsampled_diffraction_patterns_seen_thus_far_with_labels_adjusted_to_binary == 0)[0]        

    elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far = np.zeros(num_downsampled_diffraction_patterns_seen_thus_far, dtype=np.bool)
    elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far[elliptic_envelope_single_hit_indices_for_downsampled_diffraction_patterns_seen_thus_far] = True
    elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far[elliptic_envelope_outlier_indices_for_downsampled_diffraction_patterns_seen_thus_far] = False

    elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_file = os.path.join(incremental_pca_results_dir, "elliptic-envelope-outlier-predictions-for-downsampled-diffraction-patterns-seen-dataset_name={}-downsampled_shape={}x{}-num_diffraction_patterns={}-batch_size={}-batch_number_converged={}.hdf5".format(dataset_name, downsampled_diffraction_pattern_height, downsampled_diffraction_pattern_width, num_downsampled_diffraction_patterns, num_downsampled_diffraction_patterns_to_fit_per_batch, batch_number_converged + 1))
    elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_key = "elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far"
    
    elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle = h5.File(elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_file, 'w')
    elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle.create_dataset(elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_key, (num_downsampled_diffraction_patterns_seen_thus_far,), dtype='u1', data=elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far)
    elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle.close()

    while new_batch_number < num_batches_of_downsampled_diffraction_patterns:

        num_downsampled_diffraction_patterns_seen_thus_far = num_downsampled_diffraction_patterns_seen_thus_far + num_downsampled_diffraction_patterns_to_fit_per_batch

        new_downsampled_diffraction_patterns_batch_to_flag_outliers = downsampled_diffraction_patterns[new_batch_number * num_downsampled_diffraction_patterns_to_fit_per_batch : (new_batch_number + 1) * num_downsampled_diffraction_patterns_to_fit_per_batch].reshape((num_downsampled_diffraction_patterns_to_fit_per_batch, -1))

        if new_batch_number > batch_number_converged + 1 and (new_batch_number + 1) % num_iters_to_update_latent_space_after_convergence == 0:
            
            incremental_pca.partial_fit(new_downsampled_diffraction_patterns_batch_to_flag_outliers)

            incremental_pca_mean_for_diffraction_patterns_seen_thus_far = incremental_pca.mean_
            incremental_pca_components_for_diffraction_patterns_seen_thus_far = incremental_pca.components_

            incremental_pca_model_file = os.path.join(incremental_pca_results_dir, "incremental-pca-after-convergence-incremental-pca-model-dataset_name={}-downsampled_shape={}x{}-num_diffraction_patterns={}-batch_size={}-batch_number={}.pkl".format(dataset_name, downsampled_diffraction_pattern_height, downsampled_diffraction_pattern_width, num_downsampled_diffraction_patterns, num_downsampled_diffraction_patterns_to_fit_per_batch, new_batch_number + 1))
            
            with open(incremental_pca_model_file, 'wb') as incremental_pca_model_file_handle:
                pickle.dump(incremental_pca, incremental_pca_model_file_handle)

            if verbose_output:
                print("Saved incremental_pca_model to: {}".format(incremental_pca_model_file))

            latent_space_projection_of_new_downsampled_diffraction_patterns_batch_to_flag_outliers = np.dot(new_downsampled_diffraction_patterns_batch_to_flag_outliers - incremental_pca_mean_for_diffraction_patterns_seen_thus_far, incremental_pca_components_for_diffraction_patterns_seen_thus_far.T)

            latent_space_projection_updated_with_downsampled_diffraction_patterns_seen_thus_far = np.concatenate((latent_space_projection_updated_with_downsampled_diffraction_patterns_seen_thus_far, latent_space_projection_of_new_downsampled_diffraction_patterns_batch_to_flag_outliers))
            
            latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle = h5.File(latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file, 'r+')

            latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far = latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle[latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_key][:]
            del latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle[latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_key]
            latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle.create_dataset(latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_key, (num_downsampled_diffraction_patterns_seen_thus_far, num_latent_dims), dtype='f', data=latent_space_projection_updated_with_downsampled_diffraction_patterns_seen_thus_far)

            latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle.close()

            elliptic_envelope_for_outlier_prediction = EllipticEnvelope(random_state=13).fit(latent_space_projection_updated_with_downsampled_diffraction_patterns_seen_thus_far)      

        else:
            
            latent_space_projection_of_new_downsampled_diffraction_patterns_batch_to_flag_outliers = np.dot(new_downsampled_diffraction_patterns_batch_to_flag_outliers - incremental_pca_mean_for_diffraction_patterns_seen_thus_far, incremental_pca_components_for_diffraction_patterns_seen_thus_far.T)

            latent_space_projection_updated_with_downsampled_diffraction_patterns_seen_thus_far = np.concatenate((latent_space_projection_updated_with_downsampled_diffraction_patterns_seen_thus_far, latent_space_projection_of_new_downsampled_diffraction_patterns_batch_to_flag_outliers))
            
            latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle = h5.File(latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file, 'r+')

            latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far = latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle[latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_key][:]
            del latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle[latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_key]
            latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle.create_dataset(latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_key, (num_downsampled_diffraction_patterns_seen_thus_far, num_latent_dims), dtype='f', data=latent_space_projection_updated_with_downsampled_diffraction_patterns_seen_thus_far)

            latent_space_projection_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle.close()

        elliptic_envelope_outlier_predictions_for_new_downsampled_diffraction_patterns_batch_to_flag_outliers = elliptic_envelope_for_outlier_prediction.predict(latent_space_projection_of_new_downsampled_diffraction_patterns_batch_to_flag_outliers)

        elliptic_envelope_outlier_predictions_for_new_downsampled_diffraction_patterns_batch_to_flag_outliers_with_labels_adjusted_to_binary = (elliptic_envelope_outlier_predictions_for_new_downsampled_diffraction_patterns_batch_to_flag_outliers + 1) // 2

        elliptic_envelope_single_hit_indices_for_new_downsampled_diffraction_patterns_batch_to_flag_outliers = np.where(elliptic_envelope_outlier_predictions_for_new_downsampled_diffraction_patterns_batch_to_flag_outliers_with_labels_adjusted_to_binary == 1)[0]
        elliptic_envelope_outlier_indices_for_new_downsampled_diffraction_patterns_batch_to_flag_outliers = np.where(elliptic_envelope_outlier_predictions_for_new_downsampled_diffraction_patterns_batch_to_flag_outliers_with_labels_adjusted_to_binary == 0)[0]

        elliptic_envelope_outlier_prediction_mask_for_new_downsampled_diffraction_patterns_batch_to_flag_outliers = np.zeros(num_downsampled_diffraction_patterns_to_fit_per_batch, dtype=np.bool)
        elliptic_envelope_outlier_prediction_mask_for_new_downsampled_diffraction_patterns_batch_to_flag_outliers[elliptic_envelope_single_hit_indices_for_new_downsampled_diffraction_patterns_batch_to_flag_outliers] = True
        elliptic_envelope_outlier_prediction_mask_for_new_downsampled_diffraction_patterns_batch_to_flag_outliers[elliptic_envelope_outlier_indices_for_new_downsampled_diffraction_patterns_batch_to_flag_outliers] = False
    
        elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle = h5.File(elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_file, 'r+')

        elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_boolean_array = elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle[elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_key][:]
        del elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle[elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_key]
        elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_boolean_array = np.concatenate((elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_boolean_array, elliptic_envelope_outlier_prediction_mask_for_new_downsampled_diffraction_patterns_batch_to_flag_outliers))
        elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle.create_dataset(elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_key, (num_downsampled_diffraction_patterns_seen_thus_far,), dtype='u1', data=elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_boolean_array)

        elliptic_envelope_outlier_prediction_mask_for_downsampled_diffraction_patterns_seen_thus_far_h5_file_handle.close()

        if not verbose_output:
            progress_bar_for_num_batches_of_downsampled_diffraction_patterns_processed.update(1)

        new_batch_number += 1

    if not verbose_output:
        progress_bar_for_num_batches_of_downsampled_diffraction_patterns_processed.close()

    downsampled_diffraction_patterns_h5_file_handle.close()

def parse_input_arguments(args):
    del args[0]
    
    parse = argparse.ArgumentParser()
    parse.add_argument('-c', '--config', type=str, help='JSON Config file')
    parse.add_argument('-d', '--dataset', type=str, help='Dataset name: 3iyf-10K-mixed-hit-99-single-hits-labeled')

    return vars(parse.parse_args(args))
    
if __name__ == '__main__':
    main()
