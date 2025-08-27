import os
import numpy as np
from deepwonder.VM.utils import *
from natsort import natsorted


def run_view_merging_pipeline(
        DATA,
        SAVE,
        psffit_matrix_file,
        upsample_rate=2,
        dz=1.5,
        Nnum=15,
        min_view_num=5,
        min_loc_num=3,
        cen_id=113,
        cutoff_spatial=20,
        cutoff_temporal=0.75
):
    """
        Run the spatiotemporal view merging + 3D localization pipeline

        Parameters:
        DATA (str): Directory containing the input .mat files
        SAVE (str): Directory to save the results
        psffit_matrix_file (str): Path to the PSF fitting matrix file
        upsample_rate (int): Upsampling factor
        dz (float): Step size along the Z-axis
        Nnum (int): Maximum number of fields of view
        min_view_num (int): Minimum number of fields of view for clustering
        min_loc_num (int): Minimum number of localization events for Z estimation
        cen_id (int): ID of the central field of view
        cutoff_spatial (float): Spatial distance threshold
        cutoff_temporal (float): Temporal correlation threshold

        Returns:
        spatial_3D (np.ndarray): 3D spatial coordinates of neurons
        valid_single_neuron_trace (np.ndarray): Temporal traces of neurons
    """

    # Step 1: Load all data
    print(f"Loading data from {DATA}...")
    os.makedirs(SAVE, exist_ok=True)
    all_mat_path = os.path.join(SAVE, 'all_data.mat')

    if os.path.exists(all_mat_path):
        print(f"{all_mat_path} exists! Loading directly...")
        loaded_data = loadmat_auto(all_mat_path)
        T_trace_array = loaded_data['all_trace']
        S_center_array = loaded_data['S_center_list']
        id_array = loaded_data['all_index']
        view_list = loaded_data['all_view_num']
    else:
        print(f"{all_mat_path} does NOT exist! Loading each .mat file...")

        files_name = [f for f in os.listdir(DATA) if f.endswith('.mat')]
        files_name = natsorted(files_name)

        # extract view IDs
        view_list = []
        for filename in files_name:
            ind1 = filename.index('_view_')
            subtmp = filename[ind1 + 6:]
            ind2 = subtmp.index('.')
            view_id = int(subtmp[:ind2])
            view_list.append(view_id)

        # storage containers
        T_trace_list = []
        S_center_list = []
        id_list = []

        # load files one by one
        for filename in files_name:
            print(f"{filename} loaded ...")
            data = loadmat_auto(os.path.join(DATA, filename))
            final_mask_list = data['final_mask_list']

            for neuron_id in range(final_mask_list.shape[1]):
                Temporal_trace = final_mask_list[0, neuron_id]['trace'][0][0]
                Spatial_center = final_mask_list[0, neuron_id]['centroid'][0][0]

                T_trace_list.append(np.squeeze(Temporal_trace))
                S_center_list.append(np.squeeze(Spatial_center))
                id_list.append([view_list[files_name.index(filename)], neuron_id])

        # convert to numpy arrays
        T_trace_array = np.array(T_trace_list)
        S_center_array = np.array(S_center_list)
        id_array = np.array(id_list)
        view_array = np.array(view_list)

        # save all data
        print(f"{all_mat_path} saving...")
        savemat(all_mat_path, {
            'all_trace': T_trace_array,
            'S_center_list': S_center_array,
            'all_index': id_array,
            'all_view_num': view_array
        }, do_compression=True)
        print(f"{all_mat_path} saved!")

    # Step 2: Spatiotemporal view merging
    print("Spatiotemporal view merging...")
    R_path = os.path.join(SAVE, 'R_matrix.mat')
    R = calculate_coef_matrix(T_trace_array, id_array, R_path)

    neuron_group = spatio_temporal_clustering(
        R, id_array, S_center_array, cutoff_spatial, cutoff_temporal, min_view_num
    )
    view_merge_C, view_merge_id, all_single_neuron_trace = group_save(
        neuron_group, S_center_array, id_array, T_trace_array, cutoff_temporal, SAVE
    )

    # Step 3: 3D Localization
    print("Estimating Z...")
    psffit_data = loadmat_auto(psffit_matrix_file)
    psffit_matrix = psffit_data['psffit_matrix']

    spatial_3D, neuron_num, invalid_flag = f_estimateZ(
        view_merge_C, view_merge_id, psffit_matrix,
        min_loc_num, Nnum, upsample_rate, dz, cen_id
    )
    valid_single_neuron_trace = all_single_neuron_trace[~invalid_flag, :]

    # save the final results
    matpath = os.path.join(SAVE, "result.mat")
    print(f"Saving final result to {matpath}...")
    savemat(matpath, {
        'spatial_3D': spatial_3D,
        'all_single_neuron_trace': valid_single_neuron_trace
    })
    print('Done!')

    # return spatial_3D, valid_single_neuron_trace
