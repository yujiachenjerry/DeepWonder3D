import h5py
from scipy.io import loadmat, savemat
import os
import numpy as np
from natsort import natsorted
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.optimize import curve_fit


def is_in(array, element):
    return element in array


def convert_data_structure(data_list):
    """Wrap Python list into MATLAB-compatible structure"""
    N = len(data_list)
    new_data = np.empty((N, 1), dtype=object)
    for i in range(N):
        sublist = data_list[i]
        subarray = np.array(sublist)
        wrapped_array = np.array(['123', subarray], dtype=object)
        wrapped_array = wrapped_array[1:]
        new_data[i] = wrapped_array
    return new_data


def loadmat_auto(mat_path):
    """
    Automatically detect MAT file version and load all variables
    (excluding __header__, __version__, __globals__)

    Args:
        mat_path (str): Path to MAT file

    Returns:
        dict: {variable_name: data}
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    # Detect if MAT file is v7.3 (HDF5 format)
    with open(mat_path, 'rb') as f:
        header = f.read(128)
        is_v73 = b'MATLAB 7.3' in header

    variables = {}
    if is_v73:
        # v7.3 use h5py
        with h5py.File(mat_path, 'r') as file:
            for var in file.keys():
                if var.startswith("__"):
                    continue  # Skip MATLAB metadata
                data = file[var][:]
                ndim = data.ndim
                if ndim > 1:
                    data = np.transpose(data, tuple(range(ndim - 1, -1, -1)))
                variables[var] = data
    else:
        # non-v7.3 use scipy.io.loadmat
        mat_data = loadmat(mat_path)
        for var, data in mat_data.items():
            if var.startswith("__"):
                continue  # Skip metadata
            variables[var] = data

    return variables


def calculate_coef_matrix(all_trace, all_index, R_path=''):
    """
    Calculate correlation coefficient matrix and set correlations from the same view to 0.
    If R_matrix.mat already exists, load it directly.

    Args:
        all_trace (ndarray): shape (n_cells, n_frames), cell time series data
        all_index (ndarray): shape (n_cells, 2) or more columns, first column is view ID
        R_path (str): Path to save/load R matrix

    Returns:
        R (ndarray): correlation coefficient matrix
    """
    if os.path.exists(R_path):
        data = loadmat_auto(R_path)
        R = data['R']
    else:
        # Compute correlation coefficient matrix
        R = np.corrcoef(all_trace)

        # Get unique view IDs
        unique_views = np.unique(all_index[:, 0])

        # Initialize mask
        mask = np.zeros_like(R, dtype=bool)

        # Create mask for same view
        for view in unique_views:
            idx = (all_index[:, 0] == view)
            mask[np.ix_(idx, idx)] = True

        # Remove diagonal
        mask &= ~np.eye(mask.shape[0], dtype=bool)

        # Set correlations from same view to 0
        R[mask] = 0

        # Save
        if R_path:
            os.makedirs(os.path.dirname(R_path), exist_ok=True)
            savemat(R_path, {'R': R}, do_compression=True)
    return R


def spatial_cluster(S_center_list, cutoff_spatial, if_show=False):
    """
    Spatial clustering to exclude wrong components

    Args:
        S_center_list: np.array, shape (N, 2), neuron spatial centroid coordinates
        cutoff_spatial: float, spatial clustering distance threshold
        if_show: bool, whether to visualize results (optional)

    Returns:
        T_spatial: np.array, spatial cluster label for each neuron (1-based)
        N_spatial_num: int, number of spatial clusters
    """
    # Compute Euclidean distance matrix
    Y_spatial = pdist(S_center_list, metric='euclidean')

    # Hierarchical clustering
    Z_spatial = linkage(Y_spatial, method='single')

    # Generate clusters based on distance threshold
    T_spatial = fcluster(Z_spatial, t=cutoff_spatial, criterion='distance')
    N_spatial_num = np.max(T_spatial)

    # Visualization (optional)
    if if_show:
        import matplotlib.pyplot as plt
        plt.scatter(S_center_list[:, 1], S_center_list[:, 0], c=T_spatial, cmap='tab20')
        plt.title(f'Spatial Clustering (cutoff={cutoff_spatial})')
        plt.axis('equal')
        plt.show()
        plt.savefig('spatial_clustering.png')

    return T_spatial, N_spatial_num


def spatio_temporal_clustering(R, all_index, S_center_list,
                               cutoff_spatial, corr_thre, min_view_num,
                               if_show_spatial_clusters=False):
    """
    Spatio-temporal clustering based on correlation and spatial clustering
    """
    # === Binarize correlation matrix ===
    R_bina = (R > corr_thre).astype(int)

    print('Start spatio-temporal clustering...')
    neuron_group = []
    neuron_num = 0

    for i in range(R_bina.shape[0]):
        R_bina_r = R_bina[i, :]
        right_idx = np.where(R_bina_r == 1)[0]

        # === Spatial clustering ===
        if len(right_idx) >= min_view_num:
            right_S_center_list = S_center_list[right_idx, :]
            T_spatial, N_spatial_num = spatial_cluster(
                right_S_center_list, cutoff_spatial, if_show_spatial_clusters
            )

            if N_spatial_num > 1:
                most_T_spatial_index = np.bincount(T_spatial).argmax()  # mode
                most_index = np.where(T_spatial == most_T_spatial_index)[0]
                right_idx = right_idx[most_index]

        # === Select neuron with max correlation in each view ===
        if len(right_idx) >= min_view_num:
            current_corr_vector = R[i, right_idx]  # correlation with neuron i
            views_in_right_p = all_index[right_idx, 0]  # view IDs

            unique_views = np.unique(views_in_right_p)
            selected_neurons = []

            for view in unique_views:
                mask_in_view = (views_in_right_p == view)
                candidates_in_view = right_idx[mask_in_view]
                max_idx = np.argmax(current_corr_vector[mask_in_view])
                selected_neurons.append(candidates_in_view[max_idx])

            right_idx = np.array(selected_neurons, dtype=int)

        # === Check min view count ===
        if len(right_idx) >= min_view_num:
            # Clear selected neurons from correlation matrix to avoid duplicates
            R_bina[right_idx, :] = 0
            R_bina[:, right_idx] = 0

            neuron_num += 1
            neuron_group.append(right_idx)

    print(f'Spatio-temporal clustering done. Found {len(neuron_group)} neurons.')
    return neuron_group


def group_save(neuron_group, S_center_list, all_index, all_trace, corr_thre, SAVE):
    """
    Save neuron grouping results to .mat file

    Args:
        neuron_group (list of list[int]): neuron index list per group
        S_center_list (np.ndarray): neuron centroid coordinates
        all_index (np.ndarray): index info for all neurons
        all_trace (np.ndarray): signal traces for all neurons
        SAVE (str): output folder path
        corr_thre (float): correlation threshold
    """

    def merge_signals(signals):
        """Merge signals by averaging"""
        return np.mean(signals, axis=0)

    view_merge_C = []
    view_merge_id = []
    all_single_neuron_trace = []

    for group in neuron_group:
        single_neuron_index = np.array(group)
        view_merge_C.append(S_center_list[single_neuron_index, :])
        view_merge_id.append(all_index[single_neuron_index, :])

        neuron_group_trace = all_trace[single_neuron_index, :]
        all_single_neuron_trace.append(merge_signals(neuron_group_trace))

    all_single_neuron_trace = np.array(all_single_neuron_trace)

    # Save
    matpath = os.path.join(SAVE, 'view_merging.mat')
    print(f'Saving view merging results to {matpath}...')
    savemat(
        matpath,
        {
            'view_merge_C': convert_data_structure(view_merge_C),
            'view_merge_id': convert_data_structure(view_merge_id),
            'all_single_neuron_trace': all_single_neuron_trace,
            'corr_thre': corr_thre
        }
    )
    print('Done.')

    return view_merge_C, view_merge_id, all_single_neuron_trace


def f_estimateZ(view_merge_C, view_merge_id, psffit_matrix,
                min_loc_num, Nnum, upsample_rate, dz, cen_id):
    """
    Estimate 3D neuron positions (lateral + axial) based on multi-view centroid locations

    Returns:
        spatial_3D : np.ndarray
            [x, y, z] coordinates for each neuron
        neuron_num : int
            number of successfully reconstructed neurons
        invalid_flag : list
            boolean array indicating neurons with fewer than min_loc_num locations
    """

    def func(alpha_uv, z):
        return z * alpha_uv

    neuron_num = 0
    spatial_3D = np.zeros((len(view_merge_C), 3))
    invalid_flag = []

    for i, view_merge_C_uni in enumerate(view_merge_C):
        view_id = view_merge_id[i][:, 0]
        view_id_uni, uind = np.unique(view_id, return_index=True)
        view_merge_C_uni = view_merge_C_uni[uind, :]

        if view_merge_C_uni.shape[0] < min_loc_num:
            invalid_flag.append(1)
            continue  # skip this neuron
        else:
            invalid_flag.append(0)

        neuron_num += 1

        diffpos = np.zeros((view_merge_C_uni.shape[0] * (view_merge_C_uni.shape[0] - 1) // 2, 2))
        diffpar = np.zeros((view_merge_C_uni.shape[0] * (view_merge_C_uni.shape[0] - 1) // 2, 2))
        count = 0

        for j in range(view_merge_C_uni.shape[0] - 1):
            for k in range(j + 1, view_merge_C_uni.shape[0]):
                count += 1
                diffpos[count - 1] = view_merge_C_uni[j, :] - view_merge_C_uni[k, :]
                ind = np.logical_and(psffit_matrix[:, 0] == view_id_uni[j],
                                     psffit_matrix[:, 1] == view_id_uni[k])
                diffpar[count - 1] = np.squeeze(np.array([psffit_matrix[ind, 2],
                                                          psffit_matrix[ind, 4]]))

        z_pos, _ = curve_fit(func, diffpar.ravel(), diffpos.ravel(),
                             method='trf', ftol=1e-6)
        z_pos = z_pos * Nnum / upsample_rate * dz
        shiftpar = np.zeros((view_merge_C_uni.shape[0], 2))

        for j in range(view_merge_C_uni.shape[0]):
            if view_id_uni[j] < cen_id:
                ind = np.logical_and(psffit_matrix[:, 0] == view_id_uni[j],
                                     psffit_matrix[:, 1] == cen_id)
                shiftpar[j, :] = np.squeeze(np.array([-psffit_matrix[ind, 2],
                                                      -psffit_matrix[ind, 4]]))
            elif view_id_uni[j] > cen_id:
                ind = np.logical_and(psffit_matrix[:, 0] == cen_id,
                                     psffit_matrix[:, 1] == view_id_uni[j])
                shiftpar[j, :] = np.squeeze(np.array([psffit_matrix[ind, 2],
                                                      psffit_matrix[ind, 4]]))
            else:
                shiftpar[j, :] = np.array([0, 0])

        lateral_pos_list = view_merge_C_uni + shiftpar * z_pos / Nnum * upsample_rate
        lateral_pos = np.mean(lateral_pos_list, axis=0)

        spatial_3D[neuron_num - 1, :] = np.concatenate([lateral_pos, z_pos])

        if neuron_num % 20 == 0:
            print(f"{neuron_num} neuron done...")

    invalid_flag = np.array(invalid_flag, dtype=bool)

    return spatial_3D, neuron_num, invalid_flag
