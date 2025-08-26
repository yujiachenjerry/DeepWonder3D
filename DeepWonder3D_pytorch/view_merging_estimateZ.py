import os
import numpy as np
from deepwonder.VM.utils import *

# ===========================
# Step 0: Setting Parameters
# ===========================

# PSF parameters
upsample_rate = 2
dz = 1.5
Nnum = 15
min_view_num = 5
min_loc_num = 3
cen_id = 113

# clustering parameters
cutoff_spatial = 20  # Euclidean distance
cutoff_temporal = 0.75  # correlation coefficient

# input/output paths
DATA = './result/STEP_6_MN/mat_0'
SAVE = './STEP_7_VM'
psffit_matrix_file = './pth/psffit_matrix.mat'

# ===========================
# Step 1: Load all data
# ===========================
print(f"Loading data from {DATA}...")
os.makedirs(SAVE, exist_ok=True)
all_mat_path = os.path.join(SAVE, f'all_data.mat')

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

    # storage containers (use list first)
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
            # Temporal_trace = zscore(Temporal_trace, axis=1)
            Spatial_center = final_mask_list[0, neuron_id]['centroid'][0][0]

            T_trace_list.append(np.squeeze(Temporal_trace))
            S_center_list.append(np.squeeze(Spatial_center))
            id_list.append([view_list[files_name.index(filename)], neuron_id])

    # convert to numpy arrays
    T_trace_array = np.array(T_trace_list)
    S_center_array = np.array(S_center_list)
    id_array = np.array(id_list)
    view_array = np.array(view_list)

    # save as all_data_xxx.mat (optional)
    print(f"{all_mat_path} saving...")
    savemat(all_mat_path, {
        'all_trace': T_trace_array,
        'S_center_list': S_center_array,
        'all_index': id_array,
        'all_view_num': view_array
    }, do_compression=True)
    print(f"{all_mat_path} saved!")


# ===========================
# Step 2: Spatiotemporal view merging
# ===========================
print(f"spatiotemporal view merging...")
R_path = os.path.join(SAVE, f'R_matrix.mat')
R = calculate_coef_matrix(T_trace_array, id_array, R_path)

neuron_group = spatio_temporal_clustering(R, id_array, S_center_array, cutoff_spatial, cutoff_temporal, min_view_num)
view_merge_C, view_merge_id, all_single_neuron_trace = group_save(neuron_group, S_center_array, id_array, T_trace_array,
                                                                  cutoff_temporal, SAVE)


# ===========================
# Step 3: 3D Localization
# ===========================
print(f"estimating Z...")
psffit_data = loadmat_auto(psffit_matrix_file)
psffit_matrix = psffit_data['psffit_matrix']

spatial_3D, neuron_num, invalid_flag = f_estimateZ(view_merge_C, view_merge_id, psffit_matrix,
                                                   min_loc_num, Nnum, upsample_rate, dz, cen_id)
valid_single_neuron_trace = all_single_neuron_trace[~invalid_flag, :]

# save the final results in a `.mat` file
matpath = os.path.join(SAVE, f"result.mat")
print(f"Saving final result to {matpath}...")
savemat(
    matpath,
    {'spatial_3D': spatial_3D, 'all_single_neuron_trace': valid_single_neuron_trace}
)
print('Done!')
