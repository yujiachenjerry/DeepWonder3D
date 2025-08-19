import os
import tifffile
import numpy as np
from natsort import natsorted
from scipy.io import savemat


def psf_weighted_centroids_array(tiff_path):
    """
    Read PSF tiff file and compute the weighted centroid (intensity-weighted)
    for each frame. Return as a numpy array.

    Args:
        tiff_path (str): Path to PSF tiff file

    Returns:
        centroids_array (np.ndarray): shape = (Nz, 2), each row is (y, x) float coordinates
    """
    psf_stack = tifffile.imread(tiff_path)
    centroids = []

    for frame in psf_stack:
        total_intensity = np.sum(frame)
        if total_intensity == 0:
            centroids.append([np.nan, np.nan])
            continue

        y_indices, x_indices = np.indices(frame.shape)
        y_center = np.sum(y_indices * frame) / total_intensity
        x_center = np.sum(x_indices * frame) / total_intensity
        centroids.append([y_center, x_center])

    # convert to numpy array
    centroids_array = np.array(centroids, dtype=np.float64)
    return centroids_array


def compute_psf_fit(psf, all_z):
    """
    Compute linear fitting parameters of centroid differences across views
    at different depths.

    Args:
        psf : numpy.ndarray, shape (N_views, N_z, 2)
            Centroid coordinates [x, y] for each view at each depth
        all_z : numpy.ndarray, shape (N_z,)
            Depth values

    Returns:
        psffit_matrix : numpy.ndarray, shape (N_views*(N_views-1), 6)
            Each row is [i, j, ax, bx, ay, by]
    """
    N_views = psf.shape[0]
    psffit_matrix = []

    for i in range(N_views):
        for j in range(N_views):
            if i == j:
                continue

            # vectorized computation of dx, dy
            dx = psf[i, :, 0] - psf[j, :, 0]
            dy = psf[i, :, 1] - psf[j, :, 1]

            # linear fitting
            ax, bx = np.polyfit(all_z, dx, 1)
            ay, by = np.polyfit(all_z, dy, 1)

            psffit_matrix.append([i, j, ax, bx, ay, by])

    return np.array(psffit_matrix)


# PARAMETERS
PSF_folder = 'your_PSF_folder_path'
all_z = (np.arange(101) - 50) * 3  ##### e.g. 101 z-slices, from -150 um to 150 um, z-step = 3 um

# Example usage
centroids_array = []

for tiff_file in natsorted(os.listdir(PSF_folder)):
    if tiff_file.endswith('.tif') or tiff_file.endswith('.tiff'):
        print('Processing file: ', tiff_file)
        centroids = psf_weighted_centroids_array(os.path.join(PSF_folder, tiff_file))
        centroids_array.append(centroids)

centroids_array = np.array(centroids_array)
psffit_matrix = compute_psf_fit(centroids_array, all_z)
savemat('psffit_matrix.mat', {'psffit_matrix': psffit_matrix})
