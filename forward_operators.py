import numpy as np

from scipy.sparse import csr_array, lil_array, eye, diags, bmat
from scipy.sparse import load_npz, save_npz
from skimage.transform import radon, iradon

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')


def construct_observation_matrix(n_dof, n_obs):
    stride = int(np.sqrt(n_dof / n_obs))
    used_n_obs = int(n_dof / stride ** 2)
    if used_n_obs != n_obs:
        raise Warning(f"Used {used_n_obs} observations instead of requested {n_obs}.")

    width = int(np.sqrt(n_dof))

    bool_rows = np.zeros((width, width), dtype=bool)
    bool_cols = np.zeros((width, width), dtype=bool)
    idx_1d = np.arange(0, width, stride, dtype=int) + int(stride / 2)
    bool_rows[idx_1d, :] = True
    bool_cols[:, idx_1d] = True
    bool_idx = bool_rows * bool_cols
    indices = np.arange(n_dof)[bool_idx.flatten()]

    A = lil_array((used_n_obs, n_dof))
    A[np.arange(used_n_obs), indices] = 1
    A = A.tocsr()

    return A


def construct_random_obs_matrix(n_dof, n_obs):
    lin_idx = np.random.choice(n_dof, n_obs, replace=False)
    
    A = lil_array((n_obs, n_dof))
    A[np.arange(n_obs), lin_idx] = 1
    A = A.tocsr()

    return A


def construct_observation_matrix_large(n_dof, n_obs, scale):
    width = int(np.sqrt(n_dof))
    n_dof_small = int((width / scale) ** 2)
    stride = int(np.sqrt(n_dof_small / n_obs))
    used_n_obs = int(n_dof_small / stride ** 2)
    if used_n_obs != n_obs:
        raise Warning(f"Used {used_n_obs} observations instead of requested {n_obs}.")

    width_small = int(np.sqrt(n_dof_small))
    w = int((width - width_small) / 2)
    bool_idx = np.ones((width, width), dtype=bool)
    idx_1d = np.arange(w, width - w, stride, dtype=int)
    bool_idx[idx_1d, :] = bool_idx[:, idx_1d] = False
    bool_idx[:w, :] = bool_idx[-w:, :] = bool_idx[:, :w] = bool_idx[:, -w:] = False
    indices = np.arange(n_dof)[bool_idx.flatten()]

    A = lil_array((used_n_obs, n_dof), dtype=int)
    A[np.arange(used_n_obs), indices] = 1
    A = A.tocsr()

    return A


def adjust_for_extra_dof(op_matrix, edge_width):
    width = int(np.sqrt(op_matrix.shape[1]))
    big_zero_mat = csr_array((op_matrix.shape[0], (width + 2 * edge_width) * edge_width))
    small_zero_mat = csr_array((op_matrix.shape[0], edge_width))
    
    mat_list = [big_zero_mat]
    for i in range(0, op_matrix.shape[1], width):
        mat_list.append(small_zero_mat)
        mat_list.append(op_matrix[:, i:i+width])
        mat_list.append(small_zero_mat)
    mat_list.append(big_zero_mat)

    mat = bmat([mat_list])

    return mat


def construct_edge_observation_matrix(n_dof, edge_width=1):
    width = int(np.sqrt(n_dof))

    bool_img = np.zeros((width, width), dtype=bool)
    bool_img[:, :edge_width] = bool_img[:, -edge_width:] = bool_img[:edge_width, :] = bool_img[-edge_width:, :] = True
    indices = np.arange(n_dof)[bool_img.flatten()]

    n_obs = bool_img.sum()
    A = lil_array((n_obs, n_dof))
    A[np.arange(n_obs), indices] = 1
    A = A.tocsr()

    return A


def construct_blurring_matrix(n_dof):
    width = int(np.sqrt(n_dof))

    single_block = eye(width)
    off_diag = np.ones(width - 1)
    single_block += diags(off_diag, -1) + diags(off_diag, 1)

    generate_blocks_list = lambda i: [single_block if np.abs(i - k) < 2 else None for k in range(width)]
    blocks = [generate_blocks_list(i) for i in range(width)]

    A = bmat(blocks).tocsr()
    A = A / 9

    return A


def construct_radon_transform_matrix(n_dof, angles):
    unit_vecs = np.eye(n_dof)

    output_dof = angles.size * int(np.sqrt(n_dof))
    dense_A = np.zeros((output_dof, n_dof))
    width = int(np.sqrt(n_dof))

    for i in range(n_dof):
        unit_img = np.reshape(unit_vecs[:, i], (width, width))
        dense_A[:, i] = radon(unit_img, angles).flatten()

    A = csr_array(dense_A)

    return A


def get_radon_transform_matrix(n_dof, angles):
    width = int(np.sqrt(n_dof))
    if width != np.sqrt(n_dof):
        raise Exception("Number of degrees of freedom (n_dof) must be a square number.")

    filename = f"rt_matrix_{width}x{angles.size}.npz"
    try:
        A = load_npz(filename)
    except OSError:
        print("Computing RT matrix...")
        A = construct_radon_transform_matrix(n_dof, angles)
        save_npz(filename, A)
        print("Saved RT matrix to: " + filename)
    
    return A


def construct_fbp_matrix(n_dof, angles):
    input_dof = angles.size * int(np.sqrt(n_dof))
    unit_vecs = np.eye(input_dof)

    dense_A = np.zeros((n_dof, input_dof))
    width = int(np.sqrt(n_dof))

    for i in range(input_dof):
        unit_img = np.reshape(unit_vecs[:, i], (width, angles.size))
        dense_A[:, i] = iradon(unit_img, theta=angles, output_size=width).flatten()

    A = csr_array(dense_A)

    return A


def get_fbp_matrix(n_dof, angles):
    width = int(np.sqrt(n_dof))
    if width != np.sqrt(n_dof):
        raise Exception("Number of degrees of freedom (n_dof) must be a square number.")

    filename = f"MCMC_2D/fbp_matrix_{width}x{angles.size}.npz"
    try:
        A = load_npz(filename)
    except OSError:
        print("Computing FBP matrix...")
        A = construct_fbp_matrix(n_dof, angles)
        save_npz(filename, A)
        print("Saved FBP matrix to: " + filename)
    
    return A


def apply_forward_operator(forward_op, true_img, noise_std, noise_mask=None):
    true_obs = forward_op @ true_img
    obs = true_obs
    if noise_mask is None:
        obs += noise_std * np.random.randn(true_obs.size)
    else:
        obs[noise_mask] += noise_std * np.random.randn(noise_mask.sum())

    return obs
