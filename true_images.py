import numpy as np

from skimage.data import shepp_logan_phantom, camera
from skimage.transform import rescale


def shepp_logan(n_dof):
    large_img = shepp_logan_phantom()    # 400 x 400 image of the Shepp-Logan phantom
    scaling = np.sqrt(n_dof) / 400
    img = rescale(large_img, scale=scaling, mode='reflect', channel_axis=None).flatten()

    return img


def square_and_circle(n_dof):
    width = int(np.sqrt(n_dof))
    img = np.zeros((width, width))
    coords_x, coords_y = np.meshgrid(np.arange(width), np.arange(width))

    square_idx_x = np.logical_and(0.1 * width < coords_x, coords_x < 0.6 * width)
    square_idx_y = np.logical_and(0.1 * width < coords_y, coords_y < 0.6 * width)
    img[np.logical_and(square_idx_x, square_idx_y)] = 0.5    # draw square

    img[(coords_x - 0.6 * width) ** 2 + (coords_y - 0.6 * width) ** 2 < (0.25 * width) ** 2] = 1.0    # draw circle
    flattened_img = img.flatten()

    return flattened_img


def step_function(n_dof):
    n_side = int(np.sqrt(n_dof))
    domain_1d = np.linspace(0, 1, n_side)
    x, y = np.meshgrid(domain_1d, domain_1d)

    f = np.logical_and(x > -.1, x < 0.5) * np.logical_and(y > -.1, y < 0.5)    # upper left square
    f = f + 0.5 * ((x + y) > 1.5)    # lower right triangle
    f_values = f.flatten().astype(float)
    
    return f_values


def eval_test_function(f_handle, x, y, shift_x, shift_y, scale=1):
    shifted_x = x - shift_x
    shifted_y = y - shift_y

    f = f_handle(scale * shifted_x, scale * shifted_y)

    return f


def subdivide_test_function(f_handle, x, y):
    n = int(np.sqrt(x.shape[0]))
    f = np.zeros((n, n))

    mask = lambda low_x, up_x, low_y, up_y: np.logical_and(x >= low_x, x <= up_x) * np.logical_and(y >= low_y, y <= up_y)

    bottom_right_ttll = mask(0.5, 0.625, 0.5, 0.625)
    bottom_right_ttl  = mask(0.625, 0.75, 0.5, 0.625)
    bottom_right_ttr  = mask(0.75, 0.875, 0.5, 0.625)
    bottom_right_ttrr = mask(0.875, 1, 0.5, 0.625)
    bottom_right_tll  = mask(0.5, 0.625, 0.625, 0.75)
    bottom_right_tl   = mask(0.625, 0.75, 0.625, 0.75)
    bottom_right_tr   = mask(0.75, 0.875, 0.625, 0.75)
    bottom_right_trr  = mask(0.875, 1, 0.625, 0.75)
    bottom_right_bll  = mask(0.5, 0.625, 0.75, 0.875)
    bottom_right_bl   = mask(0.625, 0.75, 0.75, 0.875)
    bottom_right_br   = mask(0.75, 0.875, 0.75, 0.875)
    bottom_right_brr  = mask(0.875, 1, 0.75, 0.875)
    bottom_right_bbll = mask(0.5, 0.625, 0.875, 1)
    bottom_right_bbl  = mask(0.625, 0.75, 0.875, 1)
    bottom_right_bbr  = mask(0.75, 0.875, 0.875, 1)
    bottom_right_bbrr = mask(0.875, 1, 0.875, 1)
    f.ravel()[bottom_right_ttll] = eval_test_function(f_handle, x[bottom_right_ttll], y[bottom_right_ttll], 0.5, 0.5, scale=8)
    f.ravel()[bottom_right_ttl]  = - eval_test_function(f_handle, x[bottom_right_ttl], y[bottom_right_ttl], 0.625, 0.5, scale=8)
    f.ravel()[bottom_right_ttr]  = eval_test_function(f_handle, x[bottom_right_ttr], y[bottom_right_ttr], 0.75, 0.5, scale=8)
    f.ravel()[bottom_right_ttrr] = - eval_test_function(f_handle, x[bottom_right_ttrr], y[bottom_right_ttrr], 0.875, 0.5, scale=8)
    f.ravel()[bottom_right_tll]  = - eval_test_function(f_handle, x[bottom_right_tll], y[bottom_right_tll], 0.5, 0.625, scale=8)
    f.ravel()[bottom_right_tl]   = eval_test_function(f_handle, x[bottom_right_tl], y[bottom_right_tl], 0.625, 0.625, scale=8)
    f.ravel()[bottom_right_tr]   = - eval_test_function(f_handle, x[bottom_right_tr], y[bottom_right_tr], 0.75, 0.625, scale=8)
    f.ravel()[bottom_right_trr]  = eval_test_function(f_handle, x[bottom_right_trr], y[bottom_right_trr], 0.875, 0.625, scale=8)
    f.ravel()[bottom_right_bll]  = eval_test_function(f_handle, x[bottom_right_bll], y[bottom_right_bll], 0.5, 0.75, scale=8)
    f.ravel()[bottom_right_bl]   = - eval_test_function(f_handle, x[bottom_right_bl], y[bottom_right_bl], 0.625, 0.75, scale=8)
    f.ravel()[bottom_right_br]   = eval_test_function(f_handle, x[bottom_right_br], y[bottom_right_br], 0.75, 0.75, scale=8)
    f.ravel()[bottom_right_brr]  = - eval_test_function(f_handle, x[bottom_right_brr], y[bottom_right_brr], 0.875, 0.75, scale=8)
    f.ravel()[bottom_right_bbll] = - eval_test_function(f_handle, x[bottom_right_bbll], y[bottom_right_bbll], 0.5, 0.875, scale=8)
    f.ravel()[bottom_right_bbl]  = eval_test_function(f_handle, x[bottom_right_bbl], y[bottom_right_bbl], 0.625, 0.875, scale=8)
    f.ravel()[bottom_right_bbr]  = - eval_test_function(f_handle, x[bottom_right_bbr], y[bottom_right_bbr], 0.75, 0.875, scale=8)
    f.ravel()[bottom_right_bbrr] = eval_test_function(f_handle, x[bottom_right_bbrr], y[bottom_right_bbrr], 0.875, 0.875, scale=8)

    top_right_tl = mask(0.5, 0.75, 0, 0.25)
    top_right_tr = mask(0.75, 1, 0, 0.25)
    top_right_bl = mask(0.5, 0.75, 0.25, 0.5)
    top_right_br = mask(0.75, 1, 0.25, 0.5)
    f.ravel()[top_right_tl] = eval_test_function(f_handle, x[top_right_tl], y[top_right_tl], 0.5, 0, scale=4)
    f.ravel()[top_right_tr] = - eval_test_function(f_handle, x[top_right_tr], y[top_right_tr], 0.75, 0, scale=4)
    f.ravel()[top_right_bl] = - eval_test_function(f_handle, x[top_right_bl], y[top_right_bl], 0.5, 0.25, scale=4)
    f.ravel()[top_right_br] = eval_test_function(f_handle, x[top_right_br], y[top_right_br], 0.75, 0.25, scale=4)

    bottom_left_tl = mask(0, 0.166, 0.5, 0.666)
    bottom_left_t  = mask(0.166, 0.333, 0.5, 0.666)
    bottom_left_tr = mask(0.333, 0.5, 0.5, 0.666)
    bottom_left_l  = mask(0, 0.166, 0.666, 0.833)
    bottom_left_m  = mask(0.166, 0.333, 0.666, 0.833)
    bottom_left_r  = mask(0.333, 0.5, 0.666, 0.833)
    bottom_left_bl = mask(0, 0.166, 0.833, 1)
    bottom_left_b  = mask(0.166, 0.333, 0.833, 1)
    bottom_left_br = mask(0.333, 0.5, 0.833, 1)
    f.ravel()[bottom_left_tl] = - eval_test_function(f_handle, x[bottom_left_tl], y[bottom_left_tl], 0, 0.5, scale=6)
    f.ravel()[bottom_left_t] = eval_test_function(f_handle, x[bottom_left_t], y[bottom_left_t], 1/6, 0.5, scale=6)
    f.ravel()[bottom_left_tr] = - eval_test_function(f_handle, x[bottom_left_tr], y[bottom_left_tr], 1/3, 0.5, scale=6)
    f.ravel()[bottom_left_l] = eval_test_function(f_handle, x[bottom_left_l], y[bottom_left_l], 0, 2/3, scale=6)
    f.ravel()[bottom_left_m] = - eval_test_function(f_handle, x[bottom_left_m], y[bottom_left_m], 1/6, 2/3, scale=6)
    f.ravel()[bottom_left_r] = eval_test_function(f_handle, x[bottom_left_r], y[bottom_left_r], 1/3, 2/3, scale=6)
    f.ravel()[bottom_left_bl] = - eval_test_function(f_handle, x[bottom_left_bl], y[bottom_left_bl], 0, 5/6, scale=6)
    f.ravel()[bottom_left_b] = eval_test_function(f_handle, x[bottom_left_b], y[bottom_left_b], 1/6, 5/6, scale=6)
    f.ravel()[bottom_left_br] = - eval_test_function(f_handle, x[bottom_left_br], y[bottom_left_br], 1/3, 5/6, scale=6)
    
    top_left = mask(0, 0.5, 0, 0.5)
    f.ravel()[top_left] = eval_test_function(f_handle, x[top_left], y[top_left], 0, 0, scale=2)

    return f


def multi_test_function(n_dof):
    n_side = int(np.sqrt(n_dof))
    domain_1d = np.linspace(0, 1, n_side)
    x, y = np.meshgrid(domain_1d, domain_1d)

    mask = lambda low_x, up_x, low_y, up_y: np.logical_and(x >= low_x, x <= up_x) * np.logical_and(y >= low_y, y <= up_y)
    f = np.zeros((n_side, n_side))

    top_left = mask(0, 0.5, 0, 0.5)
    f[top_left] = subdivide_test_function(lambda _, __: 1, x[top_left] * 2, y[top_left] * 2).ravel()
    
    top_right = mask(0.5, 1, 0, 0.5)
    f[top_right] = subdivide_test_function(lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y), (x[top_right] - 0.5) * 2, y[top_right] * 2).ravel()

    bottom_left = mask(0, 0.5, 0.5, 1)
    f[bottom_left] = subdivide_test_function(lambda x, y: 4 * (0.5 - abs(x - 0.5)) * (0.5 - abs(y - 0.5)), x[bottom_left] * 2, (y[bottom_left] - 0.5) * 2).ravel()
    
    bottom_right = mask(0.5, 1, 0.5, 1)
    f[bottom_right] = eval_test_function(lambda x, y: (x * y) ** 2, x[bottom_right], y[bottom_right], 0.5, 0.5, scale=2).ravel()

    f_values = f.flatten()
    
    return f_values


def straight_edge(n_dof):
    n_edge = int(np.sqrt(n_dof))
    midpoint = int(n_edge / 2)

    f = np.ones((n_edge, n_edge))
    f[:, :midpoint] = -1
    f_values = f.flatten().astype(float)

    e = np.zeros((n_edge, n_edge))
    t = 3   # controls line thickness
    e[:, midpoint-t:midpoint+t] = 1
    edge_map = e.flatten()

    return f_values, edge_map


def small_zigzag_edge(n_dof):
    def small_zigzag_f(x):
        def zig(z, mid, shift):
            return np.abs(z - mid) + shift
        def mask(z, left, right):
            return np.logical_and(z > left, z < right)

        f = 0.5 + mask(x, 0.1, 0.2) * zig(x, 0.15, -0.05) - mask(x, 0.2, 0.3) * zig(x, 0.25, -0.05) \
                + mask(x, 0.3, 0.4) * zig(x, 0.35, -0.05) - mask(x, 0.4, 0.5) * zig(x, 0.45, -0.05) \
                + mask(x, 0.5, 0.6) * zig(x, 0.55, -0.05) - mask(x, 0.6, 0.7) * zig(x, 0.65, -0.05) \
                + mask(x, 0.7, 0.8) * zig(x, 0.75, -0.05) - mask(x, 0.8, 0.9) * zig(x, 0.85, -0.05)

        return f

    n_edge = int(np.sqrt(n_dof))
    domain_1d = np.linspace(0, 1, n_edge)
    x, y = np.meshgrid(domain_1d, domain_1d)

    f = x > small_zigzag_f(y)
    f = 2 * f - 1
    f_values = f.flatten().astype(float)

    e = np.zeros((n_edge, n_edge))
    t = 3   # controls line thickness
    middle_idx = (small_zigzag_f(domain_1d) * n_edge).astype(int)
    straight_idx = np.concatenate((np.arange(0.1 * n_edge, dtype=int), np.arange(int(0.9 * n_edge), n_edge, dtype=int)))
    for shift in range(-t, t):
        e[straight_idx, middle_idx[straight_idx]+shift] = 1
    angled_idx = np.arange(int(0.1 * n_edge) + 1, 0.9 * n_edge - 1, dtype=int)
    angled_t = int(np.sqrt(2) * t)
    for shift in range(-angled_t, angled_t):
        e[angled_idx, middle_idx[angled_idx]+shift] = 1
    edge_map = e.flatten()
    
    return f_values, edge_map
