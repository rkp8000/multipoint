"""
Function for generating multi-point perspective projection
of three-dimensional wireframe.

rich pang
june 3rd, 2018
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_wires_3d(wires, p_e=None, v_e=None, fig_size=(10, 10), azim=-60, elev=30):
    """Plot 3D wire frame using built-in matplotlib functions."""
    fig = plt.figure(figsize=fig_size, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    for wire, kwargs in wires.values():
        ax.plot(*wire.T, **kwargs)
        
    if p_e is not None:
        ax.scatter(*p_e, s=100, c='k')
        
    if v_e is not None:
        x = [p_e[0], p_e[0] + v_e[0]]
        y = [p_e[1], p_e[1] + v_e[1]]
        z = [p_e[2], p_e[2] + v_e[2]]
        
        ax.plot(x, y, z, c='k', lw=3)
        
    ax.azim = azim
    ax.elev = elev
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    return fig, ax


def to_mpp(wires, p_e, v_e, phi, a, b, width=10):
    
    assert p_e.shape == (3,)
    assert v_e.shape == (3,)
    assert -180 <= phi <= 180
    assert -180 <= a <= 180
    assert -180 <= b <= 180
    
    phi = phi * np.pi / 180
    a = a * np.pi / 180
    b = b * np.pi / 180
    
    # load wires
    if isinstance(wires, str):
        with open(wires, 'rb') as f:
            # extract wires from json file
            raise NotImplementedError
    
    # transform each wire to a 2D path
    paths_2d = {}

    for key, (wire, kwargs) in wires.items():
        paths_2d[key] = (transform(
            w=wire, p_e=p_e, v_e=v_e, phi=phi, a=a, b=b), kwargs)
        
    # draw paths
    fig_x = width
    fig_y = width * np.tan(b/2) / np.tan(a/2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_x, fig_y))
    
    for key, (path_2d, kwargs) in paths_2d.items():
        ax.plot(path_2d[:, 0], path_2d[:, 1], **kwargs)
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-np.tan(b/2) / np.tan(a/2), np.tan(b/2) / np.tan(a/2))
    
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    return fig, ax


def transform(w, p_e, v_e, phi, a, b):
    """Transform a 3D path (wire) into a 2D path in perspective.
    
    The basic algorithm is as follows:
        1. shift wire to be centered at p_e
        2. project wire onto new basis aligned with v_e
        3. project wire in new basis onto 2D view plane
        4. rotate view plane by phi
        5. set x & y limits of view plane by a & b, respectively
    
    :param w: N x 3 numpy array describing 3D path through space
    :param p_e: eye position
    :param v_e: vector representation of view direction
        (note: only angle matters, magnitude is ignored)
    :param phi: view rotation (in deg, between -180 and 180)
    :param a: horizontal extent of view plane (in deg, max 180)
    :param b: vertical extent of view plane (in deg, max 180)
    
    :return: 2D path projected onto perspective plane
    """
    # check params
    assert w.shape[1] == 3
    
    # recenter w
    w = w - p_e
    
    # project w onto basis aligned with v_e
    w = w.dot(get_basis(v_e))
    
    # get dist to view plane of width 1
    delta = 1 / np.tan(a/2)
    
    # get projections onto view plane
    x_ = -delta * w[:, 1] / w[:, 0]
    y_ = delta * w[:, 2] / w[:, 0]
    p_ = np.array([x_, y_])
    
    # rotate view plane by phi
    r = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    p = r.dot(p_)
    
    # ignore points behind view plane
    p[:, w[:, 0] < 0] = np.nan
    
    return p.T


def get_basis(v_e):
    """Get new basis with b_1 aligned with v_e and b_2 parallel to the xy-axis."""
    assert v_e.shape == (3,)
    assert np.linalg.norm(v_e) > 0
    
    # align first basis vector with v_e
    b_1 = v_e / np.linalg.norm(v_e)
    
    # create second basis vector orth. to b_1 parallel to xy-axis
    if b_1[0] == b_1[1] == 0:
        
        b_2 = np.array([1, 0, 0]) * np.sign(b_1[2])
        
    else:
        
        z_hat = np.array([0, 0, 1])
        tmp = np.cross(z_hat, b_1)
        b_2 = tmp / np.linalg.norm(tmp)
    
    # create third basis vector orth. to b_1 & b_2 via right hand rule
    b_3 = np.cross(b_1, b_2)
    
    # return matrix with cols b_1, b_2, b_3
    return np.array([b_1, b_2, b_3]).T
