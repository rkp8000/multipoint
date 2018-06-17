"""
Functions for creating wireframes of simple shapes.
"""
import numpy as np


def cyl(
        x, y, z, a, b, h, n_spokes, density=500,
        normal=None, phi=None, kwargs=None):
    """Elliptical cylinder."""
    
    if normal is not None:
        raise NotImplementedError
    if phi is not None:
        raise NotImplementedError
    if kwargs is None:
        kwargs = {}
    
    ws = {}
    
    # top & bottom
    th = np.linspace(0, 2*np.pi, density, endpoint=False)
    xs = x + a*np.cos(th)
    ys = y + b*np.sin(th)
    zs_top = np.repeat(z + h/2, density)
    zs_bot = np.repeat(z - h/2, density)
    
    ws['top'] = [np.array([xs, ys, zs_top]).T, kwargs]
    ws['bottom'] = [np.array([xs, ys, zs_bot]).T, kwargs]
    
    # spokes
    zs = np.linspace(z - h/2, z + h/2, density)
    
    for th in np.linspace(0, 2*np.pi, n_spokes, endpoint=False):
        xs = np.repeat(x + a*np.cos(th), density)
        ys = np.repeat(y + b*np.sin(th), density)
        
        k = 'spk_{0:.2f}'.format(th * 180 / np.pi)
        ws[k] = [np.array([xs, ys, zs]).T, kwargs]
        
    return ws


def box(
        x, y, z, l, w, h, normal=None, phi=0, kwargs=None):
    """Box."""
    
    if normal is not None:
        raise NotImplementedError
    if kwargs is None:
        kwargs = {}
        
    phi = phi * np.pi / 180
        
    ws = {}
    
    # top
    xs = np.array([x + l/2, x - l/2, x - l/2, x + l/2, x + l/2])
    ys = np.array([y + w/2, y + w/2, y - w/2, y - w/2, y + w/2])
    zs = np.array([z + h/2, z + h/2, z + h/2, z + h/2, z + h/2])
    
    ws['top'] = [np.array([xs, ys, zs]).T, kwargs]
    
    # bottom
    xs = np.array([x + l/2, x - l/2, x - l/2, x + l/2, x + l/2])
    ys = np.array([y + w/2, y + w/2, y - w/2, y - w/2, y + w/2])
    zs = np.array([z - h/2, z - h/2, z - h/2, z - h/2, z - h/2])
    
    ws['bottom'] = [np.array([xs, ys, zs]).T, kwargs]
    
    # sides
    ## ne
    xs = np.array([x + l/2, x + l/2])
    ys = np.array([y + w/2, y + w/2])
    zs = np.array([z - h/2, z + h/2])
    
    ws['side_ne'] = [np.array([xs, ys, zs]).T, kwargs]
    
    ## nw
    xs = np.array([x - l/2, x - l/2])
    ys = np.array([y + w/2, y + w/2])
    zs = np.array([z - h/2, z + h/2])
        
    ws['side_nw'] = [np.array([xs, ys, zs]).T, kwargs]
    
    ## sw
    xs = np.array([x - l/2, x - l/2])
    ys = np.array([y - w/2, y - w/2])
    zs = np.array([z - h/2, z + h/2])
        
    ws['side_sw'] = [np.array([xs, ys, zs]).T, kwargs]
    
    ## se
    xs = np.array([x + l/2, x + l/2])
    ys = np.array([y - w/2, y - w/2])
    zs = np.array([z - h/2, z + h/2])
        
    ws['side_se'] = [np.array([xs, ys, zs]).T, kwargs]
    
    # rotation
    rot_mat = np.array([
        [np.cos(phi), np.sin(phi)],
        [-np.sin(phi), np.cos(phi)],
    ])
    
    for k, (w, kw) in ws.items():
        w_rotated = w.copy()
        
        # shift box to origin
        w_rotated[:, 0] -= x
        w_rotated[:, 1] -= y
        
        # rotate box
        w_rotated[:, :2] = w_rotated[:, :2].dot(rot_mat.T)
        
        # shift box away from origin
        w_rotated[:, 0] += x
        w_rotated[:, 1] += y
        
        ws[k] = [w_rotated, kw]
        
    return ws
