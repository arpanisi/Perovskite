from os.path import basename, join
from time import time
import pickle
# from math import ceil, floor
import numpy as np
import scipy as sp
from sklearn import manifold
from sklearn.metrics import pairwise
import matplotlib as mpl

from vispy import scene, app
from vispy.scene import visuals

#mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # Line3DCollection

plt.style.use('seaborn-poster')


# mpl.rcParams.update({'axes.titlesize': 28})
# mpl.rcParams.update({'axes.labelsize': 24})
# mpl.rcParams.update({'xtick.labelsize': 22})
# mpl.rcParams.update({'ytick.labelsize': 22})


def find_tag(abs_path):
    return basename(abs_path).strip().split(".")[0]


def str_to_num_list(str_list):
    str_list[::] = list(map(str.strip, str_list))
    # sum([item.is_integer() for item in str_list[0:3]]) == len(str_list[0:3]):
    # sum([item.is_integer() for item in str_list[0]]) == len(str_list[0]):

    if '.' in str_list[0]:
        try:
            str_list[::] = list(map(float, str_list))
        except ValueError:
            str_list[::] = [list(map(float, (str_.split()))) for str_ in str_list]
    else:
        try:
            str_list[::] = list(map(int, str_list))
        except ValueError:
            str_list[::] = [list(map(int, (str_.split()))) for str_ in str_list]


def keys_vals(str_list, f, str_list_size, output_type=None):
    for _ in range(str_list_size):
        str_list.append(next(f))
    if output_type == 'str':
        return str_list
    if next(f).split()[0] != "end":
        raise ValueError("The number of vertice in the file did not\
        match the atcual number vertice in the file: di/de values.")
    else:
        pass
    str_to_num_list(str_list)


def fingerprint(d_i, d_e, bin_num):
    x = np.asarray(d_i, dtype='float64')
    y = np.asarray(d_e, dtype='float64')
    edg_min, edg_max = 0, 4
    # edg_min, edg_max = 0, 3
    bin_arr = np.linspace(edg_min, edg_max, bin_num)
    hist, xedges, yedges = np.histogram2d(x=x, y=y, bins=(bin_arr, bin_arr), normed=True)
    return hist.T


def plot_signature1(d_i, d_e, plot_title, bin_num, file_path):
    hist = fingerprint(d_i, d_e, bin_num)
    edg_min, edg_max = 0, 4
    # edg_min, edg_max = 0, 3
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(plot_title)
    ax.set_xlabel('$d_i$')
    ax.set_ylabel('$d_e$')
    plt.xlim(edg_min, edg_max)
    plt.ylim(edg_min, edg_max)
    plt.xticks([0, 1, 2, 3, 4])
    plt.yticks([0, 1, 2, 3, 4])
    # plt.xticks([0, 1, 2, 3])
    # plt.yticks([0, 1, 2, 3])
    plt.imshow(hist, plt.get_cmap('jet'), norm=LogNorm(), aspect='equal', interpolation='nearest',
               origin='lower', extent=[edg_min, edg_max, edg_min, edg_max])
    plt.gcf()
    plt.savefig(file_path, bbox_inches='tight', dpi=600)


def plot_signature2(d_i, d_e, plot_title, bin_num, file_path):
    x = np.asarray(d_i, dtype='float64')
    y = np.asarray(d_e, dtype='float64')
    edg_min, edg_max = 0, 4
    # edg_min, edg_max = 0, 3
    # prepare canvas
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(plot_title)
    ax.set_xlabel('$d_i$')
    ax.set_ylabel('$d_e$')
    ax.hist2d(x, y, bins=np.linspace(edg_min, edg_max, bin_num),
              cmap=plt.get_cmap('jet'), norm=LogNorm())
    plt.xlim(edg_min, edg_max)
    plt.ylim(edg_min, edg_max)
    plt.xticks([0, 1, 2, 3, 4])
    plt.yticks([0, 1, 2, 3, 4])
    # plt.xticks([0, 1, 2, 3])
    # plt.yticks([0, 1, 2, 3])
    plt.gcf()
    plt.savefig(file_path, bbox_inches='tight', dpi=600)


# Function: prepare colorScales for Hirshfeld Surface
def surface_color_scale(surface_property, surfaceIdx, color_scheme):  ## 'bwr_r'
    color_scale = []
    color_range = [np.average([surface_property[i] for i in idx]) for idx in surfaceIdx]
    color_map = mpl.cm.get_cmap(color_scheme)
    color_norm = mpl.colors.Normalize(min(color_range), max(color_range))
    color_scale += [color_map(color_norm(value)) for value in color_range]
    return color_scale


# Function: prepare 3D polygons for plotting Hirshfeld Surface
def polygons(vtx, idx):
    tuple_list = np.array(vtx)
    poly3d = [[tuple_list[idx[ix][iy]] for iy in range(len(idx[0]))] for ix in range(len(idx))]
    return poly3d


# Function: prepare 3D vectors for plotting Hirshfeld Surface
def vec3D(vtx):
    x = np.array(vtx).T[0]
    y = np.array(vtx).T[1]
    z = np.array(vtx).T[2]
    return {'x': x, 'y': y, 'z': z}


# plot Hirshfeld surface with specific surface property
def plot_iso_surf(verts, idx, surface_property, color_scheme, view):
    fig1 = plt.figure(figsize=(8, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(vec3D(verts)['x'], vec3D(verts)['y'], vec3D(verts)['z'], s=1)
    collection = Poly3DCollection(polygons(verts, idx), linewidths=.0, edgecolor='k', alpha=1)
    collection.set_facecolors(colors=surface_color_scale(surface_property, idx, color_scheme))
    ax1.add_collection3d(collection)
    ax1.view_init(view['x'], view['y'])
    plt.show()


def save_obj(obj, dir_out, fo):
    with open(join(dir_out, fo + '.pkl'), 'wb') as obj_out:
        pickle.dump(obj, obj_out, pickle.HIGHEST_PROTOCOL)


def map_correlation(x, i, j, step=1, is_save_obj=False, dir_out=None, tag=None):
    err = np.zeros(j // step, dtype=float)
    x_iso = manifold.Isomap(n_neighbors=i, n_components=j, neighbors_algorithm='kd_tree')
    y = x_iso.fit_transform(x)
    for n_comp in range(1 * step, j + 1, step):
        err[n_comp // step - 1] = 1 - sp.stats.pearsonr(x_iso.dist_matrix_.flatten(),
                                                        sp.spatial.distance.cdist(y[:, :n_comp],
                                                                                  y[:, :n_comp]).flatten())[0] ** 2
    if is_save_obj:
        if tag is None:
            save_obj(y, dir_out, 'mapped_fingerprints_' + str(i))
        else:
            save_obj(y, dir_out, 'mapped_fingerprints_' + str(i) + tag)
    return err


def pairwise_distances_(x):
    return pairwise.pairwise_distances(X=x, Y=None, metric='euclidean')


def plot_surf_vispy(vertx, idx, surface_property, color_scheme, text=False):
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()
    cm = plt.get_cmap('gist_rainbow')
    p = len(vertx)
    cmap = np.asarray([cm(1. * i / p) for i in range(p)])
    # p = scene.visuals.Markers(pos=vertx, size=1, face_color='b')  # Same color for all vertices
    p = scene.visuals.Markers(pos=vertx, size=10, face_color=cmap) # Color the vertices according to their numbering
    view.add(p)
    c = surface_color_scale(surface_property, idx, 'bwr_r') # Colormap according to crystalexplorer visualization
    # c = np.tile([1, 1, 1], (len(idx), 1)) # Colormap for same color applied to all faces
    mesh = scene.visuals.Mesh(vertices=vertx, faces=idx, face_colors=c)
    view.add(mesh)

    # t1 = scene.visuals.Text(str(1), pos=vertx[0], color='black')
    # t1.font_size = 12
    # view.add(t1)
    #
    # t1 = scene.visuals.Text(str(10+1), pos=vertx[10], color='black')
    # t1.font_size = 12
    # view.add(t1)
    if text is True:
        for k, v in enumerate(vertx):
            t1 = scene.visuals.Text(str(k + 1), pos=v, color='black')
            t1.font_size = 16
            view.add(t1)


    view.camera = 'turntable'
    axis = visuals.XYZAxis(parent=view.scene)



    import sys
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()

    return canvas


