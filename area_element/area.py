from pymatgen.io.vasp import Poscar
import numpy as np
import matplotlib.pyplot as plt
# from triangulation import noncenter_permu
from scipy.spatial import Delaunay

'''
Utility and foundation functions
'''
def read_structure(fname:str):
    abc, xyz, atom = [], [], []
    poscar = Poscar.from_file(fname)
    lattice = poscar.as_dict()['structure']['lattice']['matrix']
    for site in poscar.as_dict()['structure']['sites']:
        abc.append(site['abc'])
        xyz.append(site['xyz'])
        atom.append(site['label'])
    return np.array(lattice), np.array(abc), np.array(xyz), atom

# This method neglects z-direction
# It only adds images at the nearest quadrants of the center atom (See the logic below)
def minimum_pbc_2d(abc:np.ndarray, atom:list, center:np.ndarray):
    # Along x-direction, add right image if center is at the right side, vice versa
    a_shift = 1. if center[0] > 0.5 else -1.
    img_a = abc + np.array([a_shift, 0., 0.])
    # Along y-direction, add top image if center is at the top side, vice versa
    b_shift = 1. if center[1] > 0.5 else -1.
    img_b = abc + np.array([0., b_shift, 0.])
    # Along xy-direction (corners), add extra top-right image if top and right image are added, vice versa
    img_ab = abc + np.array([a_shift, b_shift, 0.])
    # return 4 times the original atom list because 3 images are added
    return np.concatenate([abc, img_a, img_b, img_ab], axis=0), atom * 4

# This method finds first and second nearest neighbours (of different kind of atoms than center atom) in fractional coords
def find_nn(abc:np.ndarray, atom:list, center:np.ndarray, center_atom:str, first_nn=3, second_nn=3):
    atom = np.array(atom)
    # Shift the center atom to origin, for distance calculation
    shifted_abc = abc - center
    dist = np.dot(shifted_abc, shifted_abc.T).diagonal()
    # Basically filter and get the coords and dist of Nitrogen atoms, dropping Boron atoms (that is, it only extracts other kinds of atoms (N) than the center atom (B) for this case)
    abc_filt = abc[atom != center_atom]
    dist_filt = dist[atom != center_atom]
    # Sort all Nitrogen atoms (in frac coords) by their distance compared to center
    idx_sort = np.argsort(dist_filt)
    abc_filt = abc_filt[idx_sort, :]
    # Extract n atoms with shortest distance as first NN, followed by m atoms as second NN
    first_nn_abc = abc_filt[:first_nn]
    second_nn_abc = abc_filt[first_nn:(first_nn+second_nn)]
    return first_nn_abc, second_nn_abc

# This method is for obtaining the 7 points of the area element, includes computing the non-atom vertices by the triangle-averaging method (using second NN and first NN)
def compute_area_vertices(first_nn_abc:np.ndarray, second_nn_abc:np.ndarray, center:np.ndarray):
    # Area element vertices, the center and first NN atoms must be the vertices
    vertices = np.array(first_nn_abc)
    # Find the non-atom sites for vertices
    for site in second_nn_abc:
        # For each atom of second NN, shift it as origin and calculate the dist of first NNs
        shifted_abc = first_nn_abc - site
        dist = np.dot(shifted_abc, shifted_abc.T).diagonal()
        # The two closer first NN points to the second NN atom site forms the triangle (for averaging)
        close_points = first_nn_abc[dist != dist.max()]
        # The average of the triangle vertices gives the non-atom site vertex for area element
        non_atom_vert = np.append([site], close_points, axis=0).mean(axis=0)
        vertices = np.append(vertices, [non_atom_vert], axis=0)
    # Sort all six vertices by angles (such that they are in cyclic order) 
    # Basically get phi = arctan(y / x) with quadrants
    shifted_vert = vertices - center
    phi = np.arctan2((shifted_vert[:,1]), (shifted_vert[:,0]))
    vertices = vertices[np.argsort(phi), :]
    # Return vertices as [center, ... cyclic-sorted vertices ...]
    return np.append([center], vertices, axis=0)

'''
Triangulation and area calculation functions (currently not needed)
'''
def triangle_area(vert):
    # The norm of the cross product of two sides is twice the area
    n = np.cross(vert[1] - vert[0], vert[2] - vert[0])
    return np.linalg.norm(n) / 2

def compute_all_areas(vert_abc, lat, tri_idx=None):
    # Rescaling
    vert_xyz = np.dot(vert_abc, lat)
    area = 0
    # If triangulation indices is not given, Delaunay (trivial) triangulation of hexagon is done automatically
    if tri_idx is None:
        tri = Delaunay(vert_xyz[:, :2])
        tri_vert = np.array([vert_xyz[idx] for idx in tri.simplices])
    else:
        tri_vert = np.array([vert_xyz[idx] for idx in tri_idx])
    for vert in tri_vert:
        area += triangle_area(vert)
    return tri_vert, area

def plot_triangulation(tri_vert, area, xyz, atom, fname="area_element", fid="0"):
    fig = plt.figure(figsize=(8, 8))
    for coords in tri_vert:
        coords = np.append(coords, [coords[0]], axis=0)
        xs, ys, zs = zip(*coords)
        plt.plot(xs, ys, c="b")
    for ele in np.unique(atom):
        xyz_at = xyz[np.array(atom) == ele]
        plt.scatter(xyz_at[:, 0], xyz_at[:, 1], label=ele)
    plt.title(f"Area = {area}")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f"./img/{fname}_{fid}.png")
    plt.clf()
    return

'''
Compute all area elements function
lat: lattice cell vectors (for rescaling)
abc: frac coords of all atoms
atom: atom type of all atoms
n_center: assumed first 30 atoms are center of area elements (all B atoms)
return_frac: return all vertices as frac coords, otherwise rescale the vertices
ignore_center: reject the center atom of area elements as one of the vertices
'''
def all_element_vertices(lat:np.ndarray, abc:np.ndarray, atom:list, n_center=30, return_frac=True, ignore_center=True):
    all_ele = []
    for center_idx in range(n_center):
        center, center_atom = abc[center_idx], atom[center_idx]
        abc_pbc, atom_pbc = minimum_pbc_2d(abc, atom, center)
        first_nn, second_nn = find_nn(abc_pbc, atom_pbc, center, center_atom)
        area_vert_abc = compute_area_vertices(first_nn, second_nn, center)
        if ignore_center:
            area_vert_abc = area_vert_abc[1:]
        if return_frac:
            all_ele.append(area_vert_abc)
        else:
            all_ele.append(np.dot(area_vert_abc, lat))
    return np.array(all_ele)

def plot_elements(ele_vert:np.ndarray, atom_pos:np.ndarray, atom:list, fname="./img/all_ele.png"):
    fig = plt.figure(figsize=(8, 8))
    for ele in np.unique(atom):
        target_pos = atom_pos[np.array(atom) == ele]
        plt.scatter(target_pos[:, 0], target_pos[:, 1], label=ele)
    for coords in ele_vert:
        coords = np.append(coords, [coords[0]], axis=0)
        xs, ys, zs = zip(*coords)
        plt.plot(xs, ys, c="b")
    plt.title(f"All elements")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(fname)
    plt.clf()

if __name__ == "__main__":
    '''
    Sample usage to generate all area elements from a POSCAR
    ele contains 6 vertices of each area elements (3D array), iterate it yields 6 vertices of each element (2D array)
    e.g.
    [[[A1x1, A1y1, A1z1],
      [A1x2, A1y2, A1z2],
      [A1x3, A1y3, A1z3],
      [A1x4, A1y4, A1z4],
      [A1x5, A1y5, A1z5],
      [A1x6, A1y6, A1z6]],
     [[A2x1, A2y1, A2z1],
      [A2x2, A2y2, A2z2],
      [A2x3, A2y3, A2z3],
      [A2x4, A2y4, A2z4],
      [A2x5, A2y5, A2z5],
      [A2x6, A2y6, A2z6]],
      ...
    ]]]
    '''
    fname = '../Vaspinput/POSCAR'
    lat, abc, xyz, atom = read_structure(fname)
    ele = all_element_vertices(lat, abc, atom, return_frac=False)
    print(ele)
    # print(len(ele))
    # plot_elements(ele, xyz, atom)

    # filename, center_idx = 'supercell35_60_perturbated_f.vasp', 28
    # lat, abc, xyz, atom = read_structure(filename)
    # center, center_atom = abc[center_idx], atom[center_idx]

    # abc_pbc, atom_pbc = minimum_pbc_2d(abc, atom, center)
    # first_nn, second_nn = find_nn(abc_pbc, atom_pbc, center, center_atom)
    # # print("Center, First NN, Second NN\n", np.concatenate([[center], first_nn, second_nn], axis=0))

    # area_vert_abc = compute_area_vertices(first_nn, second_nn, center)
    # print("7 points for area elements\n", area_vert_abc)

    # tri_vert, area = compute_all_areas(area_vert_abc, lat)
    # plot_triangulation(tri_vert, area, xyz, atom, fid="trivial")
    # print("trivial centered config area (angstrom squared):", area)

    # for i, permu in enumerate(noncenter_permu()):
    #     # Center is not used for the 14 unique permu
    #     tri_vert, area = compute_all_areas(area_vert_abc[1:], lat, permu)
    #     plot_triangulation(tri_vert, area, xyz, atom, fid=i)
    #     print(f"{i} non-center permu area (angstrom squared):", area)