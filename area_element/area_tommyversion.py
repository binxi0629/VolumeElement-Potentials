from pymatgen.io.vasp import Poscar
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay,delaunay_plot_2d,ConvexHull
from itertools import cycle

def read_structure(fname):
    abc, xyz, atom = [], [], []
    poscar = Poscar.from_file(fname)
    lattice = poscar.as_dict()['structure']['lattice']['matrix']
    for site in poscar.as_dict()['structure']['sites']:
        abc.append(site['abc'])
        xyz.append(site['xyz'])
        atom.append(site['label'])
    return np.array(lattice), np.array(abc), np.array(xyz), atom

# This method neglects z-direction
def pbc_2d(abc, atom, center):
    shift = []
    # Along a-direction (aka x-direction)
    # add right image if center is at the right side, vice versa
    if center[0] > 0.5:
        shift.append([1., 0., 0.])
    else:
        shift.append([-1., 0., 0.])
    # Along b-direction (aka y-direction)
    # add top image if center is at the top side, vice versa
    if center[1] > 0.5:
        shift.append([0., 1., 0.])
    else:
        shift.append([0., -1., 0.])
    # Along xy-direction (aka corners)
    # add extra top-right image if top and right image are needed, vice versa
    shift.append([shift[0][0], shift[1][1], 0.])

    img_a = abc + np.array(shift[0])
    img_b = abc + np.array(shift[1])
    img_ab = abc + np.array(shift[2])
    img_atom = atom.copy()
    for i in range(3):
        atom += img_atom
    return atom, np.concatenate([abc, img_a, img_b, img_ab], axis=0)

# This method finds first and second nearest neighbours (of different kind of atoms than center atom) in fractional coords
def find_nn(abc, atom, center, center_atom, first_nn=3, second_nn=6):
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
    second_nn_abc = abc_filt[first_nn:second_nn]
    return first_nn_abc, second_nn_abc

# This method is for obtaining the 7 points of the area element, includes computing the non-atom vertices by the triangle-averaging method (using second NN and first NN)
def compute_area_vertices(center, first_nn_abc, second_nn_abc):
    # Area element vertices, the center and first NN atoms must be the vertices
    area_vertices = np.append([center], first_nn_abc, axis=0)
    # print(area_vertices)
    # Find the non-atom sites for vertices
    for site in second_nn_abc:
        # For each atom of second NN, shift the atom to origin and calculate the dist of first NNs
        shifted_abc = first_nn_abc - site
        dist = np.dot(shifted_abc, shifted_abc.T).diagonal()
        # The two closer points to the second NN atom site forms the triangle (for averaging)
        close_points = first_nn_abc[dist != dist.max()]
        # The average of the triangle vertices gives the non-atom site vertex for area element
        non_atom_vert = np.append([site], close_points, axis=0).mean(axis=0)
        area_vertices = np.append(area_vertices, [non_atom_vert], axis=0)
    return area_vertices

def triangle_area(vert):
    # The norm of the cross product of two sides is twice the area
    n = np.cross(vert[1] - vert[0], vert[2] - vert[0])
    return np.linalg.norm(n) / 2

def genTriangles(i, j):
    if j - i < 2:
        yield []
        return
    
    if j - i == 2:
        yield [(i, i+1, j)]
        return 
    
    for k in range(i + 1, j):
        for x in genTriangles(i, k):
            for y in genTriangles(k, j):
                yield x + y + [(i, k, j)]


# WIP: Stuck at computing all permutation of triangles inside hexagon
def compute_all_areas(vertices, lat):
    # print(vertices,lat)
    rescaled_vert = np.dot(vertices, lat)
    # The special case with delanuay
    tri = Delaunay(rescaled_vert[:,:-1])

    # Obtain all edges, and then start looping 
    hull = ConvexHull(rescaled_vert[:,:-1])
    n = len(hull.vertices)
    with open("TrianglesLOG",'w+') as log:
        for k, tr in enumerate(genTriangles(0, n - 1), 1):
            log.write(f"{k}\t\n")
            areas = 0
            for triangles in tr:
                print(rescaled_vert[hull.vertices])
                vert = rescaled_vert[hull.vertices][triangles,:]
                area = triangle_area(vert)
                areas= areas + area
                log.write(f"{area}\t")
                vertstr = np.char.mod('%f', vert.flatten())
                log.write("\t".join(vertstr))
                log.write("\n")
            log.write(f"AREA:{areas}\t")
    return tri,rescaled_vert
    
filename = 'POSCAR_Perturbed'
# Make sure the center is in fractional coords, and the center atom is really the correct kind of atom
center, center_atom = np.array([0.722214904, 0.299999936, 0.750039416]), 'B'
lat, abc, _, atom = read_structure(filename)
atom, abc = pbc_2d(abc, atom, center)

first_nn, second_nn = find_nn(abc, atom, center, center_atom)
print("Center, First NN, Second NN\n", np.concatenate([[center], first_nn, second_nn], axis=0))

area_vert = compute_area_vertices(center, first_nn, second_nn)
print("7 points for area elements\n", area_vert)

tri,rescaled_vert = compute_all_areas(area_vert, lat)