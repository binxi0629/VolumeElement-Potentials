from pymatgen.io.vasp import Poscar, Outcar
import numpy as np
import matplotlib.pyplot as plt
import math

# from triangulation import noncenter_permu
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from tqdm import tqdm


def read_structure(fname: str):
    abc, xyz, atom = [], [], []
    poscar = Poscar.from_file(fname)
    lattice = poscar.as_dict()['structure']['lattice']['matrix']
    for site in poscar.as_dict()['structure']['sites']:
        abc.append(site['abc'])
        xyz.append(site['xyz'])
        atom.append(site['label'])
    return np.array(lattice), np.array(abc), np.array(xyz), atom


def get_forces(fname, center_idx):
    out = Outcar(fname)
    forces = np.array(out.read_table_pattern(
        header_pattern=r"\sPOSITION\s+TOTAL-FORCE \(eV/Angst\)\n\s-+",
        row_pattern=r"\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)",
        footer_pattern=r"\s--+",
        postprocess=lambda x: float(x),
        last_one_only=False
    ))
    return forces[0][center_idx, :2]


'''
Area Element Divider Object for processing POSCAR. Sample usage in __main__ section.

Initialization:
center_idxs: array of indices of center atoms, i.e. line numbers in POSCAR files

Object Attributes:

Available after calling fetch_structure
lat: lattice vectors of the latest fetched structure
abc: fractional coordinates of the latest fetched structure
xyz: real space coordinates of the latest fetched structure
atom: atom types of the latest fetched structure

Available after calling all_element_vertices_nnsearch
all_ele: vertices coordinates of all area elements of the fetched structure,
         first row is center atom (if return center enabled), 
         subsequent rows are vertices sorted in cyclic order (if sorting enabled).
         e.g.
            [[[A1xc, A1yc, A1zc],
              [A1x1, A1y1, A1z1],
              [A1x2, A1y2, A1z2],
              [A1x3, A1y3, A1z3],
              [A1x4, A1y4, A1z4],
              [A1x5, A1y5, A1z5],
              [A1x6, A1y6, A1z6]],

             [[A2xc, A2yc, A2zc],
              [A2x1, A2y1, A2z1],
              [A2x2, A2y2, A2z2],
              [A2x3, A2y3, A2z3],
              [A2x4, A2y4, A2z4],
              [A2x5, A2y5, A2z5],
              [A2x6, A2y6, A2z6]],
            ...
            ]]]
all_ele_idx: atom indices (i.e. line numbers in POSCAR, -1 for non atom vertices) of vertices of all area elements of the fetched structure,
             same structure as above but 1 column per row (index) instead of 3 (xyz coords).
all_nn_avg_idx: atom indices (i.e. line numbers in POSCAR) of nearest neighbors atoms for averaging to give vertices coordinates,
                can be used to set as reference and compute area elements for other structures.
all_nn_avg_img: image shifts for each nearest neighbors atom above (i.e. PBC conditions),
                can be used to set as reference and compute area elements for other structures. 

Available after calling set_reference, should be input to all_element_vertices_map_by_ref
ref_nn_avg_idx: reference atom indices
ref_nn_avg_img: reference atom image shifts

Other attributes are junk / intermediate values of function calls, not meaningful
'''


class AreaElementDivider():
    def __init__(self, center_idxs):
        self.center_idxs = center_idxs

    def fetch_structure(self, fname):
        self.lat, self.abc, self.xyz, self.atom = read_structure(fname)

    def set_reference(self, ref_nn_avg_idx, ref_nn_avg_img):
        self.ref_nn_avg_idx = ref_nn_avg_idx
        self.ref_nn_avg_img = ref_nn_avg_img

    def sqdist_to_orgin(self, coords, frac=False):
        if frac:
            return np.dot(coords, coords.T).diagonal()
        else:
            return np.dot(np.dot(coords, self.lat), np.dot(coords, self.lat).T).diagonal()

    def minimum_pbc_2d(self, center):
        a_shift = 1. if center[0] > 0.5 else -1.
        img_a = self.abc + np.array([a_shift, 0., 0.])

        b_shift = 1. if center[1] > 0.5 else -1.
        img_b = self.abc + np.array([0., b_shift, 0.])

        img_ab = self.abc + np.array([a_shift, b_shift, 0.])

        img_flag = np.reshape(
            [0, 0, 0] * len(self.atom) + [a_shift, 0, 0] * len(self.atom) + [0, b_shift, 0] * len(self.atom) + [a_shift,
                                                                                                                b_shift,
                                                                                                                0] * len(
                self.atom), (4 * len(self.atom), 3))
        return np.concatenate([self.abc, img_a, img_b, img_ab], axis=0), self.atom * 4, img_flag

    def find_nn(self, nn1_ncen=3, nn2_ncen=3, nn1_cen=6):
        atom_idx = np.array([i for i in range(len(self.atom_pbc) // 4)] * 4)
        atom_pbc = np.array(self.atom_pbc)

        shifted_abc = self.abc_pbc - self.center
        # dist = np.dot(np.dot(shifted_abc, self.lat), np.dot(shifted_abc, self.lat).T).diagonal()
        dist = self.sqdist_to_orgin(shifted_abc)
        # Atoms of different atomic type of center processing
        abc_filt_noncen = self.abc_pbc[atom_pbc != self.center_atom]
        dist_filt_noncen = dist[atom_pbc != self.center_atom]
        atom_idx_filt_noncen = atom_idx[atom_pbc != self.center_atom]
        img_flag_filt_noncen = self.img_flag[atom_pbc != self.center_atom]

        idx_sort_noncen = np.argsort(dist_filt_noncen)
        start, end = 0, nn1_ncen
        self.nn1_abc_ncen = abc_filt_noncen[idx_sort_noncen[start:end]]
        self.nn1_idx_ncen = atom_idx_filt_noncen[idx_sort_noncen[start:end]]
        self.nn1_img_ncen = img_flag_filt_noncen[idx_sort_noncen[start:end]]

        start, end = nn1_ncen, nn1_ncen + nn2_ncen
        self.nn2_abc_ncen = abc_filt_noncen[idx_sort_noncen[start:end]]
        self.nn2_idx_ncen = atom_idx_filt_noncen[idx_sort_noncen[start:end]]
        self.nn2_img_ncen = img_flag_filt_noncen[idx_sort_noncen[start:end]]

        # Atoms of same atomic type of center processing
        abc_filt_cen = self.abc_pbc[atom_pbc == self.center_atom]
        dist_filt_cen = dist[atom_pbc == self.center_atom]
        atom_idx_filt_cen = atom_idx[atom_pbc == self.center_atom]
        img_flag_filt_cen = self.img_flag[atom_pbc == self.center_atom]

        idx_sort_cen = np.argsort(dist_filt_cen)
        start, end = 1, 1 + nn1_cen
        self.nn1_abc_cen = abc_filt_cen[idx_sort_cen[start:end]]
        self.nn1_idx_cen = atom_idx_filt_cen[idx_sort_cen[start:end]]
        self.nn1_img_cen = img_flag_filt_cen[idx_sort_cen[start:end]]

    def big_element(self, sort_vertices=False, shift_to_center=False):
        ele_vertices = np.concatenate([self.nn1_abc_ncen, self.nn2_abc_ncen, self.nn1_abc_cen], axis=0)
        ele_idxs = np.concatenate([self.nn1_idx_ncen, self.nn2_idx_ncen, self.nn1_idx_cen])
        ele_img = np.concatenate([self.nn1_img_ncen, self.nn2_img_ncen, self.nn1_img_cen], axis=0)
        if sort_vertices:
            idx_tmp = np.argsort(ele_vertices.sum(axis=1))
            ele_vertices = ele_vertices[idx_tmp, :]
            ele_idxs = ele_idxs[idx_tmp]
            ele_img = ele_img[idx_tmp, :]

        if shift_to_center:
            return np.append([[0, 0, 0]], ele_vertices - self.center, axis=0), np.append([self.idx], ele_idxs,
                                                                                         axis=0), np.append([self.idx],
                                                                                                            ele_idxs,
                                                                                                            axis=0), np.append(
                [[0, 0, 0]], ele_img, axis=0)
        else:
            return np.append([self.center], ele_vertices, axis=0), np.append([self.idx], ele_idxs, axis=0), np.append(
                [self.idx], ele_idxs, axis=0), np.append([[0, 0, 0]], ele_img, axis=0)

    def compute_area_vertices(self, sort_vertices=False, shift_to_center=False):
        n_samplers = 6  # number of atomic position for computing centroid (from hexagon)
        vertices_abc = np.array(self.nn1_abc_ncen)
        vertices_idx = np.array(self.nn1_idx_ncen)
        nn_avg_idx = np.reshape(list(self.nn1_idx_ncen) * n_samplers, (n_samplers, 3)).T
        nn_avg_img = np.array([np.reshape(list(arr) * n_samplers, (-1, 3)) for arr in self.nn1_img_ncen])

        for (site, site_idx, site_img) in zip(self.nn2_abc_ncen, self.nn2_idx_ncen, self.nn2_img_ncen):
            dist_tmp = self.sqdist_to_orgin(self.nn1_abc_ncen - site)
            idx_tmp = (dist_tmp != dist_tmp.max())
            hex1_nn1_abc_ncen = self.nn1_abc_ncen[idx_tmp]
            hex1_nn1_idx_ncen = self.nn1_idx_ncen[idx_tmp]
            hex1_nn1_img_ncen = self.nn1_img_ncen[idx_tmp]

            dist_tmp = self.sqdist_to_orgin(self.nn1_abc_cen - site)
            idx_tmp = np.argsort(dist_tmp)
            hex1_nn1_abc_cen = self.nn1_abc_cen[idx_tmp[:2]]
            hex1_nn1_idx_cen = self.nn1_idx_cen[idx_tmp[:2]]
            hex1_nn1_img_cen = self.nn1_img_cen[idx_tmp[:2]]

            non_atom_vert = (self.center + site + hex1_nn1_abc_ncen.sum(axis=0) + hex1_nn1_abc_cen.sum(
                axis=0)) / n_samplers
            non_atom_avg_idx = np.concatenate(([self.idx], [site_idx], hex1_nn1_idx_ncen, hex1_nn1_idx_cen), axis=0)
            non_atom_avg_img = np.concatenate(([[0, 0, 0]], [site_img], hex1_nn1_img_ncen, hex1_nn1_img_cen), axis=0)

            vertices_abc = np.append(vertices_abc, [non_atom_vert], axis=0)
            vertices_idx = np.append(vertices_idx, [-1], axis=0)
            nn_avg_idx = np.append(nn_avg_idx, [non_atom_avg_idx], axis=0)
            nn_avg_img = np.append(nn_avg_img, [non_atom_avg_img], axis=0)

        if sort_vertices:
            shifted_vert = vertices_abc - self.center
            # I mistakenly input (x,y) but the official docs suggest (y,x). Does not affect overall sorting though
            phi = np.arctan2(shifted_vert[:, 0], shifted_vert[:, 1])
            idx_tmp = np.argsort(phi)
            vertices_abc = vertices_abc[idx_tmp, :]
            vertices_idx = vertices_idx[idx_tmp]
            nn_avg_idx = nn_avg_idx[idx_tmp, :]
            nn_avg_img = nn_avg_img[idx_tmp, :]

        if shift_to_center:
            return np.append([[0, 0, 0]], vertices_abc - self.center, axis=0), np.append([self.idx], vertices_idx,
                                                                                         axis=0), np.append(
                [[self.idx] * n_samplers], nn_avg_idx, axis=0), np.append([np.zeros((n_samplers, 3))], nn_avg_img,
                                                                          axis=0)
        else:
            return np.append([self.center], vertices_abc, axis=0), np.append([self.idx], vertices_idx,
                                                                             axis=0), np.append(
                [[self.idx] * n_samplers], nn_avg_idx, axis=0), np.append([np.zeros((n_samplers, 3))], nn_avg_img,
                                                                          axis=0)

    def all_elements_vertices_nnsearch(self, nn1_ncen=3, nn2_ncen=3, nn1_cen=6, return_frac=True, return_center=True,
                                       sort_vertices=True, shift_to_center=False, method='small'):
        all_ele, all_ele_idx, all_nn_avg_idx, all_nn_avg_img = [], [], [], []
        for idx in self.center_idxs:
            self.idx = idx
            self.center, self.center_atom = self.abc[idx], self.atom[idx]
            self.abc_pbc, self.atom_pbc, self.img_flag = self.minimum_pbc_2d(self.center)
            self.find_nn(nn1_ncen=nn1_ncen, nn2_ncen=nn2_ncen, nn1_cen=nn1_cen)
            if method == "small":
                area_vert_abc, area_vert_idx, nn_avg_idx, nn_avg_img = self.compute_area_vertices(
                    sort_vertices=sort_vertices, shift_to_center=shift_to_center)
            elif method == "big":
                area_vert_abc, area_vert_idx, nn_avg_idx, nn_avg_img = self.big_element(sort_vertices=sort_vertices,
                                                                                        shift_to_center=shift_to_center)

            if not return_center:
                area_vert_abc = area_vert_abc[1:]
                area_vert_idx = area_vert_idx[1:]
                nn_avg_idx = nn_avg_idx[1:]
                nn_avg_img = nn_avg_img[1:]

            if return_frac:
                all_ele.append(area_vert_abc)
            else:
                all_ele.append(np.dot(area_vert_abc, self.lat))

            all_ele_idx.append(area_vert_idx)
            all_nn_avg_idx.append(nn_avg_idx)
            all_nn_avg_img.append(nn_avg_img)

        self.all_ele, self.all_ele_idx, self.all_nn_avg_idx, self.all_nn_avg_img = np.array(all_ele), np.array(
            all_ele_idx), np.array(all_nn_avg_idx), np.array(all_nn_avg_img)

    '''
    No return_center and cyclic_sort options, it is based on the input reference set
    '''

    def all_elements_vertices_map_by_ref(self, return_frac=True, shift_to_center=True, method="small"):
        if method == "small":
            ele = (self.abc[self.ref_nn_avg_idx] + self.ref_nn_avg_img).mean(axis=2)
        elif method == "big":
            ele = self.abc[self.ref_nn_avg_idx] + self.ref_nn_avg_img

        if shift_to_center: ele = ele - ele[:, 0, np.newaxis]
        if not return_frac: ele = np.dot(ele, self.lat)
        self.all_ele = ele

    def plot_elements(self, out_fname, frac=True, annotate_idx=True):
        fig = plt.figure(figsize=(8, 8))

        atom_pos = self.abc if frac else self.xyz
        for ele in np.unique(self.atom):
            target_pos = atom_pos[np.array(self.atom) == ele]
            plt.scatter(target_pos[:, 0], target_pos[:, 1], label=ele)

        ele_vert_tmp = self.all_ele[:, 1:] if len(self.all_ele[0]) == 7 else self.all_ele
        for coords in ele_vert_tmp:
            coords = np.append(coords, [coords[0]], axis=0)
            xs, ys, zs = zip(*coords)
            plt.plot(xs, ys, c="b", lw=1)

        if annotate_idx:
            for (coords, ids) in zip(self.all_ele, self.all_ele_idx):
                for (c, id) in zip(coords, ids):
                    plt.annotate(str(id), (c[0], c[1]), c='red')

        plt.title(f"All elements")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlim(-2, 15)
        plt.ylim(-2, 15)
        plt.savefig(out_fname)
        plt.close(fig)


'''
sample usage

generating Boron/Nitrogen area element vertices with force at Boron/Nitrogen center
'''
if __name__ == "__main__":
    import pandas as pd

    '''
    area_element_method: 
        "small" for old method (computing non atom vertices by centroid of three hexagons)
        "big" for new method (include the atomic position of three hexagons)

    fname: filename for reference structure (unperturbed)

    start_idx, end_idx: indices range for center atoms
        [0, 30) for B centered
        [30, 60) for N centered
    '''
    area_element_method = "small"
    fname = './test_poscar/supercell35_60.vasp'
    # fname = './test_poscar/extra_18.vasp'
    start_idx, end_idx = 30, 60
    center_idx = [i for i in range(start_idx, end_idx)]  # List of lines you want to use as center
    AE = AreaElementDivider(center_idx)
    AE.fetch_structure(fname)
    AE.all_elements_vertices_nnsearch(return_frac=False, return_center=True, sort_vertices=True,
                                      method=area_element_method)
    AE.set_reference(AE.all_nn_avg_idx, AE.all_nn_avg_img)
    # AE.plot_elements(out_fname="./img_algotest/reference_Ncenter.png", frac=False, annotate_idx=False)

    all_vol_ele = []
    all_forces = []
    for i in range(4450):
        if i % 100 == 0: print(f"processing {i}-th structure")
        fname = f'../AE_root_new/perturb_{i}/POSCAR'
        AE.fetch_structure(fname)
        AE.all_elements_vertices_map_by_ref(return_frac=False, shift_to_center=True, method=area_element_method)
        all_vol_ele.append(AE.all_ele[:, :, :2])
        # AE.plot_elements(out_fname=f"./img_algotest/perturb_{i}.png", frac=False)

        fname = f'../AE_root_new/perturb_{i}/OUTCAR'
        all_forces.append(get_forces(fname, center_idx))

    all_vol_ele = np.array(all_vol_ele)
    n_data, n_ve, n_vert, n_coords = all_vol_ele.shape
    all_vol_ele = np.reshape(all_vol_ele, (n_data * n_ve, n_vert * n_coords))

    n_atomic_coords = 6 if area_element_method == "small" else 12
    tmp = [item for sublist in [[f'x{i}', f'y{i}'] for i in range(n_atomic_coords)] for item in sublist]
    volele_df = pd.DataFrame(all_vol_ele, columns=['xc', 'yc'] + tmp)

    all_forces = np.array(all_forces)
    volele_df["fx"] = all_forces.reshape(-1, 2)[:, 0]
    volele_df["fy"] = all_forces.reshape(-1, 2)[:, 1]

    volele_df.to_csv("../tom_data/3_stress/ve_all_Ncenter_test.csv", index=False)