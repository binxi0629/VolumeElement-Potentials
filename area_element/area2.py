from pymatgen.io.vasp import Poscar
import numpy as np
import matplotlib.pyplot as plt


def read_structure(fname: str):
    abc, xyz, atom = [], [], []
    poscar = Poscar.from_file(fname)
    lattice = poscar.as_dict()['structure']['lattice']['matrix']
    for site in poscar.as_dict()['structure']['sites']:
        abc.append(site['abc'])
        xyz.append(site['xyz'])
        atom.append(site['label'])
    return np.array(lattice), np.array(abc), np.array(xyz), atom


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

    def minimum_pbc_2d(self, center):
        a_shift = 1. if center[0] > 0.5 else -1.
        img_a = self.abc + np.array([a_shift, 0., 0.])

        b_shift = 1. if center[1] > 0.5 else -1.
        img_b = self.abc + np.array([0., b_shift, 0.])

        img_ab = self.abc + np.array([a_shift, b_shift, 0.])

        img_flag = np.reshape(
            [0, 0, 0] * len(self.atom) + [a_shift, 0, 0] * len(self.atom) + [0, b_shift, 0] * len(self.atom) +
            [a_shift, b_shift, 0] * len(self.atom), (4 * len(self.atom), 3))

        return np.concatenate([self.abc, img_a, img_b, img_ab], axis=0), self.atom * 4, img_flag

    def find_nn(self, fnn_noncen=3, snn_noncen=3, fnn_cen=6):
        atom_idx = np.array([i for i in range(len(self.atom_pbc) // 4)] * 4)
        atom_pbc = np.array(self.atom_pbc)

        shifted_abc = self.abc_pbc - self.center
        dist = np.dot(np.dot(shifted_abc, self.lat), np.dot(shifted_abc, self.lat).T).diagonal()
        # Atoms of different atomic type of center processing
        abc_filt_noncen = self.abc_pbc[atom_pbc != self.center_atom]
        dist_filt_noncen = dist[atom_pbc != self.center_atom]
        atom_idx_filt_noncen = atom_idx[atom_pbc != self.center_atom]
        img_flag_filt_noncen = self.img_flag[atom_pbc != self.center_atom]

        idx_sort_noncen = np.argsort(dist_filt_noncen)

        fnn_abc_noncen = abc_filt_noncen[idx_sort_noncen[:fnn_noncen]]
        fnn_idx_noncen = atom_idx_filt_noncen[idx_sort_noncen[:fnn_noncen]]
        fnn_img_noncen = img_flag_filt_noncen[idx_sort_noncen[:fnn_noncen]]

        snn_abc_noncen = abc_filt_noncen[idx_sort_noncen[fnn_noncen:fnn_noncen + snn_noncen]]
        snn_idx_noncen = atom_idx_filt_noncen[idx_sort_noncen[fnn_noncen:fnn_noncen + snn_noncen]]
        snn_img_noncen = img_flag_filt_noncen[idx_sort_noncen[fnn_noncen:fnn_noncen + snn_noncen]]
        # Atoms of same atomic type of center processing
        abc_filt_cen = self.abc_pbc[atom_pbc == self.center_atom]
        dist_filt_cen = dist[atom_pbc == self.center_atom]
        atom_idx_filt_cen = atom_idx[atom_pbc == self.center_atom]
        img_flag_filt_cen = self.img_flag[atom_pbc == self.center_atom]

        idx_sort_cen = np.argsort(dist_filt_cen)

        fnn_abc_cen = abc_filt_cen[idx_sort_cen[1:1 + fnn_cen]]
        fnn_idx_cen = atom_idx_filt_cen[idx_sort_cen[1:1 + fnn_cen]]
        fnn_img_cen = img_flag_filt_cen[idx_sort_cen[1:1 + fnn_cen]]

        return fnn_abc_noncen, snn_abc_noncen, fnn_abc_cen, fnn_idx_noncen, snn_idx_noncen, fnn_idx_cen, fnn_img_noncen, snn_img_noncen, fnn_img_cen

    def compute_area_vertices(self, cyclic_sort=False, shift_to_center=False):
        vertices_abc = np.array(self.fnn_abc_ncen)
        vertices_idx = np.array(self.fnn_idx_ncen)
        nn_avg_idx = np.reshape(list(self.fnn_idx_ncen) * 6, (6, 3)).T
        nn_avg_img = np.array([np.reshape(list(arr) * 6, (-1, 3)) for arr in self.fnn_img_ncen])

        for (site, snn_idx, snn_img) in zip(self.snn_abc_ncen, self.snn_idx_ncen, self.snn_img_ncen):
            dist_ncen = np.dot(np.dot(self.fnn_abc_ncen - site, self.lat),
                               np.dot(self.fnn_abc_ncen - site, self.lat).T).diagonal()
            close_fnn_abc_ncen = self.fnn_abc_ncen[dist_ncen != dist_ncen.max()]
            close_fnn_idx_ncen = self.fnn_idx_ncen[dist_ncen != dist_ncen.max()]
            close_fnn_img_ncen = self.fnn_img_ncen[dist_ncen != dist_ncen.max()]

            dist_cen = np.dot(np.dot(self.fnn_abc_cen - site, self.lat),
                              np.dot(self.fnn_abc_cen - site, self.lat).T).diagonal()
            close_fnn_abc_cen = self.fnn_abc_cen[np.argsort(dist_cen)[:2]]
            close_fnn_idx_cen = self.fnn_idx_cen[np.argsort(dist_cen)[:2]]
            close_fnn_img_cen = self.fnn_img_cen[np.argsort(dist_cen)[:2]]

            non_atom_vert = (self.center + site + close_fnn_abc_ncen.sum(axis=0) + close_fnn_abc_cen.sum(axis=0)) / 6

            non_atom_avg_idx = np.concatenate(([self.idx], [snn_idx], close_fnn_idx_ncen, close_fnn_idx_cen), axis=0)
            non_atom_avg_img = np.concatenate(([[0, 0, 0]], [snn_img], close_fnn_img_ncen, close_fnn_img_cen), axis=0)

            vertices_abc = np.append(vertices_abc, [non_atom_vert], axis=0)
            vertices_idx = np.append(vertices_idx, [-1], axis=0)
            nn_avg_idx = np.append(nn_avg_idx, [non_atom_avg_idx], axis=0)
            nn_avg_img = np.append(nn_avg_img, [non_atom_avg_img], axis=0)

        if cyclic_sort:
            shifted_vert = vertices_abc - self.center
            # I mistakenly input (x,y) but the docs suggest (y,x). Does not affect overall sorting tho
            phi = np.arctan2((shifted_vert[:, 0]), (shifted_vert[:, 1]))

            vertices_abc = vertices_abc[np.argsort(phi), :]
            vertices_idx = vertices_idx[np.argsort(phi)]
            nn_avg_idx = nn_avg_idx[np.argsort(phi), :]
            nn_avg_img = nn_avg_img[np.argsort(phi), :]

        if shift_to_center:
            return np.append([[0, 0, 0]], vertices_abc - self.center, axis=0), np.append([self.idx], vertices_idx,
                                                                                         axis=0), np.append(
                [[self.idx] * 6], nn_avg_idx, axis=0), np.append([np.zeros((6, 3))], nn_avg_img, axis=0)
        else:
            return np.append([self.center], vertices_abc, axis=0), np.append([self.idx], vertices_idx,
                                                                             axis=0), np.append([[self.idx] * 6],
                                                                                                nn_avg_idx,
                                                                                                axis=0), np.append(
                [np.zeros((6, 3))], nn_avg_img, axis=0)

    def all_elements_vertices_nnsearch(self, fnn_noncen=3, snn_noncen=3, fnn_cen=6, return_frac=True,
                                       return_center=True, cyclic_sort=True, shift_to_center=False):
        all_ele, all_ele_idx, all_nn_avg_idx, all_nn_avg_img = [], [], [], []
        for idx in self.center_idxs:
            self.idx = idx
            self.center, self.center_atom = self.abc[idx], self.atom[idx]
            self.abc_pbc, self.atom_pbc, self.img_flag = self.minimum_pbc_2d(self.center)
            self.fnn_abc_ncen, self.snn_abc_ncen, self.fnn_abc_cen, self.fnn_idx_ncen, self.snn_idx_ncen, self.fnn_idx_cen, self.fnn_img_ncen, self.snn_img_ncen, self.fnn_img_cen = self.find_nn(
                fnn_noncen=fnn_noncen, snn_noncen=snn_noncen, fnn_cen=fnn_cen)
            area_vert_abc, area_vert_idx, nn_avg_idx, nn_avg_img = self.compute_area_vertices(cyclic_sort=cyclic_sort,
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

    def all_elements_vertices_map_by_ref(self, return_frac=True, shift_to_center=True):
        ele = (self.abc[self.ref_nn_avg_idx] + self.ref_nn_avg_img).mean(axis=2)
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
'''
if __name__ == "__main__":
    import pandas as pd

    fname = './test_poscar/supercell35_60.vasp'
    n_center = 30
    # fname = './test_poscar/extra_18.vasp'
    # n_center = 20
    center_idx = [i + 30 for i in range(n_center)]  # List of lines you want to use as center
    AE = AreaElementDivider(center_idx)
    AE.fetch_structure(fname)
    AE.all_elements_vertices_nnsearch(return_frac=False, return_center=True, cyclic_sort=True)
    AE.set_reference(AE.all_nn_avg_idx, AE.all_nn_avg_img)
    AE.plot_elements(out_fname="./img_algotest/reference_Ncenter.png", frac=False)

    # all_vol_ele = []
    # for i in range(4450):
    #     if i % 100 == 0: print(f"processing {i}-th structure")
    #     fname = f'../AE_root_new/perturb_{i}/POSCAR'
    #     AE.fetch_structure(fname)
    #     AE.all_elements_vertices_map_by_ref(return_frac=False, shift_to_center=False)
    #     all_vol_ele.append(AE.all_ele[:, :, :2])
    #     AE.plot_elements(out_fname=f"./img_algotest/perturb_{i}.png", frac=False)

    # all_vol_ele = np.array(all_vol_ele)
    # n_data, n_ve, n_vert, n_coords = all_vol_ele.shape
    # all_vol_ele = np.reshape(all_vol_ele, (n_data*n_ve, n_vert*n_coords))

    # volele_df = pd.DataFrame(all_vol_ele, columns=['xc', 'yc', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5'])
    # volele_df.to_csv("../tom_data/3_stress/ve_all_Ncenter.csv", index=False)