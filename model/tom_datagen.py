import random

import pandas as pd
import numpy as np
import json
import os
from model.datasets.tom_dataset import CoordsEnergyDataset
from area_element.symmetry import radial_sym, angular_sym, pairwise_dist, pairwise_vec, pairwise_dist_area_element
from area_element.area import read_structure, all_element_vertices

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from time import time
import tqdm

def compute_time(outter_f):
    def inner_f(*args, **kwargs):
        t1 = time()
        print(f"\r\t Starting Runing {outter_f.__name__}")
        outter_f(*args, **kwargs)
        t2 = time()
        print(f"Runing Complete, time cost: {round((t2-t1)/60, 2)}min")

    return inner_f

'''
May take very long (a few hours)

For generating angular and radial symmetry functions of all stuctures into csv files

sroot (str): The path of directory storing stucture POSCARs, i.e. location of your AE_root directory
lpath (str): The path of json file storing the energies output, i.e. path of your AreaElement.json
rs_fname (str): filename of storing the radial symmetry csv files
as_fname (str): filename of storing the angular symmetry csv files
'''
@compute_time
def gen_symmetry_features(sroot = "AE_root", lpath = "AreaElement.json", rs_fname='rsym.csv', as_fname='asym.csv'):
    spath = [sroot+"/"+dir+"/POSCAR" for dir in os.listdir(sroot) if 'perturb' in dir]
    ds = CoordsEnergyDataset(spath, lpath, length=4450)
    rsym, asym = [], []

    R_s = [2., 3., 4., 5., 6., 7., 8., 9., 10.]
    eta = [0.01, 0.03, 0.06, 0.10, 0.20, 0.40, 1.00, 5.00]
    zeta = [1, 2, 4, 16, 64]
    lambdas = [-1, 1]

    for i, (_, _, xyz, _, en) in tqdm.tqdm(enumerate(ds)):

        pairwise_xyz = pairwise_dist(xyz)
        for rs in R_s:
            rsym.append(radial_sym(pairwise_xyz, R_s=rs, eta=1))

        for e in eta:
            rsym.append(radial_sym(pairwise_xyz, R_s=10, eta=e))

        for lbd in lambdas:
            for zt in zeta:
                asym.append(angular_sym(pairwise_vec(xyz), pairwise_xyz, lamb=lbd, ksi=zt))

        if i == (len(ds)-1): break # For some reason i needed this

    rsym = np.array(rsym)
    asym = np.array(asym)
    rsym = rsym.reshape(4450, -1)
    asym = asym.reshape(4450, -1)

    # print(rsym.shape, asym.shape)
    # 1020 = 17 (radial symmetry functions) x 60 (atoms)
    rsym_df = pd.DataFrame(rsym, columns=[f'rsym_{i}' for i in range(1020)])
    # 600 = 10 (angular symmetry functions) x 60 (atoms)
    asym_df = pd.DataFrame(asym, columns=[f'asym_{i}' for i in range(600)])

    rsym_df.to_csv(rs_fname, index=False, float_format='%.6f')
    asym_df.to_csv(as_fname, index=False, float_format='%.6f')


'''
May take some time (~2-8mins: depending on the methods you are using)

For generating volume elements vertices of all structures into csv files
Note it is reshaped into one volume element per row, so one structure actually occupies 30 rows

sroot (str): The path of directory storing stucture POSCARs, i.e. location of your AE_root directory
lpath (str): The path of json file storing the energies output, i.e. path of your AreaElement.json
ve_fname (str): filename of storing the volume elements vertices csv files
'''
@compute_time
def gen_volele_coords(sroot="AE_root", lpath="AreaElement.json", ve_fname='volele.csv', center_atom='N',
                      shuffle_volume_elements=False,
                      shuffle_vertices=False):
    """
    Generate coordinates of area element vertices
    :param sroot: root directory
    :param lpath: json file that saves all data points
    :param ve_fname: volume element file name to save the inputs data
    :param shuffle_volume_elements:  shuffle the ordering of area elements in each cell
    :param shuffle_vertices: shuffle the ordering of vertices in each area element
    :return: None
    """

    spath = [sroot+"/"+dir+"/POSCAR" for dir in os.listdir(sroot) if "perturb" in dir]
    ds = CoordsEnergyDataset(spath, lpath, length=4450)
    all_vol_ele = []
    for i, (lat, abc, _, atom, en) in enumerate(ds):
        # 30x6x3
        ele = all_element_vertices(lat, abc, atom,
                                   return_frac=False,
                                   cyclic_sort=True,
                                   shift_to_center=True,
                                   center_atom=center_atom)

        if shuffle_volume_elements:
            ordering = list(range(len(ele)))
            random.shuffle(ordering)
            ele = ele[ordering]

        if shuffle_vertices:
            new_ele=[]
            for each in ele:
                vertices_order = list(range(len(each)))
                random.shuffle(vertices_order)

                new_ele.append([each[k] for k in vertices_order])

            ele = np.array(new_ele)

        all_vol_ele.append(ele[:, :, :2])
        if i == (len(ds)-1): break # For some reason i needed this

    all_vol_ele = np.array(all_vol_ele)
    n_data, n_ve, n_vert, n_coords = all_vol_ele.shape
    all_vol_ele = np.reshape(all_vol_ele, (n_data*n_ve, n_vert*n_coords))
    # all_vol_ele = np.reshape(all_vol_ele, (27000, -1))
    volele_df = pd.DataFrame(all_vol_ele, columns=['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5'])
    volele_df.to_csv(ve_fname, index=False)


"""
May take some time (~2-5mins)

For generating volume elements pairwise distance and angles of all structures into csv files
Note it is reshaped into one volume element per row, so one structure actually occupies 30 rows

sroot (str): The path of directory storing stucture POSCARs, i.e. location of your AE_root directory
lpath (str): The path of json file storing the energies output, i.e. path of your AreaElement.json
ve_fname (str): filename of storing the volume elements pairwise distance and angle csv files
"""
@compute_time
def gen_pairwise_dist_angle(sroot="AE_root", lpath="AreaElement.json", ve_fname='pairwise_d_angle.csv', center_atom='B',
                            shuffle_volume_elements=False, shuffle_permuation=False):

    spath = [sroot + "/" + dir + "/POSCAR" for dir in os.listdir(sroot) if "perturb" in dir]
    ds = CoordsEnergyDataset(spath, lpath, length=4450)
    all_vol_ele = []
    for i, (lat, abc, _, atom, en) in enumerate(ds):
        # 30x6x3
        ele = all_element_vertices(lat, abc, atom,
                                   return_frac=False,
                                   cyclic_sort=True,
                                   shift_to_center=True,
                                   center_atom=center_atom)

        if shuffle_volume_elements:
            ordering = list(range(len(ele)))
            random.shuffle(ordering)
            ele = ele[ordering]

        descriptor = pairwise_dist_area_element(ele[:,:,:2])

        if shuffle_permuation:
            start_idx = random.randint(0, 6)
            descriptor = np.concatenate((descriptor[:, 2 * start_idx:], descriptor[:, :2 * start_idx]), axis=1)

        all_vol_ele.append(descriptor)

        if i == (len(ds) - 1): break  # For some reason i needed this

    all_vol_ele = np.array(all_vol_ele)
    n_data, n_ve, n_descriptor = all_vol_ele.shape
    all_vol_ele = np.reshape(all_vol_ele, (n_data * n_ve, n_descriptor))
    # all_vol_ele = np.reshape(all_vol_ele, (27000, -1))
    volele_df = pd.DataFrame(all_vol_ele,
                             columns=['d0', 'angle0', 'd1', 'angle1', 'd2', 'angle2', 'd3', 'angle3', 'd4', 'angle4',
                                      'd5', 'angle5'])
    volele_df.to_csv(ve_fname, index=False)

'''
For generating PCA features into csv files
Note this PCA is fitting by the full datasets, 
one may want to use energy_volele_pca_train_test_split instead to generate PCA features train-test csv

pca_fname (str): The filename of storing the pca features csv files
'''

@compute_time
def gen_pca_features(volele_df, pca_fname='pca.csv'):
    pca = PCA()
    pca.fit(volele_df)
    pca_df = pd.DataFrame(pca.transform(volele_df), columns=[f'PC{i}' for i in range(12)])
    pca_df.to_csv(pca_fname, index=False)

'''
Extract JSON file (AreaElement.json) to get all total energies and perturb strengths

lpath (str): The path of json file storing the energies output, i.e. path of your AreaElement.json
label_fname (str): The filename of storing the energy and perturb strengths csv files
n (int): The number of structures you have in the json file. For now it is 1000
'''
@compute_time
def energy_ps_labels(lpath='AreaElement.json', label_fname='label.csv', n=4450):
    with open(lpath) as json_file:
        label_dict = json.load(json_file)
    energy = np.full(n+1, np.nan, dtype=float)
    perturb_strength = np.full(n+1, np.nan, dtype=float)
    for key in label_dict.keys():
        if int(key)>=n:
            break
        energy[int(key)] = label_dict[key].get("total_energy", np.nan)
        perturb_strength[int(key)] = label_dict[key].get("perturb_strength", np.nan)
    label_df = pd.DataFrame({'total_energy': energy, 'ps': perturb_strength})
    label_df.to_csv(label_fname, index=False)

'''
Train-test split of energy, volume elements coordinates into separate csv files
Train set volume elements are used to fit PCA, stores transformed train-test PCA features into separate csv files

ve_fname (str): The filename of storing the train-test splitted volume elements csv
pca_fname (str): The filename of storing the train-test splitted pca features csv
label_fname (str): The filename of storing the train-test splitted energy labels csv
Prefix of train and test will be added at front of filename

test (float): ratio of test set, range: (0, 1)
n_ve (int): number of volume element per structure
length (int): length of your dataset, i.e. number of total structures
seed (int): RNG seed

Returns train and test indices for other uses
'''
def energy_volele_pca_train_test_split(volele_df, label_df, test=0.2, n_ve=30, length=4450, seed=14,
                                        ve_fname='ve.csv', pca_fname='pca.csv', label_fname='label.csv'
                                        ):
    idx = np.arange(length)
    np.random.seed(seed)
    np.random.shuffle(idx)
    train_idx, test_idx = idx[:-int(length*test)], idx[-int(length*test):]

    vol_ele_train_idx = np.array([np.arange(i*n_ve, (i+1)*n_ve) for i in train_idx]).flatten()
    vol_ele_test_idx = np.array([np.arange(i*n_ve, (i+1)*n_ve) for i in test_idx]).flatten()
    train_volele_df = volele_df.loc[volele_df.index[vol_ele_train_idx]]
    test_volele_df = volele_df.loc[volele_df.index[vol_ele_test_idx]]

    train_volele_df.to_csv('dataset/train_'+ve_fname, index=False)
    test_volele_df.to_csv('dataset/test_'+ve_fname, index=False)

    pca = PCA()
    pca.fit(train_volele_df)
    pcatrain = pd.DataFrame(pca.transform(train_volele_df), columns=[f'PC{i}' for i in range(12)])
    pcatest = pd.DataFrame(pca.transform(test_volele_df), columns=[f'PC{i}' for i in range(12)])

    pcatrain.to_csv('dataset/train_'+pca_fname, index=False)
    pcatest.to_csv('dataset/test_'+pca_fname, index=False)

    label_df.loc[label_df.index[train_idx]].to_csv('dataset/train_'+label_fname, index=False)
    label_df.loc[label_df.index[test_idx]].to_csv('dataset/test_'+label_fname, index=False)
    return train_idx, test_idx