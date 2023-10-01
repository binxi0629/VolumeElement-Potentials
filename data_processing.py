import pandas as pd
import numpy as np
from model.tom_datagen import (
                                gen_symmetry_features,
                                gen_volele_coords,
                                energy_ps_labels,
                                energy_volele_pca_train_test_split,
                                gen_pairwise_dist_angle
                            )


# Generating angular and radial symmetry features: 832min
# gen_symmetry_features(sroot="../../Volume-Element-Project/data/AE_root",
#                       lpath="../../Volume-Element-Project/data/AreaElement.json",
#                       rs_fname='rsym.csv', as_fname='asym.csv')

shuffle_ve = False
shuffle_vert = False

# For generating volume elements vertices of all structures into csv file
ve_fname = "volele_shuffled.csv" if shuffle_ve or shuffle_vert else "volele.csv"
#
gen_volele_coords(sroot="../../Volume-Element-Project/data/AE_root",
                  lpath="../../Volume-Element-Project/data/AreaElement.json",
                  ve_fname=f"dataset/{ve_fname}",
                  center_atom='N',
                  shuffle_volume_elements=shuffle_ve,
                  shuffle_vertices=shuffle_vert)

# Generate pairwise distance and angle of area element as descriptors
# gen_pairwise_dist_angle(
#                   sroot="../../Volume-Element-Project/data/AE_root",
#                   lpath="../../Volume-Element-Project/data/AreaElement.json",
#                   ve_fname=f"dataset/pairwise_d_angle.csv",
#                   center_atom='B',
#                   shuffle_volume_elements=False,
#                   shuffle_permuation=True)


# Extract json file to get all energies and perturb strengths
energy_ps_labels(lpath="../../Volume-Element-Project/data/AreaElement.json", label_fname='dataset/label.csv', n=4450)

# Train-test split of energy, volume elements coordinates into separate csv files
# Train set volume elements are used to fit PCA, stores transformed train-test PCA features into separate csv files
volele_df = pd.read_csv(f'dataset/{ve_fname}')
# volele_df = pd.read_csv(f'dataset/pairwise_d_angle.csv')
label_df = pd.read_csv('dataset/label.csv')
#

"""
    prepare additional dataset
"""

volele_df2 = pd.read_csv('dataset/volele_v2.csv')
label_df2 = pd.read_csv('dataset/label_v2.csv')

frames = [volele_df, volele_df2]
lables = [label_df, label_df2]

volele_all_df = pd.concat(frames)
lables_all_df = pd.concat()

train_idx, test_idx = energy_volele_pca_train_test_split(volele_df, label_df, test=0.2, n_ve=30, length=4450, seed=42,
                                        ve_fname='ve.csv', pca_fname='pca.csv', label_fname='label.csv')

# Manually perform train test split for rysm and asym
asym_df = pd.read_csv('dataset/asym.csv')
asym_df.loc[asym_df.index[train_idx]].to_csv('dataset/train_asym.csv', index=False)
asym_df.loc[asym_df.index[test_idx]].to_csv('dataset/test_asym.csv', index=False)

rsym_df = pd.read_csv('dataset/rsym.csv')
rsym_df.loc[rsym_df.index[train_idx]].to_csv('dataset/train_rsym.csv', index=False)
rsym_df.loc[rsym_df.index[test_idx]].to_csv('dataset/test_rsym.csv', index=False)