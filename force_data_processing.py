import numpy as np
import pandas as pd
import torch

from pymatgen.io.vasp import Outcar
from area_element.area2 import read_structure
from area_element.area2 import AreaElementDivider
from model.datasets.tom_dataset import VEFeaturesEnergyDataset, CoordsEnergyDataset
from model.tom_model import SubNet, ModelContainer

"""
=========================================================================================================
|                                           Initial Settings                                            |
=========================================================================================================
"""
_SEED = 411
_LENGTH = 4450
TRAIN_TEST_SPLIT = 0.2
NUM_ELE=30
center_atom ='N'
"""
=========================================================================================================
|                                           Forces Extracting                                           |
=========================================================================================================
"""

# all_forces = []
# for i in range(_LENGTH):
#     if i % 100 == 0: print(f"processing {i}-th structure")
#
#     fname = f'../../Volume-Element-Project/data/AE_root/perturb_{i}/OUTCAR'
#     out = Outcar(fname)
#
#     forces = np.array(out.read_table_pattern(
#     header_pattern=r"\sPOSITION\s+TOTAL-FORCE \(eV/Angst\)\n\s-+",
#     row_pattern=r"\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+[+-]?\d+\.\d+\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)",
#     footer_pattern=r"\s--+",
#     postprocess=lambda x: float(x),
#     last_one_only=False
#     ))
#     if center_atom == 'B':
#         all_forces.append(forces[0][:30, :2])
#     else:
#         all_forces.append(forces[0][30:60, :2])
#
# all_forces = np.array(all_forces)

"""
=========================================================================================================
|                                         Save forces to csv                                            |
=========================================================================================================
"""

# df = pd.read_csv("dataset/volele.csv")
# label_df = pd.read_csv('dataset/label.csv')
#
# df["fx"] = all_forces.reshape(-1, 2)[:,0]
# df["fy"] = all_forces.reshape(-1, 2)[:,1]
# df.to_csv(f"dataset/ve_force_{center_atom}.csv", index=False)

"""
=========================================================================================================
|                                         Train test split                                              |
=========================================================================================================
"""
idx = np.arange(_LENGTH)

np.random.seed(11401996)
np.random.shuffle(idx)

df = pd.read_csv(f"dataset/ve_force_{center_atom}.csv")
label_df = pd.read_csv('dataset/label.csv')


train_idx, test_idx = idx[:-int(_LENGTH*TRAIN_TEST_SPLIT)], idx[-int(_LENGTH*TRAIN_TEST_SPLIT):]
print(f"\tTRAIN split: {len(train_idx)} | TEST split: {len(test_idx)}")

vol_ele_train_idx = np.array([np.arange(i*NUM_ELE, (i+1)*NUM_ELE) for i in train_idx]).flatten()
vol_ele_test_idx = np.array([np.arange(i*NUM_ELE, (i+1)*NUM_ELE) for i in test_idx]).flatten()

train_volele_df = df.loc[df.index[vol_ele_train_idx]]
test_volele_df = df.loc[df.index[vol_ele_test_idx]]

# train_volele_df = total_df.loc[total_df.index[vol_ele_train_idx]]
# test_volele_df = total_df.loc[total_df.index[vol_ele_test_idx]]

train_volele_df.to_csv("dataset/train_ve_force.csv", index=False)
test_volele_df.to_csv("dataset/test_ve_force.csv", index=False)

label_df.loc[label_df.index[train_idx]].to_csv('dataset/train_label_force.csv', index=False)
label_df.loc[label_df.index[test_idx]].to_csv('dataset/test_label_force.csv', index=False)

# total_label.loc[total_label.index[train_idx]].to_csv('dataset/train_label_force.csv', index=False)
# total_label.loc[total_label.index[test_idx]].to_csv('dataset/test_label_force.csv', index=False)

"""
=========================================================================================================
|                                     predict elastic energy                                            |
=========================================================================================================
"""
# energy_model_cols=["x0","y0","x1","y1","x2","y2","x3","y3","x4","y4","x5","y5"]
# ve_train = pd.read_csv("dataset/train_ve_force.csv")[energy_model_cols]
# ve_test = pd.read_csv("dataset/test_ve_force.csv")[energy_model_cols]
# label_train = pd.read_csv('dataset/train_label_force.csv')
# label_test = pd.read_csv('dataset/test_label_force.csv')
#
# print(ve_train.shape, label_train.shape, ve_test.shape, label_test.shape)
#
# n_input = 12
# ds_train = VEFeaturesEnergyDataset(ve_train, label_train, n_col=n_input)
# ds_train.normalize_y(mode='offset', offset=527)
# ds_train.normalize_X(mode='standard')
#
# # This shuffle is shuffle each data sample when iterating the whole dataset
# train_loader = torch.utils.data.DataLoader(ds_train, batch_size=8, shuffle=True)
#
# torch.manual_seed(_SEED)
# model = ModelContainer(SubNet(n_input, c=8))
# print(model.model)
# model.fit(train_loader, 50)
#
# ds_test = VEFeaturesEnergyDataset(ve_test, label_test, n_col=n_input)
# ds_test.normalize_y(mode='offset', offset=527)
# ds_test.normalize_X(mode='standard', pre_mean=ds_train.pca_mean, pre_std=ds_train.pca_std)
#
# # This shuffle is shuffle each data sample when iterating the whole dataset
# train_force_loader = torch.utils.data.DataLoader(ds_train, batch_size=8, shuffle=False)
# test_force_loader = torch.utils.data.DataLoader(ds_test, batch_size=8, shuffle=False)
#
# dtrain = pd.read_csv("dataset/train_ve_force.csv")
# dtrain['sub_e'] = model.predict(train_force_loader, save_sub_preds=True)[2]
# dtest = pd.read_csv("dataset/test_ve_force.csv")
# dtest['sub_e'] = model.predict(test_force_loader, save_sub_preds=True)[2]
#
# dtrain.to_csv("dataset/train_ve_force_sube.csv", index=False)
# dtest.to_csv("dataset/test_ve_force_sube.csv", index=False)

"""
=========================================================================================================
|                                     prepare forces data                                              |
=========================================================================================================
"""


