import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

from model.datasets.tom_dataset import SymEnergyDataset
from model.tom_model import SubNet, ModelContainer
from area_element.area import read_structure
from area_element.symmetry import pairwise_dist, pairwise_vec, radial_sym, angular_sym
from model.datasets.tom_dataset import VEFeaturesEnergyDataset, CoordsEnergyDataset, VEFeaturesEnergyStressDataset
from area_element.area import read_structure, all_element_vertices, align_with_template
from area_element.symmetry import pairwise_dist_area_element

from sklearn.decomposition import PCA
from scipy.misc import derivative
from copy import deepcopy


def plot_results(preds, labels, title=None):
    plt.figure(figsize=(10, 10))
    plt.scatter(labels, preds, c='crimson')
    # plt.yscale('log')
    # plt.xscale('log')

    p1 = max(max(preds), max(labels))
    p2 = min(min(preds), min(labels))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    if title:
      plt.title(title)
    plt.show()
    # plt.savefig("res.png")

def areaELement_stress():

    #load dataset
    ve_train = pd.read_csv('dataset/4450/Expanded_ve/train_ve_force_4450.csv')
    ve_test = pd.read_csv('dataset/4450/Expanded_ve/test_ve_force_4450.csv')
    label_train_energy = pd.read_csv('dataset/4450/Expanded_ve/train_label_force_4450.csv')
    label_test_energy = pd.read_csv('dataset/4450/Expanded_ve/test_label_force_4450.csv')

    # num of VE vertices as inputs
    n_clo = 26

    # prepare training set
    ds_train = VEFeaturesEnergyStressDataset(ve_train, label_train_energy, n_clo, 30, shuffle_X=False, spring_model=False)
    ds_train.normalize_y(mode='offset', offset=527)
    ds_train.normalize_X(mode='standard')
    # ds_train.normalize_X(mode='minmax')

    # prepare testing set
    ds_test = VEFeaturesEnergyStressDataset(ve_test, label_test_energy, n_clo, 30, shuffle_X=False)
    ds_test.normalize_X(mode='standard', pre_mean=ds_train.pca_mean, pre_std=ds_train.pca_std)
    # ds_test.normalize_X(mode='minmax', pre_max=ds_train.pca_max, pre_min=ds_train.pca_min)
    ds_test.normalize_y(mode='offset', offset=527)

    # prepare dataloader
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=8, shuffle=False)

    # build model
    model = ModelContainer(SubNet(n_clo, c=48, stress_fit=True), stress_fit=True)
    print(model.model)

    # start training
    model.fit(train_loader, 5)

    from stress_run import rmse, mape, process_df2, predict2, plot_df2
    forces1, pred_force1, elastics1, pred_elastic1 = predict2(model, test_loader)

    res = pd.DataFrame({'pred_force_x': pred_force1[:, 0].flatten(), 'pred_force_y': pred_force1[:, 1].flatten(),
                        'force_x': forces1[:, 0].flatten(), 'force_y': forces1[:, 1].flatten()})

    res2 = pd.DataFrame({'pred_energy': pred_elastic1.flatten(), 'energy': elastics1.flatten()})
    res, res2 = process_df2(res, res2)
    pd.set_option('max_columns', None)
    print(res.describe())
    print(res2.describe())
    plot_df2(res, res2, title='test set performance')
    plt.show()
    # plt.savefig('stress_res/testset.png')

    outlier_thres = res.mean() + 3 * res.std()
    for col, thres in zip(outlier_thres.index, outlier_thres):
        print(col, thres, (res[col] > thres).sum(), (res[col] > thres).sum() / len(res))
        if 'ape' in col:
            tmp = res.loc[res[col] < thres, col]
            print(tmp.mean())


def areaElement():
    # Load dataset
    ve_train = pd.read_csv('dataset/ve/train_ve_force_total.csv')
    ve_test = pd.read_csv('dataset/ve/test_ve_force_total.csv')
    label_train = pd.read_csv('dataset/ve/train_label_force_total.csv')
    label_test = pd.read_csv('dataset/ve/test_label_force_total.csv')

    # num of vertices as inputs
    n_clo = 12

    # prepare training set
    ds_train = VEFeaturesEnergyDataset(ve_train, label_train, n_col=n_clo)
    ds_train.normalize_y(mode='offset', offset=527)
    ds_train.normalize_X(mode='standard')

    # prepare testing set
    ds_test = VEFeaturesEnergyDataset(ve_test, label_test, n_col=n_clo)
    ds_test.normalize_y(mode='offset', offset=527)
    ds_test.normalize_X(mode='standard', pre_mean=ds_train.pca_mean, pre_std=ds_train.pca_std)

    # prepare data loader
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=10, shuffle=False)

    # build model
    model = ModelContainer(SubNet(n_clo, c=32))
    print(model.model)

    # start training
    model.fit(train_loader, 100)

    # trainset error statistic
    train_truth, train_pred = model.predict(train_loader)
    train_predictions = pd.DataFrame({'train truth': train_truth, 'train pred': train_pred})
    train_predictions['error'] = train_predictions['train pred'] - train_predictions['train truth']
    train_predictions['abs_error'] = train_predictions['error'].abs()
    train_predictions['square_error'] = train_predictions['error'] * train_predictions['error']

    train_predictions.error.plot.hist(bins=30)
    plt.title('Error distribution of total energy (trainset)')
    plt.xlabel('Error from truth')
    plt.xlim((-1, 1))
    plt.show()
    print("-------------------------------------------")
    print(train_predictions.describe())
    print("-------------------------------------------")

    #testset error statistic
    truth, pred = model.predict(test_loader)
    predictions = pd.DataFrame({'truth': truth, 'pred': pred})
    predictions['error'] = predictions['pred'] - predictions['truth']
    predictions['abs_error'] = predictions['error'].abs()
    predictions['square_error'] = predictions['error'] * predictions['error']

    predictions.error.plot.hist(bins=30)
    plt.title('Error distribution of total energy (testset)')
    plt.xlabel('Error from truth')
    plt.xlim((-1, 1))
    plt.show()
    print("-------------------------------------------")
    print(predictions.describe())
    print("-------------------------------------------")

    idx = list(range(len(pred)))
    random.shuffle(idx)
    new_idx = idx[:300]
    random_pred = []
    random_truth = []
    for i in new_idx:
        random_pred.append(pred[i])
        random_truth.append(truth[i])

    fig = plt.figure(figsize=(12, 4))
    plt.plot(random_truth, label='truth')
    plt.plot(random_pred, label='pred')
    plt.legend()
    plt.ylabel('energy (eV) shifted by +527eV')
    plt.xlabel('stucture number')
    plt.show()

    plot_results(pred, truth, title="Volume Element Model")


def pca():
    pca_train = pd.read_csv('dataset/train_pca.csv')
    pca_test = pd.read_csv('dataset/test_pca.csv')
    label_train = pd.read_csv('dataset/train_label.csv')
    label_test = pd.read_csv('dataset/test_label.csv')
    n_col= 12

    ds_train = VEFeaturesEnergyDataset(pca_train, label_train, n_col=n_col)
    ds_train.normalize_y(mode='offset', offset=527)
    ds_train.normalize_X(mode='standard')

    ds_test = VEFeaturesEnergyDataset(pca_test, label_test, n_col=n_col)
    ds_test.normalize_y(mode='offset', offset=527)
    ds_test.normalize_X(mode='standard', pre_mean=ds_train.pca_mean, pre_std=ds_train.pca_std)

    # This shuffle is shuffle each data sample when iterating the whole dataset
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=8, shuffle=False)
    print(f"# Train: {len(train_loader)}, # test: {len(test_loader)}")

    model = ModelContainer(SubNet(n_col, c=8))
    print(model.model)
    model.fit(train_loader, 50)

    # trainset error statistic
    train_truth, train_pred = model.predict(train_loader)
    train_predictions = pd.DataFrame({'train truth': train_truth, 'train pred': train_pred})
    train_predictions['error'] = train_predictions['train pred'] - train_predictions['train truth']
    train_predictions['abs_error'] = train_predictions['error'].abs()
    train_predictions['square_error'] = train_predictions['error'] * train_predictions['error']

    train_predictions.error.plot.hist(bins=30)
    plt.title('Error distribution of total energy (trainset)')
    plt.xlabel('Error from truth')
    plt.xlim((-1, 1))
    plt.show()
    print("-------------------------------------------")
    print(train_predictions.describe())
    print("-------------------------------------------")

    # testset error statistic
    truth, pred = model.predict(test_loader)
    predictions = pd.DataFrame({'truth': truth, 'pred': pred})
    predictions['error'] = predictions['pred'] - predictions['truth']
    predictions['abs_error'] = predictions['error'].abs()
    predictions['square_error'] = predictions['error'] * predictions['error']

    predictions.error.plot.hist(bins=30)
    plt.title('Error distribution of total energy (testset)')
    plt.xlabel('Error from truth')
    plt.xlim((-1, 1))
    plt.show()
    print("-------------------------------------------")
    print(predictions.describe())
    print("-------------------------------------------")

    idx = list(range(len(pred)))
    random.shuffle(idx)
    new_idx = idx[:300]
    random_pred = []
    random_truth = []
    for i in new_idx:
        random_pred.append(pred[i])
        random_truth.append(truth[i])

    fig = plt.figure(figsize=(12, 4))
    plt.plot(random_truth, label='truth')
    plt.plot(random_pred, label='pred')
    plt.legend()
    plt.ylabel('energy (eV) shifted by +527eV')
    plt.xlabel('stucture number')
    plt.show()

    plot_results(pred,truth, title="PCA model")


if __name__ == '__main__':

    # potential only prediction
    # areaElement()

    # potential & force prediction
    areaELement_stress()
