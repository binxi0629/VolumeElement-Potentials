from area_element.area2 import AreaElementDivider
from model.datasets.tom_dataset import VEStressDataset
from model.tom_model import Spring, Force_ElasticEnergy_WeightsNormalization_Loss, SubNet

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import pandas as pd
import torch
torch.set_default_tensor_type(torch.FloatTensor)
import matplotlib.pyplot as plt


def trainloop0(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    model.train()
    for epoch in range(30):
        floss_sum, eloss_sum = 0, 0
        for xs, force, elastic in train_loader:
            optimizer.zero_grad()
            out_res = model(xs)

            if np.nan in force: continue
            if np.nan in elastic: continue
            out_force, out_elastic = out_res[:, 0:2], out_res[:, 2]
            # print(f"Pred Force: {out_force} | Ground Truth force: {force} | Pred Energy: {out_elastic}|  Ground Truth energy {elastic.squeeze()} ")
            floss = torch.nn.functional.mse_loss(out_force, force)
            eloss = torch.nn.functional.mse_loss(out_elastic, elastic.squeeze())

            loss = floss + eloss
            loss.backward()
            floss_sum += floss.item()
            eloss_sum += eloss.item()
            optimizer.step()
        scheduler.step()
        print("Epoch: ", epoch, "Force MSE", np.round(floss_sum / len(train_loader), 6), "Elastic Energy MSE", np.round(eloss_sum / len(train_loader), 6))


def trainloop(model, single_weight=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    model.train()
    for epoch in range(20):
        floss_sum, eloss_sum, wloss_sum = 0,0,0
        for xs, force, elastic in train_loader:
            optimizer.zero_grad()
            out_force, out_elastic = model(xs)
            w_sum = model.weights_norm_sum()
            floss, eloss, wloss = Force_ElasticEnergy_WeightsNormalization_Loss(out_force, force, out_elastic, elastic, w_sum, single_weight=single_weight)
            loss = floss + eloss + wloss
            loss.backward()
            floss_sum += floss.item()
            eloss_sum += eloss.item()
            wloss_sum += wloss.item()
            optimizer.step()
        scheduler.step()
        print("Epoch: ", epoch, "Force MSE", np.round(floss_sum / len(train_loader), 3), "Elastic Energy MSE", np.round(eloss_sum / len(train_loader), 3), "Weight Normalize MSE", np.round(wloss_sum / len(train_loader), 3))


def predict2(model, loader, num_ve=30):
    model.model.eval()
    forces, elastics = [], []
    pred_force, pred_elastic = [], []

    for xs, force, elastic in loader:
        xs, force, elastic = torch.tensor(xs), torch.tensor(force), torch.tensor(elastic)
        # torch.Size([2, 30, 12]) torch.Size([2, 30, 2]) torch.Size([2])
        energy = torch.zeros(elastic.size())
        for i in range(len(xs[0])):
            out_res = model.model(xs[:, i, :])
            out_force, out_elastic = out_res[:, 0:2], out_res[:, 2]
            forces.append(force[:, i, :].detach().numpy())
            pred_force.append(out_force.detach().numpy())

            energy += out_elastic.squeeze().detach().numpy()
        pred_elastic.append(np.array(energy))
        elastics.append(elastic.detach().numpy())

    # (890, 30, 2) (890, 2, 3) (32,) (32, 3)

    return np.concatenate(forces, axis=0), np.concatenate(pred_force, axis=0), np.concatenate(elastics,
                                                                                              axis=0), np.concatenate(
        pred_elastic, axis=0)

def predict(model, loader):
    model.eval()
    forces, elastics = [], []
    pred_force, pred_elastic = [], []
    for xs, force, elastic in loader:
        xs, force, elastic = torch.tensor(xs), torch.tensor(force), torch.tensor(elastic)
        # print(xs, force, elastic)
        out_res = model(xs)
        out_force, out_elastic = out_res[:, 0:2], out_res[:, 2]

        forces.append(force.detach().numpy())
        pred_force.append(out_force.detach().numpy())

        elastics.append(elastic.detach().numpy())
        pred_elastic.append(out_elastic.detach().numpy())
    return np.concatenate(forces, axis=0), np.concatenate(pred_force, axis=0), np.concatenate(elastics,
                                                                                              axis=0), np.concatenate(
        pred_elastic, axis=0)


def rmse(pred, truth):
    return np.sqrt(((pred - truth)**2).mean())
def mape(pred, truth):
    return ((pred - truth)/truth).abs().mean()


def process_df2(res, res2):
    res['pred_total_force'] = np.sqrt(res['pred_force_x']**2 + res['pred_force_y']**2)
    res['total_force'] = np.sqrt(res['force_x']**2 + res['force_y']**2)

    res['sqerr_force_x'] = (res['pred_force_x'] - res['force_x'])**2
    res['sqerr_force_y'] = (res['pred_force_y'] - res['force_y'])**2
    res['sqerr_force'] = (res['pred_total_force'] - res['total_force'])**2
    res2['sqerr_energy'] = (res2['pred_energy'] - res2['energy'])**2

    res['ape_force_x'] = ((res['pred_force_x'] - res['force_x']) / res['force_x']).abs()
    res['ape_force_y'] = ((res['pred_force_y'] - res['force_y']) / res['force_y']).abs()
    res['ape_force'] = ((res['pred_total_force'] - res['total_force']) / res['total_force']).abs()
    res2['ape_energy'] = ((res2['pred_energy'] - res2['energy']) / res2['energy']).abs()
    return res, res2

def process_df(res):
    res['pred_total_force'] = np.sqrt(res['pred_force_x']**2 + res['pred_force_y']**2)
    res['total_force'] = np.sqrt(res['force_x']**2 + res['force_y']**2)

    res['sqerr_force_x'] = (res['pred_force_x'] - res['force_x'])**2
    res['sqerr_force_y'] = (res['pred_force_y'] - res['force_y'])**2
    res['sqerr_force'] = (res['pred_total_force'] - res['total_force'])**2
    res['sqerr_elastic'] = (res['pred_elastic'] - res['elastic'])**2

    res['ape_force_x'] = ((res['pred_force_x'] - res['force_x']) / res['force_x']).abs()
    res['ape_force_y'] = ((res['pred_force_y'] - res['force_y']) / res['force_y']).abs()
    res['ape_force'] = ((res['pred_total_force'] - res['total_force']) / res['total_force']).abs()
    res['ape_elastic'] = ((res['pred_energy'] - res['energy']) / res['elastic']).abs()
    return res

def plot_df(res, title='single weight per spring', seed=42):
    fig, axs = plt.subplots(2,4, figsize=(16,8))
    # tmp = res.loc[res.force_x.abs() < 1, ['pred_force_x', 'force_x']]
    tmp = res[['pred_force_x', 'force_x']]
    tmp.plot(x='pred_force_x', y='force_x', kind='scatter', ax=axs[0][0], title='x_component force (eV/Angstrom)', alpha=0.1)
    axs[0][0].set_xlim((-5, 5))
    axs[0][0].set_ylim((-5, 5))
    axs[0][0].plot([-5,5], [-5,5], c='r')
    tmp.sample(50, random_state=seed).reset_index(drop=True).plot(ax=axs[0][1], title='x_component force samples')
    # tmp = res.loc[res.force_y.abs() < 1, ['pred_force_y', 'force_y']]
    tmp = res[['pred_force_y', 'force_y']]
    tmp.plot(x='pred_force_y', y='force_y', kind='scatter', ax=axs[0][2], title='y_component force (eV/Angstrom)', alpha=0.1)
    axs[0][2].set_xlim((-5, 5))
    axs[0][2].set_ylim((-5, 5))
    axs[0][2].plot([-5,5], [-5,5], c='r')
    tmp.sample(50, random_state=seed).reset_index(drop=True).plot(ax=axs[0][3], title='y_component force samples')

    tmp = res[['pred_total_force', 'total_force']]
    tmp.plot(x='pred_total_force', y='total_force', kind='scatter', ax=axs[1][0], title='force magnitude (eV/Angstrom)', alpha=0.1)
    axs[1][0].set_xlim((-1, 5))
    axs[1][0].set_ylim((-1, 5))
    axs[1][0].plot([-1,5], [-1,5], c='r')
    tmp.sample(50, random_state=seed).reset_index(drop=True).plot(ax=axs[1][1], title='force magnitude samples')

    tmp = res[['pred_elastic', 'elastic']]
    tmp.plot(x='pred_elastic', y='elastic', kind='scatter', ax=axs[1][2], title='elastic energy (eV)', alpha=0.1)
    axs[1][2].set_xlim((-0.1, 0.25))
    axs[1][2].set_ylim((-0.1, 0.25))
    axs[1][2].plot([-0.1,0.25], [-0.1,0.25], c='r')
    tmp.sample(50, random_state=seed).reset_index(drop=True).plot(ax=axs[1][3], title='elastic energy samples')
    plt.suptitle(title)
    plt.tight_layout()
    return

def plot_df2(res, res2, title='single weight per spring', seed=42):
    fig, axs = plt.subplots(2,4, figsize=(16,8))
    # tmp = res.loc[res.force_x.abs() < 1, ['pred_force_x', 'force_x']]
    tmp = res[['pred_force_x', 'force_x']]
    tmp.plot(x='pred_force_x', y='force_x', kind='scatter', ax=axs[0][0], title='x_component force (eV/Angstrom)', alpha=0.1)
    axs[0][0].set_xlim((-5, 5))
    axs[0][0].set_ylim((-5, 5))
    axs[0][0].plot([-5,5], [-5,5], c='r')

    axs[0][1].set_ylim((-4, 4))
    axs[0][3].set_ylim((-4, 4))
    axs[1][1].set_ylim((-4, 4))
    axs[1][3].set_ylim((-1.5, 1.5))

    tmp.sample(50, random_state=seed).reset_index(drop=True).plot(ax=axs[0][1], title='x_component force samples')
    # tmp = res.loc[res.force_y.abs() < 1, ['pred_force_y', 'force_y']]
    tmp = res[['pred_force_y', 'force_y']]
    tmp.plot(x='pred_force_y', y='force_y', kind='scatter', ax=axs[0][2], title='y_component force (eV/Angstrom)', alpha=0.1)
    axs[0][2].set_xlim((-5, 5))
    axs[0][2].set_ylim((-5, 5))
    axs[0][2].plot([-5,5], [-5,5], c='r')
    tmp.sample(50, random_state=seed).reset_index(drop=True).plot(ax=axs[0][3], title='y_component force samples')

    tmp = res[['pred_total_force', 'total_force']]
    tmp.plot(x='pred_total_force', y='total_force', kind='scatter', ax=axs[1][0], title='force magnitude (eV/Angstrom)', alpha=0.1)
    axs[1][0].set_xlim((-1, 5))
    axs[1][0].set_ylim((-1, 5))
    axs[1][0].plot([-1,5], [-1,5], c='r')
    tmp.sample(50, random_state=seed).reset_index(drop=True).plot(ax=axs[1][1], title='force magnitude samples')

    tmp = res2[['pred_energy', 'energy']]
    tmp.plot(x='pred_energy', y='energy', kind='scatter', ax=axs[1][2], title='energy (eV)', alpha=0.1)
    axs[1][2].set_xlim((-2, 2))
    axs[1][2].set_ylim((-2, 2))
    axs[1][2].plot([-2,2], [-2,2], c='r')
    tmp.sample(50, random_state=seed).reset_index(drop=True).plot(ax=axs[1][3], title=' energy samples')
    plt.suptitle(title)
    plt.tight_layout()
    return


if __name__ == '__main__':
    ref_fname = './area_element/test_poscar/supercell35_60.vasp'
    unpert_en = -528.02684
    n_center = 30
    center_idx = np.arange(n_center)  # List of lines you want to use as center
    AE = AreaElementDivider(center_idx)
    AE.fetch_structure(ref_fname)
    AE.all_elements_vertices_nnsearch(return_center=True, cyclic_sort=True, return_frac=False, shift_to_center=True)
    AE.set_reference(AE.all_nn_avg_idx, AE.all_nn_avg_img)
    ref_ele = AE.all_ele[:, :, :2].mean(axis=0)

    # ref_x = ref_ele[[2,4,6]].flatten()
    # FIXME: try all vertices
    ref_x = ref_ele[[1, 2, 3, 4, 5, 6]].flatten()

    ref_sub_e = (unpert_en + 527) / 30

    traindf = pd.read_csv("dataset/train_ve_force_sube.csv")
    testdf = pd.read_csv("dataset/test_ve_force_sube.csv")

    trainset = VEStressDataset(traindf, ref_x, ref_sub_e)
    testset = VEStressDataset(testdf, ref_x, ref_sub_e)
    trainset.normalize_X(mode='standard')
    testset.normalize_X(mode='standard', pre_mean=trainset.xs_mean, pre_std=trainset.xs_std)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

    model0 = SubNet(12, c=8, stress_fit=True)
    model1 = Spring(single_weight=True)
    model2 = Spring(single_weight=False)

    for xs, force, elastic in train_loader:
        xs = torch.tensor(xs, dtype=torch.float)
        force, elastic = torch.tensor(force, dtype=torch.float), torch.tensor(elastic, dtype=torch.float)
        print(xs[:, :2])
        out_force, out_elastic = model2(xs)
        w_sum = model2.weights_norm_sum()
        print(out_force, force)
        print(model2.w1)
        print(out_elastic, elastic)
        print(w_sum)
        print(Force_ElasticEnergy_WeightsNormalization_Loss(out_force, force, out_elastic, elastic, w_sum,
                                                            single_weight=False))
        break

    trainloop0(model0)
    # trainloop(model1)
    # print(
    #       model1.weights_norm_sum().item(),
    #       model1.k.item(),
    #       model1.w1.item(),
    #       model1.w2.item(),
    #       model1.w3.item()
    #       )

    # trainloop(model2, single_weight=False)
    # print(
    #     model2.weights_norm_sum().detach().numpy(),
    #     model2.k.item(),
    #     model2.w1.detach().numpy(),
    #     model2.w2.detach().numpy(),
    #     model2.w3.detach().numpy()
    # )

    forces1, pred_force1, elastics1, pred_elastic1 = predict(model0, test_loader)
    # forces1, pred_force1, elastics1, pred_elastic1 = predict(model1, test_loader)
    # forces2, pred_force2, elastics2, pred_elastic2 = predict(model2, test_loader)

    res = pd.DataFrame({'pred_force_x': pred_force1[:, 0].flatten(), 'pred_force_y': pred_force1[:, 1].flatten(),
                        'force_x': forces1[:, 0].flatten(), 'force_y': forces1[:, 1].flatten(),
                        'pred_elastic': pred_elastic1.flatten(), 'elastic': elastics1.flatten()})
    res = process_df(res)
    print(res.describe())
    plot_df(res)
    plt.show()

    outlier_thres = res.mean() + 3 * res.std()
    for col, thres in zip(outlier_thres.index, outlier_thres):
        print(col, thres, (res[col] > thres).sum(), (res[col] > thres).sum() / len(res))
        if 'ape' in col:
            tmp = res.loc[res[col] < thres, col]
            print(tmp.mean())

    # res = pd.DataFrame({'pred_force_x': pred_force2[:, 0].flatten(), 'pred_force_y': pred_force2[:, 1].flatten(),
    #                     'force_x': forces2[:, 0].flatten(), 'force_y': forces2[:, 1].flatten(),
    #                     'pred_elastic': pred_elastic2.flatten(), 'elastic': elastics2.flatten()})
    # res = process_df(res)
    # print(res.abs().describe())
    # plot_df(res, title='two weight per spring')