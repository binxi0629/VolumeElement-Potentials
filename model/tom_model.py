import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

'''
Subnet architecture

c (int): base number of units, scaled by different integers in hidden layers
'''


class SubNet(nn.Module):
    def __init__(self, input_size, c=3, stress_fit=False):
        super().__init__()
        self.FC1 = nn.Linear(input_size, c).double()
        self.FC2 = nn.Linear(c, 2 * c).double()
        self.FC3 = nn.Linear(2 * c, c).double()
        # self.FC5 = nn.Linear(c, c).double()
        # self.FC6 = nn.Linear(c, c).double()

        self.FC4 = nn.Linear(c, 3).double() if stress_fit else nn.Linear(c, 1).double()
        self.leaky_relu = nn.LeakyReLU()
        # self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.FC1(x)
        # x = self.tanh(x)
        x = self.leaky_relu(x)

        x = self.FC2(x)
        # x = self.tanh(x)
        x = self.leaky_relu(x)

        x = self.FC3(x)
        # x = self.tanh(x)
        x = self.leaky_relu(x)

        # x = self.FC5(x)
        # x = self.leaky_relu(x)
        # x = self.FC6(x)
        # x = self.leaky_relu(x)

        x = self.FC4(x)

        return x


'''
Temperory class, for fitting and predicting functions

model (Pytorch model object): your initialized model

The training params and setting are fixed in the function for now...
'''


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()


class ModelContainer():
    def __init__(self, model, stress_fit=False):
        self.model = model
        self.stress_fit = stress_fit
        self.criterion = torch.nn.MSELoss()

    def fit(self, loader, ep, learning_rate=1e-3):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.model.train()

        if self.stress_fit:
            for epoch in range(ep):
                f_loss_sum, e_loss_sum, loss_sum = 0, 0, 0
                f_loss = 0
                count = 0
                for X, forces, energy in loader:
                    # print(X.shape, forces.shape, energy.shape)
                    energy = energy.float()
                    self.optimizer.zero_grad()
                    if np.nan in energy: continue
                    tot_e = torch.zeros(energy.size())

                    for i in range(len(X[0])):
                        res = self.model(X[:, i, :])
                        tot_e += res[:, 2]
                        # count+=1
                        # print(X[:,i,:])
                        # print(count)
                        # print("energy:", res)
                        # out_forces = res[:, 0:2]
                        # print("------------", out_forces, forces[:,i,:].squeeze())

                        # loss = torch.nn.functional.mse_loss(out_forces, forces[:, i, :].squeeze())
                        # f_loss_sum += loss.item()
                        # if i != len(X[0])-1:
                        #     loss.backward(retain_graph=True)
                        #

                    out_forces = self.model(X)[:, :, 0:2]
                    # print(self.model(X).shape)
                    # print("forces:", out_forces)

                    f_loss = torch.nn.functional.mse_loss(out_forces, forces)
                    e_loss = self.criterion(tot_e.squeeze(), energy)
                    loss = f_loss + e_loss
                    # loss += e_loss
                    loss.backward()

                    # if count >=480:
                    #     plot_grad_flow(self.model.named_parameters())

                    f_loss_sum += f_loss.item()
                    e_loss_sum += e_loss.item()
                    loss_sum += loss.item()
                    # nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.optimizer.step()
                self.scheduler.step()
                print("Epoch: ", epoch, "Energy Loss (MSE): ", np.round(e_loss_sum / len(loader), 6),
                      "Forces Loss (MSE): ", np.round(f_loss_sum / len(loader), 6))
        else:
            for epoch in range(ep):
                loss_sum = 0
                for X, y in loader:
                    y = y.float()
                    self.optimizer.zero_grad()
                    if np.nan in y: continue
                    tot_e = torch.zeros(y.size()).unsqueeze(1)
                    for i in range(len(X[0])):
                        tot_e += self.model(X[:, i, :])
                        # print("y_pred, y_0", tot_e)

                    loss = self.criterion(tot_e.squeeze(), y)
                    loss.backward()

                    loss_sum += loss.item()
                    # nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.optimizer.step()
                self.scheduler.step()
                print("Epoch: ", epoch, "Loss (MSE): ", loss_sum / len(loader))
        return

    def predict(self, loader, save_sub_preds=False):
        truth, pred, sub_pred = [], [], []
        self.model.eval()
        loss_sum = 0
        for X, y in loader:
            y = y.float()
            if np.nan in y: continue
            tot_e = torch.zeros(y.size()).unsqueeze(1)
            with torch.no_grad():
                if save_sub_preds: batch_sub_pred = []
                for i in range(len(X[0])):
                    sub_e = self.model(X[:, i, :])
                    tot_e += sub_e
                    if save_sub_preds: batch_sub_pred.append(sub_e.squeeze().tolist())
                if save_sub_preds: sub_pred.extend(np.array(batch_sub_pred).T.flatten())
                loss = self.criterion(tot_e.squeeze(), y)
                truth.extend(y.tolist())
                pred.extend(tot_e.squeeze().tolist())
                # print("Truth: ", list(y))
                # print("Prediction: ", list(tot_e.squeeze()))
            loss_sum += loss.item()
        print("Test MSELoss", loss_sum / len(loader))
        if save_sub_preds:
            return truth, pred, sub_pred
        return truth, pred


class Spring(torch.nn.Module):
    def __init__(self, single_weight=True):
        super().__init__()
        self.single_weight = single_weight
        self.k = torch.nn.Parameter(torch.rand(()))
        if self.single_weight:
            self.w1 = torch.nn.Parameter(torch.rand(()))
            self.w2 = torch.nn.Parameter(torch.rand(()))
            self.w3 = torch.nn.Parameter(torch.rand(()))
            # FIXME: Try all vertices
            self.w4 = torch.nn.Parameter(torch.rand(()))
            self.w5 = torch.nn.Parameter(torch.rand(()))
            self.w6 = torch.nn.Parameter(torch.rand(()))
        else:
            self.w1 = torch.nn.Parameter(torch.rand((1, 2)))
            self.w2 = torch.nn.Parameter(torch.rand((1, 2)))
            self.w3 = torch.nn.Parameter(torch.rand((1, 2)))
            # FIXME: Try all vertices
            self.w4 = torch.nn.Parameter(torch.rand((1, 2)))
            self.w5 = torch.nn.Parameter(torch.rand((1, 2)))
            self.w6 = torch.nn.Parameter(torch.rand((1, 2)))

    def forward(self, x):
        # return Force Pred, Elastic Energy Pred
        # force = self.k * (self.w1 * x[:, :2] + self.w2 * x[:, 2:4] + self.w3 * x[:, 4:])

        # FIXME: Try all vertices
        force = self.k * (self.w1 * x[:, :2] + self.w2 * x[:, 2:4] + self.w3 * x[:, 4:6]
                          + self.w4 * x[:, 6:8] + self.w5 * x[:, 8:10] + self.w6 * x[:, 10:])

        # elastic_energy = 1 / 2 * self.k * (
        #             torch.sum(self.w1 * x[:, :2] ** 2, dim=1) + torch.sum(self.w2 * x[:, 2:4] ** 2, dim=1) + torch.sum(
        #         self.w3 * x[:, 4:] ** 2, dim=1))

        elastic_energy = 1 / 2 * self.k * (
                torch.sum(self.w1 * x[:, :2] ** 2, dim=1) + torch.sum(self.w2 * x[:, 2:4] ** 2, dim=1) + torch.sum(
            self.w3 * x[:, 4:6] ** 2, dim=1) + torch.sum(self.w4 * x[:, 6:8] ** 2, dim=1) +
                torch.sum(self.w5 * x[:, 8:10] ** 2, dim=1) + torch.sum(self.w6 * x[:, 10:] ** 2, dim=1))

        return force, elastic_energy.reshape((-1, 1))

    def weights_norm_sum(self):
        # wsum = self.w1 + self.w2 + self.w3
        # FIXME: Try all vertices
        wsum = self.w1 + self.w2 + self.w3 + self.w4 + self.w5 + self.w6
        return wsum


def Force_ElasticEnergy_WeightsNormalization_Loss(force_pred, force, elastic_pred, elastic, w_sum, single_weight=True):
    force_loss = nn.functional.mse_loss(force_pred, force)
    elastic_e_loss = nn.functional.mse_loss(elastic_pred, elastic)
    if single_weight:
        w_sum_loss = nn.functional.mse_loss(w_sum, torch.ones(()))
    else:
        w_sum_loss = nn.functional.mse_loss(w_sum, torch.ones((1, 2)))
    return force_loss, elastic_e_loss * 100, 100 * w_sum_loss


class SpringModelContainer():
    def __init__(self, model):
        self.model = model
