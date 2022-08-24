"""
@author: Yong Zheng Ong
implements the training procedure
"""

from timeit import default_timer

import os
from matplotlib.pyplot import ylim
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np

from .model import Model
from ...loss.loss import LpLoss

class Trainer():
    def __init__(self, modelConfig, dataset, prefix="default"):
        """
        the trainer object for performing training, testing, etc
        """
        # save dataset
        self.dataset = dataset
        self.prefix = prefix
        self.save_folder = "./results/iae_net_{}.pt".format(self.prefix)

        # add additional dataset related details to modelConfig
        modelConfig["num_d"] = self.dataset.num_d
        modelConfig["input_channel"] = self.dataset.input_channel
        modelConfig["output_channel"] = self.dataset.output_channel
        modelConfig["size"] = self.dataset.s
        print(modelConfig)

        # instantiate model
        self.model = Model(**modelConfig).cuda()
        print("model will be saved to.. {}".format(self.save_folder))

    def train(self, epochs, batch_size, learning_rate, gamma):
        """
        perform training of the model for n epochs
        """
        available_subs = self.dataset.available_subs

        best_results = {
            'epoch': 0,
            'train': 0,
            'test': 100000000
        }

        if self.dataset.load_type in ["default", "multi"]:
            if self.dataset.y_train is None: # already packaged into dataset class
                train_loader = torch.utils.data.DataLoader(self.dataset.x_train, batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(self.dataset.x_test, batch_size=batch_size, shuffle=False)
            else:
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.dataset.x_train, self.dataset.y_train), batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.dataset.x_test, self.dataset.y_test), batch_size=batch_size, shuffle=False)
        elif self.dataset.load_type in ["multi3"]:
            train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*self.dataset.x_train, *self.dataset.y_train), batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*self.dataset.x_test, *self.dataset.y_test), batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=20)

        myloss = LpLoss(size_average=False)

        for ep in range(epochs):
            self.model.train()
            t1 = default_timer()
            train_l2 = 0.0

            if self.dataset.load_type in ["default", "multi"]:
                for x, y in train_loader:

                    optimizer.zero_grad()

                    if self.dataset.num_d == 1:
                        if self.dataset.load_type == "default":
                            x, y = x.cuda(), y.cuda()
                            out = self.model(x)

                            l2 = myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))
                            loss = l2
                            loss.backward()

                        elif self.dataset.load_type == "multi":
                            for i in range(len(available_subs)):
                                # subsample the training data
                                x_temp = x[:,::available_subs[i]].cuda()
                                y_temp = y[:,::available_subs[i]].cuda()
                                out = self.model(x_temp)

                                loss = myloss(out.reshape(batch_size, -1), y_temp.reshape(batch_size, -1))
                                loss.backward()

                    if self.dataset.num_d == 2:
                        if self.dataset.load_type == "multi":
                            for i in range(len(available_subs)):
                                # subsample the training data
                                x_temp, y_temp = self.dataset.build_grid_data(x, y, i)
                                x_temp = x_temp.cuda()
                                y_temp = y_temp.cuda()
                                out = self.model(x_temp).squeeze()

                                if self.dataset.num_d == 2 and self.dataset.y_normalizer is not None:
                                    out = self.dataset.y_normalizer[i].decode(out)
                                    y_temp = self.dataset.y_normalizer[i].decode(y_temp)

                                loss = myloss(out.reshape(batch_size, -1), y_temp.reshape(batch_size, -1))
                                loss.backward()

                    optimizer.step()
                    train_l2 += loss.item()

            elif self.dataset.load_type in ["multi3"]:
                for data in train_loader:

                    optimizer.zero_grad()

                    loss = 0.0
                    for i in range(len(available_subs)):
                        x = data[i].cuda()
                        y = data[len(available_subs)+i].cuda()
                        out = self.model(x)

                        if self.dataset.num_d == 2 and self.dataset.y_normalizer is not None:
                            out = self.dataset.y_normalizer[i].decode(out)
                            y = self.dataset.y_normalizer[i].decode(y)

                        loss += myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))

                    loss.backward()
                    optimizer.step()
                    train_l2 += loss.item()

            train_l2 /= self.dataset.ntrain

            # test the model
            self.model.eval()
            test_l2 = np.array([0.0] * len(available_subs))
            with torch.no_grad():
                if self.dataset.load_type in ["default", "multi"]:
                    for x, y in test_loader:

                        x, y = x.cuda(), y.cuda()

                        if self.dataset.num_d == 1:
                            if self.dataset.load_type in ["default", "multi"]:
                                for i in range(len(available_subs)):
                                    # subsample the training data
                                    x_temp = x[:,::available_subs[i]]
                                    y_temp = y[:,::available_subs[i]]
                                    out = self.model(x_temp)

                                    test_l2[i] += myloss(out.reshape(batch_size, -1), y_temp.reshape(batch_size, -1)).item()

                        if self.dataset.num_d == 2:
                            if self.dataset.load_type in ["default", "multi"]:
                                for i in range(len(available_subs)):
                                    # subsample the training data
                                    x_temp, y_temp = self.dataset.build_grid_data(x, y, i)
                                    out = self.model(x_temp).squeeze()

                                    if self.dataset.num_d == 2 and self.dataset.y_normalizer is not None:
                                        out = self.dataset.y_normalizer[i].decode(out)
                                        y_temp = self.dataset.y_normalizer[i].decode(y_temp)

                                    test_l2[i] += myloss(out.reshape(batch_size, -1), y_temp.reshape(batch_size, -1)).item()

                elif self.dataset.load_type in ["multi3"]:
                    for data in test_loader:
                        for i in range(len(available_subs)):
                            x = data[i].cuda()
                            y = data[len(available_subs)+i].cuda()
                            out = self.model(x).squeeze()

                            if self.dataset.num_d == 2 and self.dataset.y_normalizer is not None:
                                out = self.dataset.y_normalizer[i].decode(out)
                                y = self.dataset.y_normalizer[i].decode(y)

                            test_l2[i] += myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1)).item()

            test_l2 /= self.dataset.ntest
            scheduler.step(test_l2[self.dataset.ts])

            t2 = default_timer()

            # if results improved, update best results
            if self.dataset.num_d == 1 and test_l2[self.dataset.ts] <= best_results['test']:
                print("saving improved model...")
                best_results['epoch'] = ep
                best_results['train'] = train_l2
                best_results['test'] = test_l2[self.dataset.ts]
                torch.save(self.model, self.save_folder)

            if self.dataset.num_d == 2 and test_l2[self.dataset.ts] <= best_results['test']:
                print("saving improved model...")
                best_results['epoch'] = ep
                best_results['train'] = train_l2
                best_results['test'] = test_l2[self.dataset.ts]
                torch.save(self.model, self.save_folder)

            print(ep, f'{t2-t1:.4e}', f'{train_l2:.4e}', test_l2, scheduler.optimizer.param_groups[0]['lr'])

        print("best results:", best_results["epoch"], best_results["train"], best_results["test"])