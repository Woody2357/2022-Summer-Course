"""
@author: Yong Zheng Ong
Launch the training of the model

# how to use:
python train.py MODELNAME DATASETNAME LOADTYPE
"""
import os
import sys

# # choose CUDA environments
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="4"

import argparse

from libs.utils.loader import loaddataset, loadtrainer

if __name__ == "__main__":
    # implement argparse
    parser = argparse.ArgumentParser(description='process input parameters')

    parser.add_argument('model_name', type=str)
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('load_type', type=str)

    args = parser.parse_args()

    # check values
    assert args.load_type in ["default", "multi", "multi3"], "{} not found in valid load types, {}".format(args.load_type, [])

    # some constants (will be replaced further down)
    datasetConfig = { # contains dataset configs
        "sub": 3,
        "target_sub": 3,
        "ntrain": 1000,
        "ntest": 100,
        "load_type": args.load_type
    }
    modelConfig = { # contains model configs
        "modes": 128,
        "width": 64
    }

    # configurations list
    if args.model_name in ["iae_net"]:
        if args.dataset_name in ["burgers"]:

            datasetConfig["sub"] = 3
            datasetConfig["target_sub"] = 3

            # add additional dataset configs
            if args.dataset_name in ["burgers"]:
                datasetConfig["ntrain"] = 10000
                datasetConfig["ntest"] = 1000

            # add additional model configs
            modelConfig["modes"] = 256
            modelConfig["width"] = 64

            # prepare training configs
            batch_size = 50
            learning_rate = 0.001
            epochs = 500
            gamma = 0.5

        if args.dataset_name in ["fecgsyndb"]:

            datasetConfig["sub"] = 1
            datasetConfig["target_sub"] = 1

            # add additional dataset configs
            datasetConfig["ntrain"] = 25000
            datasetConfig["ntest"] = 6150

            # add additional model configs
            modelConfig["modes"] = 256
            modelConfig["width"] = 128

            # prepare training configs
            batch_size = 50
            learning_rate = 0.001
            epochs = 500
            gamma = 0.5

        if args.dataset_name in ["darcy"]:

            datasetConfig["sub"] = 3
            datasetConfig["target_sub"] = 3

            # add additional dataset configs
            datasetConfig["ntrain"] = 1000
            datasetConfig["ntest"] = 100
            # add additional model configs
            modelConfig["modes"] = 64
            modelConfig["width"] = 64

            # use their defaults
            batch_size = 10
            learning_rate = 0.001
            epochs = 500
            gamma = 0.5

        if args.dataset_name in ["scattering", "inverse_scattering", "ellipses"]:

            datasetConfig["sub"] = 1
            datasetConfig["target_sub"] = 1

            # add additional dataset configs
            # try with smaller dataset
            datasetConfig["ntrain"] = 10000
            datasetConfig["ntest"] = 1000
            # add additional model configs
            modelConfig["modes"] = 64
            modelConfig["width"] = 64

            # prepare training configs
            batch_size = 5
            learning_rate = 0.001
            epochs = 500
            gamma = 0.5

    print(datasetConfig)
    print(modelConfig)

    # load dataset
    dataset = loaddataset(args.dataset_name)()
    dataset.build_data(**datasetConfig)

    # load model
    model = loadtrainer(args.model_name)(modelConfig, dataset, prefix=args.model_name+"_"+args.dataset_name+"_"+args.load_type+"")
    print("number of model parameters: ", model.model.count_params())

    model.train(epochs, batch_size, learning_rate, gamma)
