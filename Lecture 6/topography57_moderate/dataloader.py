import torch
from torch.utils import data
import os
import numpy as np
from scipy.io import loadmat

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, dir, seq_length):
        'Initialization'
        self.root = dir
        self.seq_length = seq_length
        self.u = np.load(os.path.join(dir, 'u.npy'))
        self.F = np.load(os.path.join(dir, 'f.npy'))
        self.list_IDs = os.listdir(dir)
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # select sample
        ID = self.list_IDs[index]

        # load data and get label
        half_seq_length = self.u.shape[0]//2 - 1
        idx_randint = torch.randint(low = 0, high = half_seq_length-self.seq_length, size = (1,))
        u_t = self.u[idx_randint:idx_randint + self.seq_length]
        F_t = self.F[idx_randint:idx_randint + self.seq_length]
        F_tt = self.F[idx_randint + 1:idx_randint + 1 + self.seq_length]
        return u_t, F_t, F_tt # index, idx_randint