"""
@author: Yong Zheng Ong
loads the dataset
"""
from numpy.lib.npyio import load
import torch
import numpy as np
import scipy.io
import scipy.interpolate as interpolate
import h5py

from .utilities import UnitGaussianNormalizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DataEncapsulator():
    """
    a class to manage dataset loading
    """
    def __init__(self,
                 file_path=None,
                 to_torch=True,
                 to_cuda=False,
                 to_float=True,
                 **_
        ):
        # set default filepath to fno darcy data
        if file_path is None:
            self.file_path = "datasets/darcy/data/darcy_data_a2_t3.mat"

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        print("read field: converting to float...")
        if self.to_float:
            x = x.astype(np.float32)

        print("read field: converting to tensor...")
        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                print("read field: converting to device...")
                x = x.to(device)

        return x

    def build_grid_data(self, data_x, data_y, index, cuda=True):
        batch_size = data_x.size()[0]
        batch_x = data_x[:,::self.available_subs[index],::self.available_subs[index]][:,:self.grid_sizes[index],:self.grid_sizes[index]]
        batch_y = data_y[:,::self.available_subs[index],::self.available_subs[index]][:,:self.grid_sizes[index],:self.grid_sizes[index]]

        if cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        batch_x = self.x_normalizer[index].encode(batch_x)
        batch_y = self.y_normalizer[index].encode(batch_y)

        # build the locations information - here, uniform grid is used
        grids = []
        grids.append(np.linspace(0, 1, self.grid_sizes[index]))
        grids.append(np.linspace(0, 1, self.grid_sizes[index]))
        grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
        grid = grid.reshape(1,self.grid_sizes[index],self.grid_sizes[index],2)
        grid = torch.tensor(grid, dtype=torch.float).to(batch_x.device)

        batch_x = torch.cat([batch_x.reshape(batch_size,self.grid_sizes[index],self.grid_sizes[index],1), grid.repeat(batch_size,1,1,1)], dim=3)

        return batch_x, batch_y

    def build_data(self, sub=0, target_sub=0, ntrain=1000, ntest=100, load_type="default"):
        available_build_types = ["default", "multi", "multi3"]
        if load_type not in available_build_types:
            load_type = "multi"

        self.y_test_mask = None

        # set basic details
        self.ntrain = ntrain
        self.ntest = ntest
        self.ts = 1
        self.num_d = 2 # 2 dimension problem
        self.input_channel = 3 # (a(x, y), x, y)
        self.output_channel = 1 # (u(x, y))
        self.load_type = load_type
        # self.available_subs = [1, 2, 3, 4, 5]
        self.available_subs = [2, 3, 4, 5]
        # build grid sizes
        self.grid_sizes = []
        for i in self.available_subs:
            self.grid_sizes.append(int(((421 - 1)/i) + 1))
        print("grid sizes: {}".format(self.grid_sizes))

        # set data original size
        self.sub = sub #subsampling rate
        self.s = int(((421 - 1)/self.sub) + 1) #total grid size divided by the subsampling rate

        # set target data size
        self.target_sub = target_sub
        self.target_s = int(((421 - 1)/self.target_sub) + 1)

        # set max data size
        self.max_sub = 1 #subsampling rate
        self.max_s = int(((421 - 1)/self.max_sub) + 1) #total grid size divided by the subsampling rate

        # get the data
        x_data = self.read_field('coeff')[:,::self.available_subs[self.ts],::self.available_subs[self.ts]][:,:self.grid_sizes[self.ts],:self.grid_sizes[self.ts]]
        y_data = self.read_field('sol')[:,::self.available_subs[self.ts],::self.available_subs[self.ts]][:,:self.grid_sizes[self.ts],:self.grid_sizes[self.ts]]
        data_length = x_data.shape[0]
        x_data = x_data.reshape(data_length, 1, self.grid_sizes[self.ts], self.grid_sizes[self.ts])
        y_data = y_data.reshape(data_length, 1, self.grid_sizes[self.ts], self.grid_sizes[self.ts])
        x_data = torch.nn.functional.interpolate(x_data, size=(self.max_s,self.max_s), mode='bicubic').reshape(data_length, self.max_s, self.max_s)
        y_data = torch.nn.functional.interpolate(y_data, size=(self.max_s,self.max_s), mode='bicubic').reshape(data_length, self.max_s, self.max_s)

        # build the max size data
        self.x_train = x_data[:self.ntrain,:,:]
        self.y_train = y_data[:self.ntrain,:,:]

        self.x_test = x_data[-self.ntest:,:,:]
        self.y_test = y_data[-self.ntest:,:,:]

        print("train subsampling value: {}, dataset size: {}".format(self.sub, self.s))
        print("test subsampling value: {}, dataset size: {}".format(self.target_sub, self.target_s))

        # create data normalizers
        self.x_normalizer = []
        self.y_normalizer = []
        for i in range(len(self.available_subs)):
            self.x_normalizer.append(UnitGaussianNormalizer(self.x_train[:,::self.available_subs[i],::self.available_subs[i]][:,:self.grid_sizes[i],:self.grid_sizes[i]]))
            self.y_normalizer.append(UnitGaussianNormalizer(self.y_train[:,::self.available_subs[i],::self.available_subs[i]][:,:self.grid_sizes[i],:self.grid_sizes[i]]))

        if self.load_type == "default":
            print("building data for case = 1: x_train is of target grid size, x_test is of all the grid sizes")

            # build the training data
            self.x_train, self.y_train = self.build_grid_data(self.x_train, self.y_train, self.ts, cuda=False)

            print("training shape: x - {}, y - {}".format(self.x_train[0].numpy().shape, self.y_train[0].numpy().shape))
            print("testing shape: x - {}, y - {}".format(self.x_test[0].numpy().shape, self.y_test[0].numpy().shape))

        if self.load_type == "multi":
            print("building data for case = 2: x_train is of all the grid sizes, x_test is of all the grid sizes")

            print("training shape: x - {}, y - {}".format(self.x_train[0].numpy().shape, self.y_train[0].numpy().shape))
            print("testing shape: x - {}, y - {}".format(self.x_test[0].numpy().shape, self.y_test[0].numpy().shape))
        
        if self.load_type == "multi3":
            raise NotImplementedError("multi3 is not implemented for darcy problem")

        for normalizer in self.y_normalizer:
            normalizer.cuda()
        for normalizer in self.x_normalizer:
            normalizer.cuda()
