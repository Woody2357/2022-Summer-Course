"""
@author: Yong Zheng Ong
loads the dataset
"""
from numpy.lib.npyio import load
import torch
import numpy as np

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
        # set default filepath to fno burgers data
        self.file_path = file_path
        if file_path is None:
            self.file_path = "datasets/ellipses/data/train_data.npz"
            self.file_path_train = "datasets/ellipses/data/train_data.npz"
            self.file_path_test = "datasets/ellipses/data/test_data_ellipses.npz"

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.data = None
        self._load_file()

    def _load_file(self):
        self.data = np.load(self.file_path)
        print(self.data.files)

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

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
        batch_x = data_x[:,::self.available_subs[index],::self.available_subs[index]]
        batch_y = data_y[:,::self.available_subs[index],::self.available_subs[index]]

        if cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        if self.x_normalizer is not None:
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

    def build_data(self, sub=0, target_sub=0, ntrain=10000, ntest=1000, load_type="default"):
        available_build_types = ["default", "multi", "deeponet"]
        if load_type not in available_build_types:
            load_type = "default"

        self.y_test_mask = None

        # set basic details
        self.ntrain = ntrain
        self.ntest = ntest
        self.ts = 1
        self.num_d = 2 # 2 dimension problem
        self.input_channel = 3 # (a(x,y), x, y)
        self.output_channel = 1 # (u(x, y))
        self.load_type = load_type
        self.available_subs = [2**0, 2**1, 2**2, 2**3]
        # build grid sizes
        self.grid_sizes = []
        for i in self.available_subs:
            self.grid_sizes.append(int(256/i))
        print("grid sizes: {}".format(self.grid_sizes))

        # build the max size data
        self.size_max = 128
        self.x_train = self.read_field('arr_0').reshape(self.ntrain,self.size_max,self.size_max)
        y_train = self.read_field('arr_2').reshape(self.ntrain,self.size_max,self.size_max)

        self.load_file(self.file_path_test)
        self.x_test = self.read_field('arr_0').reshape(self.ntest,self.size_max,self.size_max)
        y_test = self.read_field('arr_2').reshape(self.ntest,self.size_max,self.size_max)

        # t_min, t_max = torch.min(x_train), torch.max(x_train)
        # self.x_train = (x_train - t_min)/(t_max - t_min)
        # print("x_train: ", torch.min(self.x_train), torch.max(self.x_train))

        # t_min, t_max = torch.min(x_test), torch.max(x_test)
        # self.x_test = (x_test - t_min)/(t_max - t_min)
        # print("x_test: ", torch.min(self.x_test), torch.max(self.x_test))

        t_min, t_max = torch.min(y_train), torch.max(y_train)
        self.y_train = (y_train - t_min)/(t_max - t_min)
        print("y_train: ", torch.min(self.y_train), torch.max(self.y_train))

        t_min, t_max = torch.min(y_test), torch.max(y_test)
        self.y_test = (y_test - t_min)/(t_max - t_min)
        print("y_test: ", torch.min(self.y_test), torch.max(self.y_test))

        # set data original size
        self.sub = sub #subsampling rate
        self.s = int(256/(2**self.sub)) #total grid size divided by the subsampling rate

        # set target data size
        self.target_sub = target_sub
        self.target_s = int(256/(2**self.target_sub))

        print("train subsampling value: {}, dataset size: {}".format(self.sub, self.s))
        print("test subsampling value: {}, dataset size: {}".format(self.target_sub, self.target_s))

        # create data normalizers
        self.x_normalizer = None
        self.y_normalizer = None

        if self.load_type == "default":
            print("building data for case = 1: x_train is of target grid size, x_test is of all the grid sizes")

            # build the training data
            self.x_train, self.y_train = self.build_grid_data(self.x_train, self.y_train, self.ts, cuda=False)

            print("training shape: x - {}, y - {}".format(self.x_train[0].numpy().shape, self.y_train[0].numpy().shape))
            print("testing shape: x - {}, y - {}".format(self.x_test[0].numpy().shape, self.y_test[0].numpy().shape))
        
        if self.load_type == "multi":
            print("building data for case = 2: x_train is of all the grid sizes, x_test is of all the grid sizes")
            self.max_s = 256
            # interpolate training data
            self.x_train = self.x_train.reshape(self.ntrain, 1, self.grid_sizes[self.ts], self.grid_sizes[self.ts])
            self.x_train = torch.nn.functional.interpolate(self.x_train, size=(self.max_s,self.max_s), mode='bicubic', align_corners=True).reshape(self.ntrain, self.max_s, self.max_s)
            self.y_train = self.y_train.reshape(self.ntrain, 1, self.grid_sizes[self.ts], self.grid_sizes[self.ts])
            self.y_train = torch.nn.functional.interpolate(self.y_train, size=(self.max_s,self.max_s), mode='bicubic', align_corners=True).reshape(self.ntrain, self.max_s, self.max_s)

            # interpolate testing data
            self.x_test = self.x_test.reshape(self.ntest, 1, self.grid_sizes[self.ts], self.grid_sizes[self.ts])
            self.x_test = torch.nn.functional.interpolate(self.x_test, size=(self.max_s,self.max_s), mode='bicubic', align_corners=True).reshape(self.ntest, self.max_s, self.max_s)
            self.y_test = self.y_test.reshape(self.ntest, 1, self.grid_sizes[self.ts], self.grid_sizes[self.ts])
            self.y_test = torch.nn.functional.interpolate(self.y_test, size=(self.max_s,self.max_s), mode='bicubic', align_corners=True).reshape(self.ntest, self.max_s, self.max_s)

            print("training shape: x - {}, y - {}".format(self.x_train[0].numpy().shape, self.y_train[0].numpy().shape))
            print("testing shape: x - {}, y - {}".format(self.x_test[0].numpy().shape, self.y_test[0].numpy().shape))
        
        if self.load_type == "multi3":
            raise NotImplementedError("multi3 is not implemented for ellipses problem")
