"""
@author: Yong Zheng Ong
loads the dataset
"""
import torch
import numpy as np
import scipy.io
import h5py

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
            self.file_path = "datasets/scattering/data/scafull.mat"

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        try:
            self.data = scipy.io.loadmat(file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(file_path)
            self.old_mat = False

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
        if self.available_subs[index] <= 18:
            batch_size = data_x.size()[0]
            batch_x = data_x[:,::self.available_subs[index],::self.available_subs[index]][:,:self.grid_sizes[index],:self.grid_sizes[index]]
            batch_y = data_y[:,::self.available_subs[index],::self.available_subs[index]][:,:self.grid_sizes[index],:self.grid_sizes[index],:]
        else:
            batch_size = data_x.size()[0]
            batch_x = data_x[:,::self.available_subs[self.ts],::self.available_subs[self.ts]][:,:self.grid_sizes[self.ts],:self.grid_sizes[self.ts]].reshape(batch_size,self.grid_sizes[self.ts],self.grid_sizes[self.ts],1).permute(0,3,1,2)
            batch_y = data_y[:,::self.available_subs[self.ts],::self.available_subs[self.ts]][:,:self.grid_sizes[self.ts],:self.grid_sizes[self.ts],:].permute(0,3,1,2)
            batch_x = torch.nn.functional.interpolate(batch_x, size=(self.grid_sizes[index],self.grid_sizes[index]), mode='bicubic', align_corners=True).reshape(batch_size, 1, self.grid_sizes[index], self.grid_sizes[index]).permute(0,2,3,1)
            batch_y = torch.nn.functional.interpolate(batch_y, size=(self.grid_sizes[index],self.grid_sizes[index]), mode='bicubic', align_corners=True).reshape(batch_size, 2, self.grid_sizes[index], self.grid_sizes[index]).permute(0,2,3,1)

        if cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        # build the locations information - here, uniform grid is used
        grids = []
        grids.append(np.linspace(0, 1, self.grid_sizes[index]))
        grids.append(np.linspace(0, 1, self.grid_sizes[index]))
        grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
        grid = grid.reshape(1,self.grid_sizes[index],self.grid_sizes[index],2)
        grid = torch.tensor(grid, dtype=torch.float).to(batch_x.device)

        batch_x = torch.cat([batch_x.reshape(batch_size,self.grid_sizes[index],self.grid_sizes[index],1), grid.repeat(batch_size,1,1,1)], dim=3)

        return batch_x, batch_y

    def read_data(self, filename=None, build_grid=False):
        self.build_grid = build_grid

        if filename is not None:
            # here we need to load the file
            self._load_file(file_path=filename)
        else:
            filename = self.file_path
        print("reading dataset from {}".format(filename))

        # build the max size data
        x_train1 = self.read_field('a')[:self.ntrain,:,:]
        y_train1 = self.read_field('u')[:self.ntrain,:,:]
        y_train2 = self.read_field('ui')[:self.ntrain,:,:]

        x_test1 = self.read_field('a')[-self.ntest:,:,:]
        y_test1 = self.read_field('u')[-self.ntest:,:,:]
        y_test2 = self.read_field('ui')[-self.ntest:,:,:]

        grid_size = x_train1.size(1)

        # use our normalizing, if fno, comment out this part
        t_min, t_max = torch.min(x_train1), torch.max(x_train1)
        x_train1 = (x_train1 - t_min)/(t_max - t_min)
        # print("x_train1: ", torch.min(x_train1), torch.max(x_train1))

        t_min, t_max = torch.min(x_test1), torch.max(x_test1)
        x_test1 = (x_test1 - t_min)/(t_max - t_min)
        # print("x_test1: ", torch.min(x_test1), torch.max(x_test1))

        t_min, t_max = torch.min(y_train1), torch.max(y_train1)
        y_train1 = (y_train1 - t_min)/(t_max - t_min)
        # print("y_train1: ", torch.min(y_train1), torch.max(y_train1))

        t_min, t_max = torch.min(y_test1), torch.max(y_test1)
        y_test1 = (y_test1 - t_min)/(t_max - t_min)
        # print("y_test1: ", torch.min(y_test1), torch.max(y_test1))

        t_min, t_max = torch.min(y_train2), torch.max(y_train2)
        y_train2 = (y_train2 - t_min)/(t_max - t_min)
        # print("y_train2: ", torch.min(y_train2), torch.max(y_train2))

        t_min, t_max = torch.min(y_test2), torch.max(y_test2)
        y_test2 = (y_test2 - t_min)/(t_max - t_min)
        # print("y_test2: ", torch.min(y_test2), torch.max(y_test2))

        # merge
        x_train = x_train1
        x_test = x_test1
        y_train = torch.cat([y_train1.reshape(self.ntrain, grid_size, grid_size, 1), y_train2.reshape(self.ntrain, grid_size, grid_size, 1)], -1)
        y_test = torch.cat([y_test1.reshape(self.ntest, grid_size, grid_size, 1), y_test2.reshape(self.ntest, grid_size, grid_size, 1)], -1)

        if build_grid: # in this case, append the locations information
            # build the locations information - here, uniform grid is used
            grids = []
            grids.append(np.linspace(0, 1, grid_size))
            grids.append(np.linspace(0, 1, grid_size))
            grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
            grid = grid.reshape(1,grid_size,grid_size,2)
            grid = torch.tensor(grid, dtype=torch.float)

            x_train = torch.cat([x_train.reshape(self.ntrain,grid_size,grid_size,1), grid.repeat(self.ntrain,1,1,1)], dim=3)
            x_test = torch.cat([x_test.reshape(self.ntest,grid_size,grid_size,1), grid.repeat(self.ntest,1,1,1)], dim=3)

        return x_train, x_test, y_train, y_test

    def build_data(self, sub=1, target_sub=1, ntrain=10000, ntest=1000, load_type="default"):
        available_build_types = ["default", "multi", "multi3"]
        if load_type not in available_build_types:
            load_type = "multi"

        self.y_test_mask = None

        # set basic details
        self.ntrain = ntrain
        self.ntest = ntest
        self.ts = 0
        self.num_d = 2 # 2 dimension problem
        self.input_channel = 3 # (a(x, y), x, y)
        self.output_channel = 2 # (u(x, y))
        self.load_type = load_type
        self.grid_sizes = []

        if self.load_type in ["default", "multi"]:
            self.available_subs = [1, 2, 3, 161, 241]
            for i in self.available_subs:
                if i <= 18:
                    self.grid_sizes.append(int(((81 - 1)/i) + 1))
                else:
                    self.grid_sizes.append(i)
            print("grid sizes: {}".format(self.grid_sizes))

            self.x_train, self.x_test, self.y_train, self.y_test = self.read_data()

            # set data original size
            self.sub = sub #subsampling rate
            self.s = int(((81 - 1)/self.sub) + 1) #total grid size divided by the subsampling rate

            # set target data size
            self.target_sub = target_sub
            self.target_s = int(((81 - 1)/self.target_sub) + 1)

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

                print("training shape: x - {}, y - {}".format(self.x_train[0].numpy().shape, self.y_train[0].numpy().shape))
                print("testing shape: x - {}, y - {}".format(self.x_test[0].numpy().shape, self.y_test[0].numpy().shape))
        
        if self.load_type in ["multi3"]:
            self.available_subs = [9, 12, 18]
            for i in self.available_subs:
                self.grid_sizes.append(int(9 * i))
            print("grid sizes: {}".format(self.grid_sizes))

            # in multi3, we use a list to store the different datasets
            self.x_train = []
            self.x_test = []
            self.y_train = []
            self.y_test = []

            for i in self.available_subs:
                # read the data
                file_path = "datasets/scattering/data/scafull{}.mat".format(i)

                x_train, x_test, y_train, y_test = self.read_data(filename=file_path, build_grid=True)
                self.x_train.append(x_train)
                self.x_test.append(x_test)
                self.y_train.append(y_train)
                self.y_test.append(y_test)

            # set data original size
            self.sub = sub #subsampling rate
            self.s = int(9 * self.available_subs[self.sub-1])

            # set target data size
            self.target_sub = target_sub
            self.target_s = int(9 * self.available_subs[self.target_sub-1])

            print("train subsampling value: {}, dataset size: {}".format(self.sub, self.s))
            print("test subsampling value: {}, dataset size: {}".format(self.target_sub, self.target_s))

            # create data normalizers
            self.x_normalizer = None
            self.y_normalizer = None

            print("building data for case = 3: x_train is a list of target grid size, x_test is a list of target grid size")

            print("number of datasets - {}".format(len(self.x_train)))
            for i in range(len(self.x_train)):
                print("dataset no. {} ...".format(i))
                print(" "*4+"training shape: x - {}, y - {}".format(self.x_train[i][0].numpy().shape, self.y_train[i][0].numpy().shape))
                print(" "*4+"testing shape: x - {}, y - {}".format(self.x_test[i][0].numpy().shape, self.y_test[i][0].numpy().shape))

        if load_type == "deeponet":
            raise NotImplementedError("deeponet is not implemented yet for darcy problem")
