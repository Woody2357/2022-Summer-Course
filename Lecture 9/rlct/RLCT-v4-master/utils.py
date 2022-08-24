import numpy as np
import random
import torch.multiprocessing as mp
import ctypes



class MemorySlide:
    def __init__(self,buffer_size):
        self.buffer_size=buffer_size
        self.buffer=[]

    def insert(self,data):
        self.buffer.append(data)
        #print(len(self.buffer))
        if len(self.buffer)> self.buffer_size:
            del self.buffer[0] 
   
    def clear(self):
        self.buffer=[]
        

    def sample(self,sample_size):
        if sample_size<=len(self.buffer):
            sample_data=random.sample(self.buffer,sample_size)
            return sample_data
        else:
            return self.buffer


class RunningStat(object):
    '''
    Keeps track of first and second moments (mean and variance)
    of a streaming time series.
     Taken from https://github.com/joschu/modular_rl
     Math in http://www.johndcook.com/blog/standard_deviation/
    '''
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape


''' tools to save images '''
import numpy as np
import torch
import scipy.misc as misc
import matplotlib as mp

def torch2np(x_tensor):
    if isinstance(x_tensor, np.ndarray):
        return x_tensor
    elif x_tensor.is_cuda == False:
        x = x_tensor.numpy()
        return x
    else:
        x = x_tensor.cpu().numpy()
        return x

def np2torch(x):
    if isinstance(x, torch.Tensor):
        return x
    else:
        x = torch.from_numpy(x.copy())
        return x

def imshow(x_in,str,dir = 'tmp/'):
    # x = torch2np(x_in)
    x = np.squeeze(x_in)
    # misc.toimage(x, cmin=0, cmax=1).save(dir + str)
    # misc.toimage(x, cmin=0.0,cmax=1.0).save(dir + str)
    # misc.toimage(x, cmin=0.0).save(dir + str)
    # misc.toimage(x).save(dir + str)
    mp.image.imsave(dir+str, x,vmin=np.min(x),vmax=np.max(x))
    # misc.toimage(x,cmin=0.5*(x.min()+x.max())).save(dir + str)

def imshow_ErrorMap(x_in,str,Vmin=-0.1,Vmax=0.1,dir = 'tmp/',data_mode='gpu'):
    # if data_mode=='gpu':
    if isinstance(x_in,torch.Tensor):
        x = torch2np(x_in)
    else:
        x=x_in
    # else:
        # x=x_in
    x = np.squeeze(x)
    # misc.toimage(x, cmin=0, cmax=1).save(dir + str)
    # misc.toimage(x, cmin=Vmin,cmax=Vmax).save(dir + str)
    # before Jan13
    # mp.image.imsave(dir+str, x,vmin=Vmin,vmax=Vmax,cmap='gray',dpi=300)
    # Jan13
    plt.close('all')
    plt.figure(figsize=(6,6))
    plt.imshow(x,cmap='gray',clim=[Vmin, Vmax])
    plt.axis('off') #不显示坐标尺寸
    fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(dir+str,format='png', transparent=True, dpi=300, pad_inches = 0)
    plt.close('all')


import matplotlib.pyplot as plt
def pltimshow(img,str,dir,window=[0.0,1.0],fig_size=3.0,CMAP='gray'):
    plt.close('all')
# plt.figure(figsize=(3,3)) 
    plt.figure(figsize=(6,6))
    # img = np.rot90(x_true_arr_validate[0,...,0], 1)
    # plt.imshow(img,cmap=plt.cm.gray,clim=window)
    # plt.imshow(img,cmap='bone',clim=window)
    # plt.imshow(img,cmap='gray',clim=window)
    plt.imshow(img,cmap=CMAP,clim=window)

    # plt.imshow(img,cmap='gist_gray',clim=window)

    plt.axis('off') #不显示坐标尺寸
    fig = plt.gcf()
    # fig.set_size_inches(7.0/fig_size,7.0/fig_size)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(dir+str,format='png', transparent=True, dpi=300, pad_inches = 0)
    plt.close('all')




''' wavelet toolbox for multilevel wavelet decomposition and reconstruction '''
import numpy as np
import torch

def GenerateFrameletFilter(frame):
    # Haar Wavelet
    if frame==0:
        D1=np.array([0.0, 1.0, 1.0] )/2
        D2=np.array([0.0, 1, -1])/2
# # %     D{3}='rr';
        D3=('cc')
        R1=np.array([1 , 1 ,0])/2
        R2=np.array([-1, 1, 0])/2
# # %     R{3}='rr';
        R3=('cc')
        D=[D1,D2,D3]
        R=[R1,R2,R3]
    # Piecewise Linear Framelet
    elif frame==1:
        D1=np.array([1.0, 2, 1])/4
        D2=np.array([1, 0, -1])/4*np.sqrt(2)
        D3=np.array([-1 ,2 ,-1])/4
# %     D{4}='rrr';
        D4='ccc';
        R1=np.array([1, 2, 1])/4
        R2=np.array([-1, 0, 1])/4*np.sqrt(2)
        R3=np.array([-1, 2 ,-1])/4
# %     R{4}='rrr';
        R4='ccc'
        # D1=tf.convert_to_tensor(D1, tf.float32, name='D')
        # R1=tf.convert_to_tensor(R1, tf.float32, name='R')
        D=[D1,D2,D3,D4]
        R=[R1,R2,R3,R4]
    # Piecewise Cubic Framelet
    elif frame==3:
        D1=np.array([1, 4 ,6, 4, 1])/16
        D2=np.array([1 ,2 ,0 ,-2, -1])/8
        D3=np.array([-1, 0 ,2 ,0, -1])/16*np.sqrt(6)
        D4=np.array([-1 ,2 ,0, -2, 1])/8
        D5=np.array([1, -4 ,6, -4, 1])/16
        D6='ccccc'
        R1=np.array([1 ,4, 6, 4 ,1])/16
        R2=np.array([-1, -2, 0, 2, 1])/8
        R3=np.array([-1, 0 ,2, 0, -1])/16*np.sqrt(6)
        R4=np.array([1 ,-2, 0, 2, -1])/8
        R5=np.array([1, -4, 6, -4 ,1])/16
        R6='ccccc'
        D=[D1,D2,D3,D4,D5,D6]
        R=[R1,R2,R3,R4,R5,R6]
    else:
        print('*** Error filter type !!!')
        raise ValueError('Only allowed to choose frame==0, 1 or 3')
    return D,R



D,R=GenerateFrameletFilter(frame=1)
D_tmp=torch.zeros(3,1,3,1)
for ll in range(3):
    # print('shape of D[ll]=',np.shape(D[ll]))
    D_tmp[ll,]=torch.from_numpy(np.reshape(D[ll],(-1,1)))

W=D_tmp
W2=W.permute(0,1,3,2)
kernel_dec=np.kron(W.numpy(),W2.numpy())
kernel_dec=torch.from_numpy(kernel_dec)

R_tmp=torch.zeros(3,1,1,3)
for ll in range(3):
    R_tmp[ll,]=torch.from_numpy(np.reshape(R[ll],(1,-1)))

R=R_tmp
R2=R_tmp.permute(0,1,3,2)
kernel_rec=np.kron(R2.numpy(),R.numpy())
kernel_rec=torch.from_numpy(kernel_rec).view(9,1,3,3)
# kernel_rec=torch.from_numpy(kernel_rec).view(1,9,3,3)

import torch.nn.functional as F
from torch.autograd import Variable
kernel_dec=Variable(kernel_dec, requires_grad=False).float()
# kernel_rec=torch.nn.Parameter(data=kernel_rec, requires_grad=False)
#kernel_rec=Variable(kernel_rec, requires_grad=False).to(torch.float64)
kernel_rec=Variable(kernel_rec, requires_grad=False).float()
#kernel_dec=F.pad(kernel_dec, (1,1,1,1), mode='circular')


def torch_W(img,kernel_dec=kernel_dec.cuda() if torch.cuda.is_available() else kernel_dec,highpass=True):
    if highpass:
        #assert kernel_dec.shape[0]==8
        Dec_coeff=F.conv2d(F.pad(img, (1,1,1,1), mode='circular'),kernel_dec[1:,...])
    else:
        assert kernel_dec.shape[0]==9
        Dec_coeff=F.conv2d(F.pad(img, (1,1,1,1), mode='circular'),kernel_dec)
    return Dec_coeff

def torch_Wt(Dec_coeff,kernel_rec=kernel_rec.cuda() if torch.cuda.is_available() else kernel_rec,highpass=True):
    kernel_rec=kernel_rec.view(1,9,3,3)
    if highpass:
        # print('shape of coeff=',Dec_coeff.shape)
        # print('shape of kernel_rec=',kernel_rec.shape)
        rec_img=F.conv2d(F.pad(Dec_coeff, (1,1,1,1), mode='circular'),kernel_rec[:,1:,...])
        #rec_img=crop_edge_WtW(rec_img,pad_edge=3)
    else:
        assert kernel_rec.shape[1]==9
        rec_img=F.conv2d(F.pad(Dec_coeff, (1,1,1,1), mode='circular'),kernel_rec)
    return rec_img

def torch_Weighted_Wt(Dec_coeff,mu,kernel_rec=kernel_rec.cuda() if torch.cuda.is_available() else kernel_rec,highpass=True):
    kernel_rec=kernel_rec.view(9,1,3,3)
    if highpass:
        tem_coeff=F.conv2d(F.pad(Dec_coeff, (1,1,1,1), mode='circular'),
            kernel_rec[1:,:,...],groups=8)
        assert tem_coeff.shape[1]==8
        rec_img=torch.sum(mu*tem_coeff,dim=1,keepdim=True)
    else:
        assert kernel_rec.shape[0]==9
        tem_coeff=F.conv2d(F.pad(Dec_coeff, (1,1,1,1), mode='circular'),kernel_rec,groups=9)
        assert tem_coeff.shape[1]==9
        mu_tem_coeff=mu*tem_coeff[:,1:,...]
        weighted_coeff=torch.cat([tem_coeff[:,0:1,...],mu_tem_coeff],dim=1)
        rec_img=torch.sum(weighted_coeff,dim=1,keepdim=True)
    return rec_img

def crop_edge_WtW(x,pad_edge=3):
    output=x[...,pad_edge:-pad_edge,pad_edge:-pad_edge]
    pad_zero = torch.nn.ConstantPad2d(pad_edge, 0.0)
    output=pad_zero(output)
    return output
