import numpy as np
import random
import math
import pydicom as dicom
import os
import astra
import copy
import torch
from utils import torch_W, torch_Wt,torch_Weighted_Wt
#from jsr_model_utils import load_data_list, radon_op_new, get_tr_item, norm_l2, ThreshCoeff,PDalg_Model,myAtA
from reconstruct_alg import Wavelet_Model,TVAtA,TV_model,myAtA
from odl.contrib import torch as odl_torch
import odl
from skimage.metrics import structural_similarity as ssim

def read_img(img_path):
    """
    walk the dir to load all image
    """
    img_list=[]
    print('image loading...')
    for _,_,files in os.walk(img_path):
        for f in files:
            if f.find('.dcm')>=0:
                tmp_img=dicom.dcmread(os.path.join(img_path,f))
                tmp_img=tmp_img.pixel_array#[0::2,0::2]
                img_list.append(tmp_img)
    img_data=np.array(img_list)
    print('done')
    return img_data

def error_eval(u_pre,u_true):
    e1=np.linalg.norm(u_pre-u_true,'fro')/np.linalg.norm(u_true,'fro')
    return e1

def psnr(u_pre,u_true):
    norm_pre=u_pre/np.max(u_pre)*255
    norm_true=u_true/np.max(u_true)*255
    mse=np.mean((norm_pre-norm_true)**2)
    if mse==0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr2(u_pre,u_true):
    mse=np.mean((u_pre-u_true)**2)
    if mse==0:
        return 100
    PIXEL_MAX = 4096
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def printlog(str1):
    fp=open('./test.out','a')
    fp.write(str1)
    fp.close()

def astra_alg(u0,proj_u,angle,jsr_use=False,alg='SART',iters=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s1,s2=u0.shape
    proj_u_array=np.array(proj_u)
    angle_array=np.array(angle)*np.pi/180.0
    vol_gem=astra.creators.create_vol_geom((s1,s2))
    proj_gem=astra.creators.create_proj_geom('parallel',1,s1,angle_array)
    if torch.cuda.is_available():
        proj_id=astra.creators.create_projector('cuda',proj_gem,vol_gem)
    else:
        proj_id=astra.creators.create_projector('linear',proj_gem,vol_gem)
    rec_id=astra.data2d.create('-vol',vol_gem,data=u0.copy())
    projdata_id=astra.data2d.create('-sino',proj_gem,data=proj_u_array)
    if torch.cuda.is_available():
        alg_cfg2=astra.astra_dict('SART_CUDA')
    else:
        alg_cfg2=astra.astra_dict('SART')

    alg_cfg2['ProjectorId']=proj_id
    alg_cfg2['ProjectionDataId']=projdata_id
    alg_cfg2['ReconstructionDataId']=rec_id
    alg_cfg2['option']={'MinConstraint':0,'MaxConstraint':4096}
    sart_alg_id=astra.algorithm.create(alg_cfg2)
    astra.algorithm.run(sart_alg_id,iters)
    img_rec=astra.data2d.get(rec_id)
    if alg!='SART':
        matrix_id=astra.projector.matrix(proj_id)
        radon=astra.matrix.get(matrix_id).tocoo()
        row = torch.from_numpy(radon.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(radon.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        val = torch.from_numpy(radon.data.astype(np.float64)).to(torch.float)
        with torch.no_grad():
            radon=torch.sparse.FloatTensor(edge_index, val, torch.Size(radon.shape)).to(device)
            iradon=radon.transpose(0,1)
            def r(x):
                return (radon@x.view(-1,1)).view(-1,s2)
            def ir(x):
                return (iradon@(x.view(-1,1))).view(s1,s2)
            highpass=True
            proj_u=torch.tensor(proj_u_array,dtype=torch.float).to(device)
            u1=torch.tensor(img_rec,dtype=torch.float).to(device)
            if alg=='Wavelet':
                myAtAModule=myAtA(r,ir,torch.tensor(1.0).to(device), torch_W,torch_Weighted_Wt,highpass,'sparse_view',0.0)
                img_rec=Wavelet_Model(proj_u,u1,myAtAModule,20,5,1.0,0.1,'anisotropic','fixed_iter',1e-3).squeeze()
                del myAtAModule
            elif alg=='TVmodel':
                TVAtAModule=TVAtA(r,ir)
                img_rec=TV_model(proj_u,u1,TVAtAModule,20,5,1.0,0.1,'anisotropic','fixed_iter',1e-3).squeeze()
                del TVAtAModule
        
        if torch.cuda.is_available():
            img_rec=img_rec.cpu()
        img_rec=img_rec.numpy()

    astra.clear()
    return img_rec.copy()


class CT():
    def __init__(self,img_path,have_noise=False,each_iter=50,CT_reconstruct_alg='SART'):
        self.img_data=read_img(img_path)
        self.img_data_size=self.img_data.shape[0]
        self.action_num=180+1
        self.stop_action=180
        self.each_iter=each_iter
        self.has_noise=have_noise
        self.max_photon=1.4*180/1e5
        self.scale=100000
        self.proj_data=0
        self.rest=1.0
        if type(CT_reconstruct_alg)==type('string'):
            self.alg=CT_reconstruct_alg
            self.reconstruct_alg=astra_alg
            self.mode=0
        else:
            self.reconstruct_alg=CT_reconstruct_alg
            self.mode=1

    def reset(self,set_pic=None):
        self.state_seq=[]
        self.state_proj_seq=[]
        self.angle_seq=[]
        self.true_img=self.img_data[random.randint(0,self.img_data_size-1)]

        if set_pic is not None:
            self.true_img=set_pic
        #self.true_img=self.img_data[10]
        init_act=random.randint(0,179)
        s1,s2=self.true_img.shape
        proj_data=np.zeros((s1,))
        img_size=self.true_img.shape
        self.state=np.zeros(img_size)
        #self.rest=1.+random.random()*0.2-0.1
        self.rest=1.0
        self.start=True
        self.proj_data=proj_data.copy()
        return proj_data/self.scale,0

    def step(self,action,doze,jsr_use=True,alg='SART'):
        if abs(doze)<1e-4:
            if abs(self.rest)>1e-4:
                self.rest-=doze
                done=False
                if self.rest<=1e-4:
                    done=True
                return self.proj_data.copy()/self.scale,0,done,None
            else:
                return self.proj_data.copy()/self.scale,0,True,None
        proj_data=self.get_project_data(self.true_img,action,doze,self.has_noise)
        if action!=self.stop_action and abs(self.rest)>1e-4:
            self.state_proj_seq.append(proj_data)
            self.angle_seq.append(action)
            self.old_state=self.state
            if self.mode==0:
                if len(self.angle_seq)==50:
                    self.state=self.reconstruct_alg(self.state,self.state_proj_seq,self.angle_seq,jsr_use,alg,iters=100)
                else:
                    self.state=self.reconstruct_alg(self.state,self.state_proj_seq,self.angle_seq,jsr_use,alg,iters=self.each_iter)
            elif self.mode==1:
                self.state=self.reconstruct_alg(self.state,self.state_proj_seq,self.angle_seq)

            #reward=error_eval(self.old_state,self.true_img)-error_eval(self.state,self.true_img)
            #print(error_eval(self.state,self.true_img))
            #if self.start==True:
            #   reward=psnr2(self.state,self.true_img)
            #   self.start=False
            #else:
            #    reward=psnr2(self.state,self.true_img)-psnr2(self.old_state,self.true_img)
            reward=psnr2(self.state,self.true_img)-psnr2(self.old_state,self.true_img)
            done=False
        else:
            reward=0
            done=True
        self.proj_data=proj_data.copy()
        self.rest-=doze
        if self.rest<=1e-4:
            done=True
        return proj_data/self.scale,reward,done,None
    
    def step_no_reward(self,action,doze):
        if abs(doze)<1e-4:
            if abs(self.rest)>1e-4:
                self.rest-=doze
                done=False
                if self.rest<=1e-4:
                    done=True
                return self.proj_data.copy()/self.scale,0,done,None
            else:
                return self.proj_data.copy()/self.scale,0,True,None
        proj_data=self.get_project_data(self.true_img,action,doze,self.has_noise)
        if action!=self.stop_action and abs(self.rest)>1e-4:
            self.state_proj_seq.append(proj_data)
            self.angle_seq.append(action)
            self.old_state=self.state
            done=False
        else:
            reward=0
            done=True
        self.proj_data=proj_data.copy()
        self.rest-=doze
        if self.rest<=1e-4:
            done=True
        return proj_data/self.scale,0,done,None

    def reconstruct(self,iters=500):
        self.state=self.reconstruct_alg(self.state,self.state_proj_seq,self.angle_seq,False,"SART",iters)
        return self.state
    def reconstruct2(self,alg='SART',iters=100):
        for i in range(10):
            self.state=self.reconstruct_alg(self.state,self.state_proj_seq,self.angle_seq,True,alg,iters)
        return self.state

    def get_project_data(self,img,action,doze,noise=False):
        s1,s2=img.shape
        vol_gem=astra.creators.create_vol_geom((s1,s2))
        proj_gem=astra.creators.create_proj_geom('parallel',1,s1,action*np.pi/180.0)
        if torch.cuda.is_available():
            proj_id=astra.creators.create_projector('cuda',proj_gem,vol_gem)
        else:
            proj_id=astra.creators.create_projector('linear',proj_gem,vol_gem)
        img_id=astra.data2d.create('-vol',vol_gem,data=img)
        projdata_id=astra.data2d.create('-sino',proj_gem)
        if torch.cuda.is_available():
            alg_cfg=astra.astra_dict('FP_CUDA')
        else:
            alg_cfg=astra.astra_dict('FP')
        alg_cfg['ProjectorId']=proj_id
        alg_cfg['ProjectionDataId']=projdata_id
        alg_cfg['VolumeDataId']=img_id
        sart_alg_id=astra.algorithm.create(alg_cfg)
        astra.algorithm.run(sart_alg_id)
        img_sino_data=astra.data2d.get(projdata_id)
        if noise:
            mean_p=np.mean(img_sino_data)/1.0e5
            nosie_intensity=1/(math.sqrt(self.max_photon*doze*math.exp(-mean_p)))*5
            img_sino_data+=np.random.randn(1,s1)*nosie_intensity
            img_sino_data=np.clip(img_sino_data,0,None)
        astra.clear()
        return img_sino_data.reshape((s1,)).copy()

    def show_psnr(self):
        return psnr2(self.state,self.true_img)

    def show_ssim(self):
        return ssim(self.state,self.true_img,data_range=4096)