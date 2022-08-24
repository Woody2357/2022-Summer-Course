###wavelet framework

import numpy as np
import torch
import torch.nn.functional as F
def norm_l2(img):
    return (img**2).sum()

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
        D4='ccc'
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


def Dec_Rec_kernel(torch_var=True):
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
    kernel_rec=torch.from_numpy(kernel_rec).view(1,9,3,3)
    if torch_var:
        # kernel_dec=torch.nn.Parameter(data=kernel_dec, requires_grad=False)
        kernel_dec=Variable(kernel_dec, requires_grad=False)
        # kernel_rec=torch.nn.Parameter(data=kernel_rec, requires_grad=False)
        kernel_rec=Variable(kernel_rec, requires_grad=False)

    return kernel_dec, kernel_rec

import torch.nn.functional as F
# Pad=torch.nn.ReflectionPad2d(2)
Pad=torch.nn.ReflectionPad2d(1)

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
        rec_img=F.conv2d(F.pad(Dec_coeff, (1,1,1,1), mode='circular'),kernel_rec[:,1:,...])
    else:
        assert kernel_rec.shape[1]==9
        rec_img=F.conv2d(F.pad(Dec_coeff, (1,1,1,1), mode='circular'),kernel_rec)
    return rec_img

def crop_edge_WtW(x,pad_edge=3):
    output=x[...,pad_edge:-pad_edge,pad_edge:-pad_edge]
    pad_zero = torch.nn.ConstantPad2d(pad_edge, 0.0)
    output=pad_zero(output)
    return output


def ThreshCoeff(x,gamma,thresh_mode='isotropic'):
    if x.size(1)==9:
        coeff=torch.zeros_like(x)
        # print('Threshold op removes the lowpass component!!!')
        for ii in range(1,x.size(1)):
            coeff[:,ii,:,:] = torch.sign(x[:,ii,:,:])*torch.clamp(torch.abs(x[:,ii,:,:])-gamma,0.0)
    else:
        if x.shape[1]==8:
            if thresh_mode=='anisotropic':
                coeff = torch.sign(x)*torch.clamp(torch.abs(x)-gamma,0.0)
            elif thresh_mode=='isotropic':
                tmp_sum=torch.sqrt(torch.sum(x**2,dim=(1)).unsqueeze(1))
                coeff=torch.clamp(x-x*gamma/(tmp_sum+1e-20),0.0)
        elif x.shape[1]==9:
            coeff=x
            if thresh_mode=='anisotropic':
                coeff[:,1:,...] = torch.sign(x[:,1:,...])*torch.clamp(torch.abs(x[:,1:,...])-gamma,0.0)
            elif thresh_mode=='isotropic':
                tmp_sum=torch.sqrt(torch.sum(x[:,1:,...]**2,dim=(1)).unsqueeze(1))
                coeff[:,1:,...]=torch.clamp(x[:,1:,...]-x[:,1:,...]*gamma/(tmp_sum+1e-20),0.0)
            # coeff = torch.sign(x)*torch.clamp(torch.abs(x)-gamma,0.0)
    return coeff


class myAtA():
    def __init__(self,Radon,iRadon,sino_weight, WDec,WRec,highpass,rec_task,sino_mask):
        # self.reblur = BackProjLayer()
        #printlog("init myAtA")
        self.Radon=Radon
        self.iRadon=iRadon
        self.WDec=WDec
        self.WRec=WRec
        self.highpass=highpass
        self.sino_weight=sino_weight
        self.rec_mode=rec_task
        self.sino_mask=sino_mask
        #print(Radon)
        #print(iRadon)
    def AtA_WtW_x(self,img,mu):
        Ax0=self.BwAt(self.FwA(img))+self.Wt(self.W(img),mu)
        return Ax0
    def FwA(self,img):
        return self.Radon(img)

    def BwAt(self,sino):
        # assert len(sino.shape)==4
        return self.iRadon(sino/self.sino_weight)

    def W(self,img):
        return self.WDec(img.unsqueeze(0).unsqueeze(0),highpass=self.highpass)
    def Wt(self,Wu,mu):
        if self.highpass==False:
            assert Wu.shape[1]==9
        return self.WRec(Wu,mu,highpass=self.highpass).squeeze()
    def pATAp(self,img):
        
        Ap=self.FwA(img)
        AtAp=self.BwAt(Ap)
        # pATApNorm=torch.sum(Ap*Ap,dim=(1,2,3))
        pATApNorm=torch.sum(img*AtAp)
        return pATApNorm
    def pWTWp(self,img,mu):
        Wp=self.W(img)
        if Wp.shape[1]==9:
            mu_Wp=mu*Wp[:,1:,...]*Wp[:,1:,...]
            pWTWpNorm=torch.sum(torch.cat([Wp[:,0:1,...]*Wp[:,0:1,...],mu_Wp],dim=1),dim=(1,2,3))
        elif  Wp.shape[1]==8:
            mu_Wp=mu*Wp*Wp
            pWTWpNorm=torch.sum(mu_Wp,dim=(1,2,3))
        return pWTWpNorm

def CG_alg_new(myAtA,x,mu,res,CGiter=5):
    # print('\n')
    # print('-------------------------This is the CG algorithm')
    r=res
    p=-res
    s=(res**2).sum()
    if s==0:
        return x
    for k in range(CGiter):
        pATApNorm = myAtA.pATAp(p)
        mu_pWtWpNorm=myAtA.pWTWp(p,mu)
        rTr=torch.sum(r**2)
        alphak = rTr/(mu_pWtWpNorm+pATApNorm)
        x = x+alphak*p
        r = r+alphak*myAtA.AtA_WtW_x(p,mu)
        betak = torch.sum(r**2)/ rTr
        p=-r+betak*p

    pATApNorm = myAtA.pATAp(p)
    mu_pWtWpNorm=myAtA.pWTWp(p,mu)
    rTr=torch.sum(r**2)
    alphak = rTr/(mu_pWtWpNorm+pATApNorm)
    x = x+alphak*p
    return x
def printlog(str1):
    fp=open('./test.out','a')
    fp.write(str1)
    fp.close()


def Wavelet_Model(Y,u0,myAtAModule,MaxIter,CG_Iter,mu,beta,thresh_mode,stop_criterion='fixed_iter',tol=1e-8):
    with torch.no_grad():
        u_old=u0
        y=myAtAModule.W(u_old)
        #printlog(f"y:{y}")
        s1=y.size()
        v=torch.zeros(s1)
        if torch.cuda.is_available():
            v=v.cuda()
    #print(f"v:{v}")
        for ii in range(MaxIter):
            if ii>0:
                u_old=u
            rhs=myAtAModule.BwAt(Y)+myAtAModule.Wt(y-v,mu)
        #print(f"rhs:{rhs}")
            Ax0=myAtAModule.AtA_WtW_x(u_old,mu)
        #print(f"Ax0:{Ax0}")
            res=Ax0-rhs
        #print(f"res:{res}")
            u=CG_alg_new(myAtAModule,u_old,mu,res, CG_Iter)
        #print(f"u:{u}")
            Wu=myAtAModule.W(u)
            y=ThreshCoeff(Wu+v,beta/mu,thresh_mode)
            v=v+(Wu-y)
            RelErr=norm_l2(u-u_old)/norm_l2(u)
            RelErr=torch.sqrt(RelErr)
        #print(f"RelErr:{RelErr}")
            if stop_criterion=='fixed_iter':
                pass
            else:
                if RelErr<tol:
                    print('Adopted the relative error stopping criterion!!!--RelErr=',RelErr)
                    break
    torch.cuda.empty_cache()
    return u


class TVAtA():
    def __init__(self,Radon,iRadon):
        # self.reblur = BackProjLayer()
        self.Radon=Radon
        self.iRadon=iRadon
        self.kernel=torch.tensor([[[0,0,0],[-0.5,0,0.5],[0,0,0]],[[0,0.5,0],[0,0,0],[0,-0.5,0]]]).unsqueeze(1)
        if torch.cuda.is_available():
            self.kernel=self.kernel.cuda()

    def AtA_WtW_x(self,img,mu):
        Ax0=self.BwAt(self.FwA(img))+self.Wt(self.W(img),mu)
        return Ax0
    def FwA(self,img):
        return self.Radon(img)

    def BwAt(self,sino):
        # assert len(sino.shape)==4
        return self.iRadon(sino)

    def W(self,img):
        return torch.conv2d(img.unsqueeze(0).unsqueeze(0),self.kernel)

    def Wt(self,Wu,mu):
        return torch.conv_transpose2d(Wu*mu,self.kernel).squeeze()

    def pATAp(self,img):
        
        Ap=self.FwA(img)
        AtAp=self.BwAt(Ap)
        # pATApNorm=torch.sum(Ap*Ap,dim=(1,2,3))
        pATApNorm=torch.sum(img*AtAp)
        return pATApNorm

    def pWTWp(self,img,mu):
        Wp=self.W(img)
        mu_Wp=mu*Wp*Wp
        pWTWpNorm=torch.sum(mu_Wp,dim=(1,2,3))
        return pWTWpNorm

def softThresh(u,lam):
    return torch.sign(u)*torch.clamp(torch.abs(u)-lam,0)

def TV_model(Y,u0,TvAtA,MaxIter,CG_Iter,mu,beta,thresh_mode,stop_criterion='fixed_iter',tol=1e-8):
    with torch.no_grad():
        u_old=u0
        y=TvAtA.W(u_old)
        s1=y.size()
        v=torch.zeros(s1)
        if torch.cuda.is_available():
            v=v.cuda()
        for ii in range(MaxIter):
            if ii>0:
                u_old=u
        #print(TvAtA.BwAt(Y).size())
        #print(TvAtA.Wt(y-v,mu).size())
            rhs=TvAtA.BwAt(Y)+TvAtA.Wt(y-v,mu)
            Ax0=TvAtA.AtA_WtW_x(u_old,mu)
            res=Ax0-rhs
            u=CG_alg_new(TvAtA,u_old,mu,res, CG_Iter)
            Wu=TvAtA.W(u)
            y=softThresh(Wu+v,beta/mu)
            v=v+(Wu-y)
            RelErr=norm_l2(u-u_old)/norm_l2(u)
            RelErr=torch.sqrt(RelErr)
        #print(f"RelErr:{RelErr}")
            if stop_criterion=='fixed_iter':
                pass
            else:
                if RelErr<tol:
                    print('Adopted the relative error stopping criterion!!!--RelErr=',RelErr)
                    break
    return u
