import odl
import numpy as np
from odl.contrib import torch as odl_torch

def load_data_list(data_list,radon,iradon,fbp,img_size, angles,detectors,
    sino_mask,rec_task,noise_std):
    img_num=len(data_list)
    img = np.zeros((img_num, 1, *img_size), dtype=np.float32)
    u0 = np.zeros((img_num, 1, *img_size), dtype=np.float32)
    sino = np.zeros((img_num, 1, angles,detectors), dtype=np.float32)
    for ii in range(img_num):
        file_name=data_list[ii]
        print('Current file idx={}/{},file_name={}'.format(ii,img_num,file_name))
        img[ii,0,], u0[ii,0,], sino[ii,0,] = get_tr_item(file_name,radon,fbp,img_size,
            angles,detectors,sino_mask,rec_task,noise_std)

    img = torch.from_numpy(img)
    u0 = torch.from_numpy(u0)
    sino = torch.from_numpy(sino)
    print('[****load_data_list****]-->output img  shape=',img.size())
    print('[****load_data_list****]-->output u0   shape=',u0.size())
    print('[****load_data_list****]-->output sino shape=',sino.size())
    return img,u0,sino


def radon_op(img_size=[512,512],sino_size=[300,1024]):
    # xx=128
    # xx=192
    # xx=384//3
    xx=384//4
    # xx=384//5
    # xx=384//6
    # xx=384//8
    yy=360
    # yy=700
    # print('---------------------------------------------->xx=',xx)
    # space = odl.uniform_discr([-128, -128], [128, 128], 
    space = odl.uniform_discr([-xx, -xx], [xx, xx], [img_size[0], img_size[0]],dtype='float32')
    angles=sino_size[0]
    angle_partition = odl.uniform_partition(0, 2 * np.pi, angles)
    angle_partition_half = odl.uniform_partition(0, 2 * np.pi, angles//2)
    detectors=sino_size[1]
    detector_partition = odl.uniform_partition(-yy, yy, detectors)
    # detector_partition = odl.uniform_partition(-360, 360, detectors)
    geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,src_radius=500, det_radius=500)
    geometry_half = odl.tomo.FanFlatGeometry(angle_partition_half, detector_partition,src_radius=500, det_radius=500)
# geometry = odl.tomo.parallel_beam_geometry(space, num_angles=300)
    operator = odl.tomo.RayTransform(space, geometry,impl='astra_cuda')
    operator_half = odl.tomo.RayTransform(space, geometry_half,impl='astra_cuda')
    fbp = odl.tomo.fbp_op(operator, filter_type='Hann', frequency_scaling=0.8)
    fbp_half = odl.tomo.fbp_op(operator_half, filter_type='Hann', frequency_scaling=0.8)

    op_layer = odl_torch.operator.OperatorAsModule(operator)
    op_layer_half = odl_torch.operator.OperatorAsModule(operator_half)
    op_layer_adjoint = odl_torch.operator.OperatorAsModule(operator.adjoint)
    fbp_layer = odl_torch.operator.OperatorAsModule(fbp)
    fbp_layer_half = odl_torch.operator.OperatorAsModule(fbp_half)

    return op_layer, op_layer_half, op_layer_adjoint, fbp_layer, fbp_layer_half

def radon_op_new(img_size=[512,512],sino_size=[300,1024]):
    xx=96
    space = odl.uniform_discr([-xx, -xx], [xx, xx], [img_size[0],img_size[0]],dtype='float32')
    angles=sino_size[0]
    angle_partition = odl.uniform_partition(0, 2 * np.pi, angles)
    detectors=sino_size[1]
    detector_partition = odl.uniform_partition(-200, 200, detectors)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
    # geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition,src_radius=800, det_radius=0)
    # geometry = odl.tomo.parallel_beam_geometry(space, num_angles=detectors)
    #operator = odl.tomo.RayTransform(space, geometry,impl='astra_cuda')
    operator = odl.tomo.RayTransform(space, geometry,impl='astra_cpu')
    # operator = odl.tomo.RayTransform(space, geometry,impl='skimage')
    
    op_norm=odl.operator.power_method_opnorm(operator)
    print('op.norm=',op_norm)
    # op_norm2=odl.operator.power_method_opnorm(operator.adjoint)
    # print('op.adjoint_norm=',op_norm2)
    print(dir(odl_torch.operator))
    #op_layer=odl_torch.operator.OperatorAsModule(operator)
    op_layer = odl_torch.operator.OperatorAsModule(operator)
    op_layer_adjoint = odl_torch.operator.OperatorAsModule(operator.adjoint)
    fbp = odl.tomo.fbp_op(operator, filter_type='Hann', frequency_scaling=0.9)
    # fbp = odl.tomo.fbp_op(operator,filter_type='Ram-Lak',frequency_scaling=0.9)
    op_layer_fbp = odl_torch.operator.OperatorAsModule(fbp)
    return op_layer, op_layer_adjoint, op_layer_fbp,op_norm

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

from torch.autograd import Variable
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
    #if torch_var:
    #    # kernel_dec=torch.nn.Parameter(data=kernel_dec, requires_grad=False)
    #    kernel_dec=Variable(kernel_dec, requires_grad=False)
    #    # kernel_rec=torch.nn.Parameter(data=kernel_rec, requires_grad=False)
    #    kernel_rec=Variable(kernel_rec, requires_grad=False)

    return kernel_dec, kernel_rec

import torch.nn.functional as F
# Pad=torch.nn.ReflectionPad2d(2)
Pad=torch.nn.ReflectionPad2d(1)


def torch_W_dec(img,kernel_dec,pad=Pad,highpass=True):
    if highpass:
        Dec_coeff=F.conv2d(pad(img),kernel_dec[1:,...])
    else:
        Dec_coeff=F.conv2d(pad(img),kernel_dec)
    return Dec_coeff

def torch_W_rec(Dec_coeff,kernel_rec,pad=Pad,highpass=True):
    if highpass:
        rec_img=F.conv2d(pad(Dec_coeff),kernel_rec[:,1:,...])
        rec_img=crop_edge_WtW(rec_img,pad_edge=3)
    else:
        rec_img=F.conv2d(pad(Dec_coeff),kernel_rec)
    return rec_img

# def torch_W(img,pad=Pad,kernel_dec=kernel_dec.cuda(),highpass=True):
#     if highpass:
#         Dec_coeff=F.conv2d(pad(img),kernel_dec[1:,...])
#     else:
#         Dec_coeff=F.conv2d(pad(img),kernel_dec)
#     return Dec_coeff

# def torch_Wt(Dec_coeff,pad=Pad,kernel_rec=kernel_rec.cuda(),highpass=True):
#     if highpass:
#         rec_img=F.conv2d(pad(Dec_coeff),kernel_rec[:,1:,...])
#         rec_img=crop_edge_WtW(rec_img,pad_edge=3)
#     else:
#         rec_img=F.conv2d(pad(Dec_coeff),kernel_rec)
#     return rec_img


def crop_edge_WtW(x,pad_edge=3):
    output=x[...,pad_edge:-pad_edge,pad_edge:-pad_edge]
    pad_zero = torch.nn.ConstantPad2d(pad_edge, 0.0)
    output=pad_zero(output)
    return output

import pydicom
from skimage import transform
import torch 
def np2torch(x):
    return torch.from_numpy(x)

def get_tr_item(file_name, radon, fbp,img_size, angles,detectors, sino_mask,
    rec_task,noise_std):
        '''
        load training item one by one
        '''
        # print('self.sp_file[i]=',data_list[i][-26:])
        dcm=pydicom.read_file(file_name)
        dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        data=dcm.image
        data=np.array(data).astype(float)
        data=transform.resize(data, (img_size))
        if rec_task=='limit_angle':
            phantom = np.rot90(data, 1)
        else:
            phantom = np.rot90(data, 0)
            # phantom = np.rot90(data, -1)
        phantom=(phantom-np.min(phantom))/(np.max(phantom)-np.min(phantom))
        phantom=np2torch(phantom)
        phantom = phantom.unsqueeze(0)
        
        sino=radon(phantom)
        noise = noise_std*np.random.randn(angles,detectors)
        noise = torch.from_numpy(noise)
        noisy_sino=sino.type(torch.DoubleTensor)+noise

        if rec_task=='sparse_view':
            fbpu=fbp(noisy_sino)
        elif rec_task=='limit_angle':
            fbpu=fbp(noisy_sino*sino_mask.cpu())

        return phantom, fbpu, noisy_sino


def norm_l2(img):
    return (img**2).sum()

def CG_alg(x, gamma, r,torch_W,torch_Wt, P, Pt, CGiter):
    # print('\n')
    # print('-------------------------This is the CG algorithm')
    with torch.no_grad():
        rb=r
        pb=-r
        for k in range(CGiter):
            # print('---->CG iter=',k)
            if k == 0:
                 p = -r
            else:
                betak = torch.sum(r**2,dim=(1,2,3))/ torch.sum(rb**2,dim=(1,2,3))
                betak = betak.view(-1,1,1,1)
                p = -r+betak*pb
            ap = P(p.squeeze(1))
            # print('[******************************]size of ap=',ap.size())
            pAp=torch.sum(ap**2,dim=(1,2))
            # print('[******************************]size of pAp=',pAp.size())
            Wp=torch_W(p)
            # print('[******************************]size of Wp=',Wp.size())
            pWp=Wp*Wp*gamma
            # print('[**CG_alg****]size of pWp=',pWp.shape)
            gamma_pWtWp=torch.sum(pWp,dim=(1,2,3))
            # print('[**CG_alg****]size of gamma_pWtWp=',gamma_pWtWp.shape)
            # print('[******************************]size of gamma_pWtWp=',gamma_pWtWp.size())
            # print('[**CG_alg****]size of r=',r.size())
            r_norm=torch.sum(r**2,dim=(1,2,3))
            # print('[**CG_alg****]size of r_norm=',r_norm.size())
            alphak = r_norm/(gamma_pWtWp+pAp)
            alphak = alphak.view(-1,1,1,1)
            # print('[**CG_alg****]size of x=',x.size())
            # print('[**CG_alg****]size of alphak=',alphak.shape)
            x = x+alphak*p
            rb = r
            pb = p
            # AtAp=back_rec(back_rec(p.squeeze(1),radon)*mask.squeeze(1),iradon).unsqueeze(1)
            AtAp=Pt(P(p.squeeze(1))).unsqueeze(1)
            # print('[**CG_alg****]size of AtAp=',AtAp.shape)
            # print('[**CG_alg****]size of Wp=',Wp.shape)
            gamma_WtWp=torch_Wt(Wp*gamma)
            # print('[**CG_alg****]size of gamma_WtWp=',gamma_WtWp.shape)
            tmp=AtAp+gamma_WtWp
            # print('[**CG_alg****]size of tmp=',tmp.shape)
            r = r+alphak*tmp

        x=torch.clamp(x, min=0.0)
    return x


def CG_new(x, mu,rhs,torch_W,torch_Wt, P, Pt, CGiter):
    # print('\n')
    # print('-------------------------This is the CG algorithm')
    with torch.no_grad():
        r=rhs
        p=rhs
        for k in range(CGiter):
        # print('---->CG iter=',k)
        # if k == 0:
        #     p = -r
        # else:
        #     betak = torch.sum(r**2,dim=(1,2,3))/ torch.sum(rb**2,dim=(1,2,3))
        #     betak = betak.view(-1,1,1,1)
        #     p = -r+betak*pb
            ap = P(p.squeeze(1))
            pAp=torch.sum(ap*ap,dim=(1,2))
        
        # mu*pt*Wt*W*p
            Wp=torch_W(p)
            pWp=Wp*Wp*mu
            mu_pWtWp=torch.sum(pWp,dim=(1,2,3))

            pTAp=mu_pWtWp+pAp
            rTr=torch.sum(r*r,dim=(1,2,3))
            alphak = rTr/pTAp
            alphak = alphak.view(-1,1,1,1)
        # print('[**CG_alg****]size of x=',x.size())
        # print('[**CG_alg****]size of alphak=',alphak.shape)
            x = x+alphak*p
        # rb = r
        # pb = p
        # AtAp=back_rec(back_rec(p.squeeze(1),radon)*mask.squeeze(1),iradon).unsqueeze(1)
            AtAp=Pt(P(p.squeeze(1))).unsqueeze(1)
        # print('[**CG_alg****]size of AtAp=',AtAp.shape)
        # print('[**CG_alg****]size of Wp=',Wp.shape)
            mu_WtWp=torch_Wt(torch_W(p)*mu)
        # print('[**CG_alg****]size of gamma_WtWp=',gamma_WtWp.shape)
            tmp=AtAp+mu_WtWp
        # print('[**CG_alg****]size of tmp=',tmp.shape)
            r = r-alphak*tmp
            betak=torch.sum(r*r,dim=(1,2,3))/rTr
            betak = betak.view(-1,1,1,1)
            p = r+betak*p

        x=torch.clamp(x, min=0.0)
    return x



# recU=CG_alg_JSR(u0,alpha,gamma, mu,residual,kernel_dec,kernel_rec, P, Pt,mask, CG_Iter)
def CG_alg_JSR(x,alp,gam, mu, r,kernel_dec,kernel_rec, P, Pt,mask, CGiter):
    # print('\n')
    # print('-------------------------This is the CG algorithm')
    rb=r
    pb=-r
    for k in range(CGiter):
        # print('---->CG iter=',k)
        if k == 0:
            p = -r
        else:
            betak = torch.sum(r**2,dim=(1,2,3))/ torch.sum(rb**2,dim=(1,2,3))
            betak = betak.view(-1,1,1,1)
            p = -r+betak*pb
        ap = P(p.squeeze(0)).double()
        # print('[******************************]size of ap=',ap.size())
        pAp=gam*torch.sum((mask.squeeze(0)*ap)**2,dim=(1,2))+alp*torch.sum(((1.0-mask.squeeze(0))*ap)**2,dim=(1,2))
        # print('[******************************]size of pAp=',pAp.size())
        # Wp=torch_W_dec(p,kernel_dec,highpass=False)
        mu_p_norm=mu*torch.sum(p**2,dim=(1,2,3))
        # print('[******************************]size of Wp=',Wp.size())
        # gamma_pWtWp=gamma*torch.sum(Wp**2,dim=(1,2,3))
        # print('[**CG_alg****]size of gamma_pWtWp=',gamma_pWtWp.shape)
        # print('[******************************]size of gamma_pWtWp=',gamma_pWtWp.size())
        # print('[**CG_alg****]size of r=',r.size())
        r_norm=torch.sum(r**2,dim=(1,2,3))
        # print('[**CG_alg****]size of r_norm=',r_norm.size())
        alphak = r_norm/(mu_p_norm+pAp)
        alphak = alphak.view(-1,1,1,1)
        # print('[**CG_alg****]size of x=',x.size())
        # print('[**CG_alg****]size of alphak=',alphak.shape)
        x = x+alphak*p
        rb = r
        pb = p
        # AtAp=back_rec(back_rec(p.squeeze(1),radon)*mask.squeeze(1),iradon).unsqueeze(1)
        AtAp=gam*Pt(mask.squeeze(0)*ap)+alp*Pt((1.0-mask.squeeze(0))*ap)
        AtAp=AtAp.unsqueeze(1).double()
        # print('[**CG_alg****]size of AtAp=',AtAp.shape)
        # print('[**CG_alg****]size of Wp=',Wp.shape)
        gamma_WtWp=mu*p
        # print('[**CG_alg****]size of gamma_WtWp=',gamma_WtWp.shape)
        tmp=AtAp+gamma_WtWp
        # print('[**CG_alg****]size of tmp=',tmp.shape)
        r = r+alphak*tmp

    x=torch.clamp(x, min=0.0)
    return x


def CG_alg_mask(x, gamma, r,torch_W,torch_Wt, P, Pt,mask, CGiter):
    # print('\n')
    # print('-------------------------This is the CG algorithm')
    rb=r
    pb=-r
    for k in range(CGiter):
        # print('---->CG iter=',k)
        if k == 0:
            p = -r
        else:
            betak = torch.sum(r**2,dim=(1,2,3))/ torch.sum(rb**2,dim=(1,2,3))
            betak = betak.view(-1,1,1,1)
            p = -r+betak*pb
        ap = P(p.squeeze(1))*mask.squeeze(1)
        # print('[******************************]size of ap=',ap.size())
        pAp=torch.sum(ap**2,dim=(1,2))
        # print('[******************************]size of pAp=',pAp.size())
        Wp=torch_W(p)
        # print('[******************************]size of Wp=',Wp.size())
        pWp=Wp*Wp*gamma
        # print('[**CG_alg****]size of pWp=',pWp.shape)
        gamma_pWtWp=torch.sum(pWp,dim=(1,2,3))
        # print('[**CG_alg****]size of gamma_pWtWp=',gamma_pWtWp.shape)
        # print('[******************************]size of gamma_pWtWp=',gamma_pWtWp.size())
        # print('[**CG_alg****]size of r=',r.size())
        r_norm=torch.sum(r**2,dim=(1,2,3))
        # print('[**CG_alg****]size of r_norm=',r_norm.size())
        alphak = r_norm/(gamma_pWtWp+pAp)
        alphak = alphak.view(-1,1,1,1)
        # print('[**CG_alg****]size of x=',x.size())
        # print('[**CG_alg****]size of alphak=',alphak.shape)
        x = x+alphak*p
        rb = r
        pb = p
        # AtAp=back_rec(back_rec(p.squeeze(1),radon)*mask.squeeze(1),iradon).unsqueeze(1)
        AtAp=Pt(mask.squeeze(1)* P(p.squeeze(1))).unsqueeze(1)
        # print('[**CG_alg****]size of AtAp=',AtAp.shape)
        # print('[**CG_alg****]size of Wp=',Wp.shape)
        gamma_WtWp=torch_Wt(Wp*gamma)
        # print('[**CG_alg****]size of gamma_WtWp=',gamma_WtWp.shape)
        tmp=AtAp+gamma_WtWp
        # print('[**CG_alg****]size of tmp=',tmp.shape)
        r = r+alphak*tmp

    x=torch.clamp(x, min=0.0)
    return x


#from utils import comfft as cf
#from utils import Fwv
def update_U(Y,u0,P,Pt,torch_W,torch_Wt,v,mu,CG_Iter):
    PtY=Pt(Y.squeeze(1)).unsqueeze(1)
    # print('shape of PtY=',PtY.shape)
    # print('shape of v=',v.shape)
    mu_WtV=torch_Wt(mu*v)
    # print('shape of mu_WtV=',mu_WtV.shape)
    rhs_term=PtY+mu_WtV
    mu_WtWu=torch_Wt(mu*torch_W(u0))
    residual=Pt(P(u0.squeeze(1))).unsqueeze(1)+mu_WtWu-rhs_term
    recU=CG_alg(u0, mu,residual,torch_W,torch_Wt, P, Pt, CG_Iter)
    return recU

# u=update_U_JSR(Y,ff,u_old,P,Pt,mask,kernel_dec,kernel_rec,vu-bu,alpha,gamma,muU,CG_Iter)
def update_U_JSR(Y,f,u0,P,Pt,mask,kernel_dec,kernel_rec,vu,alpha,gamma,mu,CG_Iter):
    PtMY=gamma*Pt(mask* Y).double()
    PtMcF=alpha*Pt((1.0-mask)* f).double()
    # print('shape of PtMY=',PtMY.shape)
    # print('shape of PtMcF=',PtMcF.shape)
    mu_WtV=mu*torch_W_rec(vu,kernel_rec,highpass=False)
    # print('shape of mu_WtV=',mu_WtV.shape)
    rhs_term=PtMY+PtMcF+mu_WtV
    mu_WtWu=mu*u0
    # print('dtype=',Pt((1.0-mask)*P(u).double()).dtype)
    # print('dtype=',Pt(mask*P(u).double()).dtype)
    # print('dtype=',mu_WtWu.dtype)
    # print('dtype=',rhs_term.dtype)
    residual=alpha*Pt((1.0-mask)*P(u0).double()).double()+gamma*Pt(mask*P(u0).double()).double()+mu_WtWu-rhs_term
    recU=CG_alg_JSR(u0,alpha,gamma, mu,residual,kernel_dec,kernel_rec, P, Pt,mask, CG_Iter)
    return recU

def update_U_mask(Y,u0,P,Pt,mask,torch_W,torch_Wt,v,mu,CG_Iter):
    PtY=Pt(Y.squeeze(1)*mask.squeeze(1)).unsqueeze(1)
    # print('shape of PtY=',PtY.shape)
    # print('shape of v=',v.shape)
    mu_WtV=torch_Wt(mu*v)
    # print('shape of mu_WtV=',mu_WtV.shape)
    rhs_term=PtY+mu_WtV
    mu_WtWu=torch_Wt(mu*torch_W(u0))
    residual=Pt(P(u0.squeeze(1))).unsqueeze(1)+mu_WtWu-rhs_term
    recU=CG_alg_mask(u0, mu,residual,torch_W,torch_Wt, P, Pt,mask, CG_Iter)
    return recU


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
        self.Radon=Radon
        self.iRadon=iRadon
        self.WDec=WDec
        self.WRec=WRec
        self.highpass=highpass
        self.sino_weight=sino_weight
        self.rec_mode=rec_task
        self.sino_mask=sino_mask
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
    # rTr_array=[]
    for k in range(CGiter):
        #print(f"pshape{p.size()}")
        pATApNorm = myAtA.pATAp(p)
        #print(f"pnorm:{pATApNorm}")
        # print('pATApNorm=',pATApNorm)
        mu_pWtWpNorm=myAtA.pWTWp(p,mu)
        # print('mu_pWtWpNorm=',mu_pWtWpNorm)
        rTr=torch.sum(r**2)
        # print('rTr=',rTr)
        alphak = rTr/(mu_pWtWpNorm+pATApNorm)
        # print('\nk={},alphak={}'.format(k,alphak))
        x = x+alphak*p
        # AtAp=back_rec(back_rec(p.squeeze(1),radon),iradon).unsqueeze(1)
        r = r+alphak*myAtA.AtA_WtW_x(p,mu)
        betak = torch.sum(r**2)/ rTr
        # print('k={},betak={}'.format(k,betak))
        p=-r+betak*p
        # rTr_array.append(rTr.cpu().numpy())

    pATApNorm = myAtA.pATAp(p)
    mu_pWtWpNorm=myAtA.pWTWp(p,mu)
    rTr=torch.sum(r**2)
    alphak = rTr/(mu_pWtWpNorm+pATApNorm)
    x = x+alphak*p
    
    # x=torch.clamp(x, min=0.0)
    # print('\nrTr_array=',rTr_array)
    # print('Last step, rTr=',rTr)
    return x

from skimage.measure import compare_ssim as skssim
from skimage.measure import compare_psnr as skpsnr
#from utils import wv_dec, wv_rec

def PDalg_Model(Y,u0,myAtAModule,MaxIter,CG_Iter,mu,beta,thresh_mode,
    stop_criterion='fixed_iter',tol=1e-8):
    # print('[****JSR_Model****]-->shape of u0=',u0.shape)
    u_old=u0
    v=myAtAModule.W(u_old)
    step_list=[]
    ssim_list=[]
    psnr_list=[]
    for ii in range(MaxIter):
        # print('Iteration i=',i)
        if ii>0: 
            u_old=u
        # u=update_U(Y,u_old,P,Pt,torch_W,torch_Wt,v,mu,CG_Iter)
        #print(myAtAModule.BwAt(Y))
        #print(myAtAModule.Wt(v,mu))
        rhs=myAtAModule.BwAt(Y)+myAtAModule.Wt(v,mu)
        Ax0=myAtAModule.AtA_WtW_x(u_old,mu)
        res=Ax0-rhs
        u=CG_alg_new(myAtAModule,u_old,mu,res, CG_Iter)
    # def CG_new(x, mu,rhs,torch_W,torch_Wt, P, Pt, CGiter):
         # CG_alg_new(myAtA,x,mu,res,CGiter=5):
        Wu=myAtAModule.W(u)
        v=ThreshCoeff(Wu,beta,thresh_mode)
        RelErr=norm_l2(u-u_old)/norm_l2(u)
        RelErr=torch.sqrt(RelErr)
        # Wu=wv_dec(u,Dec=Dec)
        # print('\n Before threshold')
        # for kk in range(Wu.shape[1]):
        #     print('kk=',kk,'max v[kk]=',torch.max(Wu[0,kk,]))
        #     print('kk=',kk,'min v[kk]=',torch.min(Wu[0,kk,]))

        # print('\n After threshold')
        # for kk in range(v.shape[1]):
        #     print('kk=',kk,'max v[kk]=',torch.max(v[0,kk,]))
        #     print('kk=',kk,'min v[kk]=',torch.min(v[0,kk,]))
        if stop_criterion=='fixed_iter':
            pass
        else:
            if RelErr<tol:
                print('Adopted the relative error stopping criterion!!!--RelErr=',RelErr)
                break

    return u


def SpBreg_Model(Y,u0,img,P,Pt,torch_W,torch_Wt,MaxIter,CG_Iter,mu,beta,thresh_mode,
    stop_criterion='fixed_iter',tol=1e-8):
    # print('[****JSR_Model****]-->shape of u0=',u0.shape)
    u_old=u0
    Y=Y
    u_GT=img
    # v=wv_dec(u_old,Dec=Dec)
    v=torch_W(u_old)
    dd=torch.zeros_like(v)
    step_list=[]
    ssim_list=[]
    psnr_list=[]
    for i in range(MaxIter):
        # print('Iteration i=',i)
        if i>0: 
            u_old=u

        Wu=torch_W(u_old)
        v=ThreshCoeff(Wu+dd,beta,thresh_mode)
        dd=dd+Wu-v
        u=update_U(Y,u_old,P,Pt,torch_W,torch_Wt,v-dd,mu,CG_Iter)
        ssim_val=ssim(u.squeeze(0).squeeze(0).cpu().numpy(),u_GT.squeeze(0).squeeze(0).cpu().numpy())
        psnr_val=psnr(u.squeeze(0).squeeze(0).cpu().numpy(),u_GT.squeeze(0).squeeze(0).cpu().numpy(),data_range=1.0)
        RelErr=norm_l2(u.cuda()-u_old.cuda())/norm_l2(u.cuda())
        RelErr=torch.sqrt(RelErr)
        print('iter=%d'%i, ', ssim=%.4f' % ssim_val,', psnr=%.4f'% psnr_val,', RelErr=%.10f'% RelErr)

        step_list.append(i)
        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)
        if stop_criterion=='fixed_iter':
            if i==MaxIter-1:
                print('Fixed iteration stopping criterion!!')
            pass
        else:
            if RelErr<tol:
                print('Adopted the relative error stopping criterion!!!--RelErr=',RelErr)
                break

    return u,step_list, ssim_list, psnr_list

def PDalg_mask(Y,u0,img,P,Pt,mask,torch_W,torch_Wt,MaxIter,CG_Iter,mu,beta,thresh_mode,
    stop_criterion='fixed_iter',tol=1e-8):
    u_old=u0
    Y=Y
    u_GT=img
    v=torch_W(u_old)
    step_list=[]
    ssim_list=[]
    psnr_list=[]
    for i in range(MaxIter):
        # print('Iteration i=',i)
        if i>0: 
            u_old=u

        u=update_U_mask(Y,u_old,P,Pt,mask, torch_W,torch_Wt,v,mu,CG_Iter)
        ssim_val=ssim(u.squeeze(0).squeeze(0).cpu().numpy(),u_GT.squeeze(0).squeeze(0).cpu().numpy())
        psnr_val=psnr(u.squeeze(0).squeeze(0).cpu().numpy(),u_GT.squeeze(0).squeeze(0).cpu().numpy(),data_range=1.0)
        RelErr=norm_l2(u.cuda()-u_old.cuda())/norm_l2(u.cuda())
        RelErr=torch.sqrt(RelErr)
        print('iter=%d'%i, ', ssim=%.4f' % ssim_val,', psnr=%.4f'% psnr_val,', RelErr=%.10f'% RelErr)
        Wu=torch_W(u)
        v=ThreshCoeff(Wu,beta,thresh_mode)

        step_list.append(i)
        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)
        if stop_criterion=='fixed_iter':
            pass
        else:
            if RelErr<tol:
                print('Adopted the relative error stopping criterion!!!--RelErr=',RelErr)
                break

    return u,step_list, ssim_list, psnr_list

def JSR_Model_PDalg(Y,u0,img,P,Pt,mask,torch_W,torch_Wt,MaxIter,CG_Iter,alpha,muF,muU,lamF,lamU,thresh_mode,
    stop_criterion='fixed_iter',tol=1e-8):
    # print('[****JSR_Model****]-->shape of u0=',u0.shape)
    u_old=u0
    Y=Y
    u_GT=img
    # f=Y
    f=torch.zeros_like(Y)
    vU=torch_W(u_old)
    vF=torch_W(f,highpass=False)
    step_list=[]
    ssim_list=[]
    psnr_list=[]
    for i in range(MaxIter):
        # print('Iteration i=',i)
        if i>0: 
            u_old=u

        f_tmp=P(u_old)+alpha*mask*Y+muF*torch_Wt(vF,highpass=False)
        f=f_tmp/(1.0+muF+alpha*mask)
        Wf=torch_W(f,highpass=False)
        vF=ThreshCoeff(Wf,lamF,thresh_mode)

        u=update_U(f,u_old,P,Pt,torch_W,torch_Wt,vU,muU,CG_Iter)
        Wu=torch_W(u_old)
        vU=ThreshCoeff(Wu,lamU,thresh_mode)

        ssim_val=ssim(u.squeeze(0).squeeze(0).cpu().numpy(),u_GT.squeeze(0).squeeze(0).cpu().numpy())
        psnr_val=psnr(u.squeeze(0).squeeze(0).cpu().numpy(),u_GT.squeeze(0).squeeze(0).cpu().numpy(),data_range=1.0)
        RelErr=norm_l2(u.cuda()-u_old.cuda())/norm_l2(u.cuda())
        RelErr=torch.sqrt(RelErr)
        print('iter=%d'%i, ', ssim=%.4f' % ssim_val,', psnr=%.4f'% psnr_val,', RelErr=%.10f'% RelErr)

        step_list.append(i)
        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)
        if stop_criterion=='fixed_iter':
            pass
        else:
            if RelErr<tol:
                print('Adopted the relative error stopping criterion!!!--RelErr=',RelErr)
                break

    return u,step_list, ssim_list, psnr_list

def JSR_Model_PDalg_debug(Y,u0,f0,img,P,Pt,mask,kernel_dec,kernel_rec,alpha,beta,gamma,
    muF,muU,lamF,lamU,MaxIter,CG_Iter,
    thresh_mode,stop_criterion,tol):
    # print('[****JSR_Model****]-->shape of u0=',u0.shape)
    u=u0
    u_GT=img
    ff=f0
    # f=torch.zeros_like(Y)
    # print('shape of f=',f.shape)
    vu=torch_W_dec(u,kernel_dec,highpass=False)
    # print('shape of vu=',vu.shape)
    vf=torch_W_dec(ff,kernel_dec,highpass=False)
    # print('shape of vf=',vf.shape)
    bf=torch.zeros_like(vf)
    bu=torch.zeros_like(vu)
    # bf=vf
    # bu=vu
    # print('shape of bf=',bf.shape)
    # print('shape of bu=',bu.shape)
    step_list=[]
    ssim_list=[]
    psnr_list=[]
    RelErr_list=[]
    for i in range(MaxIter):
        # print('Iteration i=',i)
        if i==0:
            u_old=u0
        elif i>0: 
            u_old=u

        f_tmp=alpha*(1.0-mask)*P(u).double()+beta*mask*Y+muF*torch_W_rec(vf-bf,kernel_rec,highpass=False)
        # print('shape of f=',f.shape)
        ff=f_tmp/(alpha*(1.0-mask)+beta*mask+muF)
        Wf=torch_W_dec(ff,kernel_dec,highpass=False)
        # print('shape of Wf=',Wf.shape)
        vf=ThreshCoeff(Wf+bf,lamF,thresh_mode='anisotropic')
        # print('shape of vf=',vf.shape)
        bf=bf+Wf-vf

        u=update_U_JSR(Y,ff,u_old,P,Pt,mask,kernel_dec,kernel_rec,vu-bu,alpha,gamma,muU,CG_Iter)
        # print('shape of u=',u.shape)
        Wu=torch_W_dec(u,kernel_dec,highpass=False)
        vu=ThreshCoeff(Wu+bu,lamU,thresh_mode='isotropic')
        bu=bu+Wu-vu

        ssim_val=ssim(u.squeeze(0).squeeze(0).cpu().numpy(),u_GT.squeeze(0).squeeze(0).cpu().numpy())
        psnr_val=psnr(u.squeeze(0).squeeze(0).cpu().numpy(),u_GT.squeeze(0).squeeze(0).cpu().numpy(),data_range=1.0)
        RelErr=norm_l2(u.cuda()-u_old.cuda())/norm_l2(u.cuda())
        RelErr=torch.sqrt(RelErr)
        print('iter=%d'%i, ', ssim=%.4f' % ssim_val,', psnr=%.4f'% psnr_val,', RelErr=%.10f'% RelErr)

        step_list.append(i)
        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)
        RelErr_list.append(RelErr)
        if stop_criterion=='fixed_iter':
            pass
        else:
            if RelErr<tol:
                print('Adopted the relative error stopping criterion!!!--RelErr=',RelErr)
                break

    return u,step_list, ssim_list, psnr_list, RelErr_list


def generate_sparse_angle_mask(angles,detectors):
# angles=total_angles-available_angles
    detectors=np.array(detectors).astype(int)
    total_angles=np.array(angles).astype(int)
    # data_mask=np.ones((detectors,total_angles))
    data_mask=np.zeros((detectors,total_angles))
    # for i in range(total_angles//2):
    #     data_mask[:,i*2+1]=0.0
    data_mask[:,::2]=1.0
    data_mask = data_mask.transpose()
    data_mask =torch.from_numpy(data_mask).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
    print('#--$--&--'*5+'sino_mask.shape=',data_mask.shape)
    # print('Sum of nonzero elements in the mask=',np.sum(data_mask))
    # print('sum of nonzero elements in the mask=',np.sum(y))
    return data_mask


def JSR_Model_SplitBregmanAlg(Y,u0,img,P,Pt,Dec,MaxIter,CG_Iter,mu,beta,stop_criterion='fixed_iter'):
    print('[****JSR_Model_SplitBregmanAlg****]-->shape of u0=',u0.shape)
    u_old=u0
    Y=Y
    u_GT=img
    # print('[****JSR_Model****]-->shape of v=',v.shape)
    v=wv_dec(u_old,Dec=Dec)
    dk=torch.zeros_like(v)
    print('[****JSR_Model_SplitBregmanAlg****]-->shape of v=',v.shape)
    print('[****JSR_Model_SplitBregmanAlg****]-->shape of dk=',dk.shape)
    # print('[****JSR_Model****]-->img idx=',idx)
    # for kk in range(v.shape[1]):
    #     print('\nkk=',kk,'max v[kk]=',torch.max(v[0,kk,]))
    #     print('kk=',kk,'min v[kk]=',torch.min(v[0,kk,]))
    step_list=[]
    ssim_list=[]
    psnr_list=[]
    RelErr_list=[]
    for i in range(MaxIter):
        # print('Iteration i=',i)
        if i>0: 
            u_old=u
        u=update_U(Y.cuda(),u_old.cuda(),P,Pt,Dec,v-dk,mu,CG_Iter)
        # print('[****JSR_Model****]-->shape of updated u=',u.shape)
        # print('[****JSR_Model****]-->shape of updated img=',u_GT.shape)
        ssim_val=ssim(u.squeeze(0).squeeze(0).cpu().numpy(),u_GT.squeeze(0).squeeze(0).cpu().numpy())
        psnr_val=psnr(u.squeeze(0).squeeze(0).cpu().numpy(),u_GT.squeeze(0).squeeze(0).cpu().numpy(),data_range=1.0)
        RelErr=norm_l2(u.cuda()-u_old.cuda())/norm_l2(u.cuda())
        print('iter=%d'%i, ', ssim=%.4f' % ssim_val,', psnr=%.4f'% psnr_val,', RelErr=%.8f'% RelErr)
        Wu=wv_dec(u,Dec=Dec)
        # print('\n Before threshold')
        # for kk in range(Wu.shape[1]):
        #     print('kk=',kk,'max v[kk]=',torch.max(Wu[0,kk,]))
        #     print('kk=',kk,'min v[kk]=',torch.min(Wu[0,kk,]))

        v=ThreshCoeff(Wu+dk,beta)
        dk=dk+Wu-v
        # print('\n After threshold')
        # for kk in range(v.shape[1]):
        #     print('kk=',kk,'max v[kk]=',torch.max(v[0,kk,]))
        #     print('kk=',kk,'min v[kk]=',torch.min(v[0,kk,]))

        step_list.append(i)
        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)
        RelErr_list.append(RelErr)
        if stop_criterion=='fixed_iter':
            pass
        else:
            if RelErr<1e-8:
                print('Adopted the relative error stopping criterion!!!')
                break


    return u,step_list, ssim_list, psnr_list,RelErr_list



def MeanStd_ssim_psnr_nrmse(img1,img2,crop_img=False,cplen=3):
    from skimage.measure import compare_ssim as skssim
    from skimage.measure import compare_psnr as skpsnr
    from skimage.measure import compare_nrmse as sknrmse
    print('\n---->Compute shape of output rec image=',img1.shape)
    print('num of rec image[0]=',img1.shape[0])
    img1=img1.squeeze(1).numpy()
    img2=img2.squeeze(1).numpy()
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if crop_img:
        img1=img1[...,cplen:-cplen,cplen:-cplen]
        img2=img2[...,cplen:-cplen,cplen:-cplen]
        print('Current cropped img1 size=',np.shape(img1))
        print('Current cropped img2 size=',np.shape(img2))
    ssimArray=[]
    psnrArray=[]
    nrmseArray=[]
    for ii in range(img1.shape[0]):
        #print('\n---->Compute shape of img1[ii,...] image=',img1[ii,...].shape)
        # img_ssim=skssim(img1[ii,...], img2[ii,...])
        # img_ssim=skssim(img1[ii,...], img2[ii,...],use_sample_covariance=False,gaussian_weights=True,
    # win_size=11,data_range=2.0,sigma=1.5)
        img_ssim=np_ms_ssim(img1[ii,...], img2[ii,...], weights=None, mean_metric=True)

        img_psnr=skpsnr(img1[ii,...], img2[ii,...],data_range=1.0)
        img_nrmse=sknrmse(img1[ii,...], img2[ii,...])
        ssimArray.append(img_ssim)
        psnrArray.append(img_psnr)
        nrmseArray.append(img_nrmse)
    #print('ssimArray=',ssimArray)
    mean_ssim=eval('%2.4f'%np.mean(ssimArray))
    mean_psnr=eval('%2.4f'%np.mean(psnrArray))
    mean_nrmse=eval('%2.4f'%np.mean(nrmseArray))
    std_ssim=eval('%2.4f'%np.std(ssimArray))
    std_psnr=eval('%2.4f'%np.std(psnrArray))
    std_nrmse=eval('%2.4f'%np.std(nrmseArray))
    # print('mean of ssimArray={},std={}'.format(mean_ssim,std_ssim))
    return mean_ssim,std_ssim,mean_psnr,std_psnr,mean_nrmse,std_nrmse


    
def limited_angle_mask(angles,detectors,limit_rate=1,limit_range=150,limit_type='interval',shift_pixel=20):
        # angles=np.array(self.args.sino_size[0]).astype(int)
        # detectors=np.array(self.args.sino_size[1]).astype(int)
        mask_tmp=torch.zeros([1,angles,detectors],device='cuda',requires_grad=False)
        # mask_tmp[...,angles//4:(angles*3//4),:]=1.0
        # shift_pixel=20
        if limit_type=='rate':
            mask_tmp[0:1,0+shift_pixel:shift_pixel+angles//limit_rate]=1.0
        elif limit_type=='interval':
            # mask_tmp[0:1,0+shift_pixel:shift_pixel+limit_range]=1.0
            mask_tmp[:,angles//2-limit_range//2:angles//2+limit_range//2,:]=1.0

        sino_mask=mask_tmp
        return sino_mask