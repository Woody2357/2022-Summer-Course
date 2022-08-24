import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.nn import init
import numpy as np
from torch.autograd import Variable
import random
import noise

def masked_softmax(vec, mask, dim=-1):
    masked_vec = vec * mask.float()
    with torch.no_grad():
        tmp_mask=1.0/mask.float()
        # print(tmp_mask)
        masked_max_vec=vec+(mask.float()-tmp_mask*torch.sign(tmp_mask))
        # print(masked_max_vec)
        max_vec = torch.max(masked_max_vec, dim=dim, keepdim=True)[0]
        # print(max_vec)
    # print(max_vec.size())
    exps = torch.exp((masked_vec-max_vec)*mask.float())
    masked_exps = (exps) * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros=(masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps/(masked_sums)

class policy_network(nn.Module):
    def __init__(self,input_num,output_num,n_hid,n_layers,dropout=0.0):
        super(policy_network, self).__init__()
        self.input_num=input_num
        self.output_num=output_num
        self.n_hid=n_hid
        self.n_layers=n_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gru=nn.GRU(n_hid, n_hid, n_layers, dropout=dropout)
        self.linear1=torch.nn.Linear(input_num+1,n_hid)
        self.linear2=torch.nn.Linear(n_hid,n_hid)
        self.linear1_1=torch.nn.Linear(n_hid,n_hid)
        self.linear_out=torch.nn.Linear(n_hid,output_num)
        self.linear1_doze=torch.nn.Linear(n_hid+self.output_num,n_hid)
        self.linear1_1_doze=torch.nn.Linear(n_hid,n_hid)
        self.linear_out_doze=torch.nn.Linear(n_hid,2)
        self.in1=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1=torch.nn.InstanceNorm1d(n_hid)
        self.in2=torch.nn.InstanceNorm1d(n_hid)
        self.in1_doze=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1_doze=torch.nn.InstanceNorm1d(n_hid)
        self.init_weights()
        self.start_act=self.input_num-self.output_num
        self.noise=1e-1

    def init_weights(self):
        torch.nn.init.orthogonal_(self.linear1.weight)
        torch.nn.init.constant_(self.linear1.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear1_1.weight)
        torch.nn.init.constant_(self.linear1_1.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear2.weight)
        torch.nn.init.constant_(self.linear2.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear_out.weight)
        torch.nn.init.constant_(self.linear_out.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear1_doze.weight)
        torch.nn.init.constant_(self.linear1_doze.bias,0.1)
        torch.nn.init.orthogonal_(self.linear_out_doze.weight)
        torch.nn.init.constant_(self.linear_out_doze.bias,0.1)
        torch.nn.init.orthogonal_(self.linear1_1_doze.weight)
        torch.nn.init.constant_(self.linear1_1_doze.bias,0.1)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.1)
            elif 'weight' in name:
                torch.nn.init.orthogonal_(param)
    
    def init_hidden(self, bsz):
        return torch.zeros(self.n_layers, bsz, self.n_hid).to(self.device)


    def forward(self,state,hidden):
        out=torch.relu(self.in1(self.linear1(state)))
        out=torch.relu(self.in1_1(self.linear1_1(out)))
        rnn_out,hidden2=self.gru(out,hidden)
        out=torch.relu(self.in2(self.linear2(rnn_out)))
        out=self.linear_out(out)
        batch_size=out.size()[1]
        # out1=torch.softmax(out,dim=2)
        return out,rnn_out,hidden2

    def forward2(self,out,one_hot):
        input=torch.cat([out,one_hot],dim=2)
        out1=torch.relu(self.in1_doze(self.linear1_doze(input)))
        out1=torch.relu(self.in1_1_doze(self.linear1_1_doze(out1)))
        out1=self.linear_out_doze(out1)
        batch_size=out1.size()[1]
        mean=out1[:,:,0].view(-1,batch_size)
        std=torch.exp(out1[:,:,1])+self.noise
        std=std.view(-1,batch_size)
        return mean,std

    def act(self,state,doze1,hidden,choose_act,deterministic=False):
        out1,rnn_out,_=self.forward(state,hidden)
        mask=torch.zeros(choose_act.size()).to(self.device)
        mask[choose_act<1e-6]=1.0
        out1=masked_softmax(out1,mask)
        dist1=Categorical(out1)
        angle=dist1.sample()
        #print(angle)
        angle_hot=F.one_hot(angle,self.output_num).type_as(rnn_out).detach()
        mean,std=self.forward2(rnn_out,angle_hot)
        normal1 = Normal(mean, std)
        if deterministic:
            angle=torch.argmax(out1,dim=2)
            if torch.cuda.is_available():
                angle=angle.cpu()
                mean=mean.cpu()
            doze=mean
            return angle.numpy()[0,0],doze.numpy()[0,0]

        if torch.cuda.is_available():
            angle=angle.cpu()
            doze=normal1.sample().cpu()
        else:
            doze=normal1.sample()
        return angle.numpy()[0,0],doze.numpy()[0,0]
    
    def batch_sample(self,state,hidden):
        out1,rnn_out,hidden2=self.forward(state,hidden)
        batch_size=out1.size()[1]
        one_hot=state[:,:,self.start_act:-1]
        mask=torch.zeros(one_hot.size()).to(self.device)
        mask[one_hot<1e-6]=1
        # out1=mask*out1+1e-10
        out1=masked_softmax(out1,mask)
        dist1=OneHotCategorical(out1)
        act1=dist1.sample()
        mean,std=self.forward2(rnn_out,act1)
        normal1=Normal(mean,std)
        doze=normal1.sample()
        return act1,doze



    def prob(self,state,hidden,act,doze,rest):
        out1,rnn_out,hidden2=self.forward(state,hidden)
        batch_size=out1.size()[1]
        one_hot=state[:,:,self.start_act:-1]
        mask=torch.zeros(one_hot.size()).to(self.device)
        mask[one_hot<1e-6]=1
        # out1=mask*out1+1e-10
        # p=torch.sum(out1,dim=2).unsqueeze(2)
        # out1=out1/p
        # out1=torch.sum(act*out1,dim=2)
        out1=masked_softmax(out1,mask)
        out1=torch.sum(act*out1,dim=2)
        mean,std=self.forward2(rnn_out,act)
        rest=rest.view(-1,batch_size)
        doze=doze.view(-1,batch_size)
        normal1=Normal(mean,std)
        log_p1=(out1+1e-10).log()
        log_p2=normal1.log_prob(doze).view(-1,batch_size)
        p1=out1
        p2=log_p2.exp()
        H=normal1.entropy()
        return p1,p2,log_p1,log_p2,hidden2,H

class policy_network_seperate(nn.Module):
    def __init__(self,input_num,output_num,n_hid,n_layers,dropout=0.0):
        super(policy_network, self).__init__()
        self.input_num=input_num
        self.output_num=output_num
        self.n_hid=n_hid
        self.n_layers=n_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gru=nn.GRU(n_hid, n_hid, n_layers, dropout=dropout)
        self.linear1=torch.nn.Linear(input_num+1,n_hid)
        self.linear2=torch.nn.Linear(n_hid,n_hid)
        self.linear1_1=torch.nn.Linear(n_hid,n_hid)
        self.linear_out=torch.nn.Linear(n_hid,output_num)
        self.linear1_doze=torch.nn.Linear(n_hid,n_hid)
        self.linear1_1_doze=torch.nn.Linear(n_hid,n_hid)
        self.linear_out_doze=torch.nn.Linear(n_hid,2)
        self.in1=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1=torch.nn.InstanceNorm1d(n_hid)
        self.in2=torch.nn.InstanceNorm1d(n_hid)
        self.in1_doze=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1_doze=torch.nn.InstanceNorm1d(n_hid)
        self.init_weights()
        self.start_act=self.input_num-self.output_num
        self.noise=0.0

    def init_weights(self):
        torch.nn.init.orthogonal_(self.linear1.weight)
        torch.nn.init.constant_(self.linear1.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear1_1.weight)
        torch.nn.init.constant_(self.linear1_1.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear2.weight)
        torch.nn.init.constant_(self.linear2.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear_out.weight)
        torch.nn.init.constant_(self.linear_out.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear1_doze.weight)
        torch.nn.init.constant_(self.linear1_doze.bias,0.1)
        torch.nn.init.orthogonal_(self.linear_out_doze.weight)
        torch.nn.init.constant_(self.linear_out_doze.bias,0.1)
        torch.nn.init.orthogonal_(self.linear1_1_doze.weight)
        torch.nn.init.constant_(self.linear1_1_doze.bias,0.1)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.1)
            elif 'weight' in name:
                torch.nn.init.orthogonal_(param)
    
    def init_hidden(self, bsz):
        return torch.zeros(self.n_layers, bsz, self.n_hid).to(self.device)


    def forward(self,state,hidden):
        out=torch.relu(self.in1(self.linear1(state)))
        out=torch.relu(self.in1_1(self.linear1_1(out)))
        rnn_out,hidden2=self.gru(out,hidden)
        out=torch.relu(self.in2(self.linear2(rnn_out)))
        out=self.linear_out(out)
        batch_size=out.size()[1]
        out1=torch.softmax(out,dim=2)
        return out1,rnn_out,hidden2

    def forward2(self,out,one_hot):
        input=out
        out1=torch.relu(self.in1_doze(self.linear1_doze(input)))
        out1=torch.relu(self.in1_1_doze(self.linear1_1_doze(out1)))
        out1=self.linear_out_doze(out1)
        batch_size=out1.size()[1]
        mean=out1[:,:,0].view(-1,batch_size)
        std=torch.exp(out1[:,:,1])+self.noise
        std=std.view(-1,batch_size)
        return mean,std

    def act(self,state,doze1,hidden,choose_act,deterministic=False):
        out1,rnn_out,_=self.forward(state,hidden)
        mask=torch.zeros(choose_act.size()).to(self.device)
        mask[choose_act<1e-6]=1.0
        out1=mask*out1+1e-8
        dist1=Categorical(out1)
        angle=dist1.sample()
        #print(angle)
        angle_hot=F.one_hot(angle,self.output_num).type_as(rnn_out).detach()
        mean,std=self.forward2(rnn_out,angle_hot)
        normal1 = Normal(mean, std)
        if deterministic:
            angle=torch.argmax(out1,dim=2)
            if torch.cuda.is_available():
                angle=angle.cpu()
                mean=mean.cpu()
            doze=mean
            return angle.numpy()[0,0],doze.numpy()[0,0]

        if torch.cuda.is_available():
            angle=angle.cpu()
            doze=normal1.sample().cpu()
        else:
            doze=normal1.sample()
        return angle.numpy()[0,0],doze.numpy()[0,0]
    
    def batch_sample(self,state,hidden):
        out1,rnn_out,hidden2=self.forward(state,hidden)
        batch_size=out1.size()[1]
        one_hot=state[:,:,self.start_act:-1]
        mask=torch.zeros(one_hot.size()).to(self.device)
        mask[one_hot<1e-6]=1
        out1=mask*out1+1e-10
        dist1=OneHotCategorical(out1)
        act1=dist1.sample()
        mean,std=self.forward2(rnn_out,act1)
        normal1=Normal(mean,std)
        doze=normal1.sample()
        return act1,doze



    def prob(self,state,hidden,act,doze,rest):
        out1,rnn_out,hidden2=self.forward(state,hidden)
        batch_size=out1.size()[1]
        one_hot=state[:,:,self.start_act:-1]
        mask=torch.zeros(one_hot.size()).to(self.device)
        mask[one_hot<1e-6]=1
        out1=mask*out1+1e-10
        p=torch.sum(out1,dim=2).unsqueeze(2)
        out1=out1/p
        out1=torch.sum(act*out1,dim=2)
        mean,std=self.forward2(rnn_out,act)
        rest=rest.view(-1,batch_size)
        doze=doze.view(-1,batch_size)
        normal1=Normal(mean,std)
        log_p1=(out1+1e-10).log()
        log_p2=normal1.log_prob(doze).view(-1,batch_size)
        p1=out1
        p2=log_p2.exp()
        H=normal1.entropy()
        return p1,p2,log_p1,log_p2,hidden2,H


class policy_network_ddpg(nn.Module):
    def __init__(self,input_num,output_num,n_hid,n_layers,dropout=0.0):
        super(policy_network_ddpg, self).__init__()
        self.input_num=input_num
        self.output_num=output_num
        self.n_hid=n_hid
        self.n_layers=n_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gru=nn.GRU(n_hid, n_hid, n_layers, dropout=dropout)
        self.linear1=torch.nn.Linear(input_num+1,n_hid)
        self.linear2=torch.nn.Linear(n_hid,n_hid)
        self.linear1_1=torch.nn.Linear(n_hid,n_hid)
        self.linear_out=torch.nn.Linear(n_hid,output_num)
        self.linear1_doze=torch.nn.Linear(n_hid+self.output_num,n_hid)
        self.linear1_1_doze=torch.nn.Linear(n_hid,n_hid)
        self.linear_out_doze=torch.nn.Linear(n_hid,1)
        self.in1=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1=torch.nn.InstanceNorm1d(n_hid)
        self.in2=torch.nn.InstanceNorm1d(n_hid)
        self.in1_doze=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1_doze=torch.nn.InstanceNorm1d(n_hid)
        self.init_weights()
        self.start_act=self.input_num-self.output_num
        self.noise=0.0

    def init_weights(self):
        torch.nn.init.orthogonal_(self.linear1.weight)
        torch.nn.init.constant_(self.linear1.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear1_1.weight)
        torch.nn.init.constant_(self.linear1_1.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear2.weight)
        torch.nn.init.constant_(self.linear2.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear_out.weight)
        torch.nn.init.constant_(self.linear_out.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear1_doze.weight)
        torch.nn.init.constant_(self.linear1_doze.bias,0.1)
        torch.nn.init.orthogonal_(self.linear_out_doze.weight)
        torch.nn.init.constant_(self.linear_out_doze.bias,0.1)
        torch.nn.init.orthogonal_(self.linear1_1_doze.weight)
        torch.nn.init.constant_(self.linear1_1_doze.bias,0.1)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.1)
            elif 'weight' in name:
                torch.nn.init.orthogonal_(param)
    
    def init_hidden(self, bsz):
        return torch.zeros(self.n_layers, bsz, self.n_hid).to(self.device)


    def forward(self,state,hidden):
        out=torch.relu(self.in1(self.linear1(state)))
        out=torch.relu(self.in1_1(self.linear1_1(out)))
        rnn_out,hidden2=self.gru(out,hidden)
        out=torch.relu(self.in2(self.linear2(rnn_out)))
        out=self.linear_out(out)
        batch_size=out.size()[1]
        out1=torch.softmax(out,dim=2)
        return out1,rnn_out,hidden2

    def forward2(self,out,one_hot):
        input=torch.cat([out,one_hot],dim=2)
        out1=torch.relu(self.in1_doze(self.linear1_doze(input)))
        out1=torch.relu(self.in1_1_doze(self.linear1_1_doze(out1)))
        out1=self.linear_out_doze(out1)
        batch_size=out1.size()[1]
        mean=out1.view(-1,batch_size)
        return mean

    def act(self,state,doze1,hidden,choose_act,deterministic=False):
        out1,rnn_out,_=self.forward(state,hidden)
        mask=torch.zeros(choose_act.size()).to(self.device)
        mask[choose_act<1e-6]=1.0
        out1=mask*out1+1e-8
        dist1=Categorical(out1)
        angle=dist1.sample()
        #print(angle)
        angle_hot=F.one_hot(angle,self.output_num).type_as(rnn_out).detach()
        mean=self.forward2(rnn_out,angle_hot)
        normal1 = Normal(mean,self.noise)
        if deterministic:
            angle=torch.argmax(out1,dim=2)
            if torch.cuda.is_available():
                angle=angle.cpu()
                mean=mean.cpu()
            doze=mean
            return angle.numpy()[0,0],doze.numpy()[0,0]

        if torch.cuda.is_available():
            angle=angle.cpu()
            doze=normal1.sample().cpu()
        else:
            doze=normal1.sample()
        return angle.numpy()[0,0],doze.numpy()[0,0]
    
    def batch_sample(self,state,hidden):
        out1,rnn_out,hidden2=self.forward(state,hidden)
        batch_size=out1.size()[1]
        one_hot=state[:,:,self.start_act:-1]
        mask=torch.zeros(one_hot.size()).to(self.device)
        mask[one_hot<1e-6]=1.0
        out1=mask*out1+1e-8
        logout1=out1.log()
        act1=F.gumbel_softmax(logout1,hard=True)
        #act2=act1.detach()
        act2=act1
        mean=self.forward2(rnn_out,act2)
        return act1,mean

    def prob(self,state,hidden):
        out1,rnn_out,hidden2=self.forward(state,hidden)
        batch_size=out1.size()[1]
        one_hot=state[:,:,self.start_act:-1]
        mask=torch.zeros(one_hot.size()).to(self.device)
        mask[one_hot<1e-6]=1.0
        out1=mask*out1+1e-8
        logout1=out1.log()
        act1=F.gumbel_softmax(logout1,hard=True)
        logp1=act1*F.softmax(logout1,dim=2).log()
        act2=act1.detach()
        mean=self.forward2(rnn_out,act2)
        m1=Normal(mean,self.noise)
        doze=m1.rsample()
        logp2=m1.log_prob(doze)
        return act1,doze,logp1,logp2


class policy_network_sac(nn.Module):
    def __init__(self,input_num,output_num,n_hid,n_layers,dropout=0.0):
        super(policy_network_sac, self).__init__()
        self.input_num=input_num
        self.output_num=output_num
        self.n_hid=n_hid
        self.n_layers=n_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gru=nn.GRU(n_hid, n_hid, n_layers, dropout=dropout)
        self.linear1=torch.nn.Linear(input_num+1,n_hid)
        self.linear2=torch.nn.Linear(n_hid,n_hid)
        self.linear1_1=torch.nn.Linear(n_hid,n_hid)
        self.linear_out=torch.nn.Linear(n_hid,output_num)
        self.linear1_doze=torch.nn.Linear(n_hid+self.output_num,n_hid)
        self.linear1_1_doze=torch.nn.Linear(n_hid,n_hid)
        self.linear_out_doze=torch.nn.Linear(n_hid,2)
        self.in1=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1=torch.nn.InstanceNorm1d(n_hid)
        self.in2=torch.nn.InstanceNorm1d(n_hid)
        self.in1_doze=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1_doze=torch.nn.InstanceNorm1d(n_hid)
        self.init_weights()
        self.start_act=self.input_num-self.output_num
        self.noise=0.0

    def init_weights(self):
        torch.nn.init.orthogonal_(self.linear1.weight)
        torch.nn.init.constant_(self.linear1.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear1_1.weight)
        torch.nn.init.constant_(self.linear1_1.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear2.weight)
        torch.nn.init.constant_(self.linear2.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear_out.weight)
        torch.nn.init.constant_(self.linear_out.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear1_doze.weight)
        torch.nn.init.constant_(self.linear1_doze.bias,0.1)
        torch.nn.init.orthogonal_(self.linear_out_doze.weight)
        torch.nn.init.constant_(self.linear_out_doze.bias,0.1)
        torch.nn.init.orthogonal_(self.linear1_1_doze.weight)
        torch.nn.init.constant_(self.linear1_1_doze.bias,0.1)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.1)
            elif 'weight' in name:
                torch.nn.init.orthogonal_(param)
    
    def init_hidden(self, bsz):
        return torch.zeros(self.n_layers, bsz, self.n_hid).to(self.device)


    def forward(self,state,hidden):
        out=torch.relu(self.in1(self.linear1(state)))
        out=torch.relu(self.in1_1(self.linear1_1(out)))
        rnn_out,hidden2=self.gru(out,hidden)
        out=torch.relu(self.in2(self.linear2(rnn_out)))
        out=self.linear_out(out)
        batch_size=out.size()[1]
        out1=torch.softmax(out,dim=2)
        return out1,rnn_out,hidden2

    def forward2(self,out,one_hot):
        input=torch.cat([out,one_hot],dim=2)
        out1=torch.relu(self.in1_doze(self.linear1_doze(input)))
        out1=torch.relu(self.in1_1_doze(self.linear1_1_doze(out1)))
        out1=self.linear_out_doze(out1)
        batch_size=out1.size()[1]
        mean=out1[:,:,0].view(-1,batch_size)
        std=torch.exp(out1[:,:,1]).view(-1,batch_size)+self.noise
        return mean,std

    def act(self,state,doze1,hidden,choose_act,deterministic=False):
        out1,rnn_out,_=self.forward(state,hidden)
        mask=torch.zeros(choose_act.size()).to(self.device)
        mask[choose_act<1e-6]=1.0
        out1=mask*out1+1e-8
        dist1=Categorical(out1)
        angle=dist1.sample()
        #print(angle)
        angle_hot=F.one_hot(angle,self.output_num).type_as(rnn_out).detach()
        mean,std=self.forward2(rnn_out,angle_hot)
        normal1 = Normal(mean,std)
        if deterministic:
            angle=torch.argmax(out1,dim=2)
            if torch.cuda.is_available():
                angle=angle.cpu()
                mean=mean.cpu()
            doze=mean
            return angle.numpy()[0,0],doze.numpy()[0,0]

        if torch.cuda.is_available():
            angle=angle.cpu()
            doze=normal1.sample().cpu()
        else:
            doze=normal1.sample()
        return angle.numpy()[0,0],doze.numpy()[0,0]
    
    def batch_sample(self,state,hidden):
        out1,rnn_out,hidden2=self.forward(state,hidden)
        batch_size=out1.size()[1]
        one_hot=state[:,:,self.start_act:-1]
        mask=torch.zeros(one_hot.size()).to(self.device)
        mask[one_hot<1e-6]=1.0
        out1=mask*out1+1e-8
        logout1=out1.log()
        act1=F.gumbel_softmax(logout1,hard=True)
        #act2=act1.detach()
        act2=act1
        mean,std=self.forward2(rnn_out,act2)
        m1=Normal(mean,std)
        doze=m1.rsample()
        return act1,doze

    def prob(self,state,hidden):
        out1,rnn_out,hidden2=self.forward(state,hidden)
        batch_size=out1.size()[1]
        one_hot=state[:,:,self.start_act:-1]
        mask=torch.zeros(one_hot.size()).to(self.device)
        mask[one_hot<1e-6]=1.0
        out1=mask*out1+1e-8
        logout1=out1.log()
        act1=F.gumbel_softmax(logout1,hard=True)
        p=torch.sum(out1,dim=2).unsqueeze(2)
        out1=out1/(p+1e-8)
        logp1=torch.log((act1*out1).sum(dim=2)+1e-8)
        act2=act1.detach()
        mean,std=self.forward2(rnn_out,act2)
        m1=Normal(mean,std)
        doze=m1.rsample()
        logp2=m1.log_prob(doze)
        return act1,doze,logp1,logp2




class critic_network(nn.Module):
    def __init__(self,input_num,n_hid,n_layers,dropout=0.0):
        super(critic_network, self).__init__()
        self.input_num=input_num
        self.n_hid=n_hid
        self.n_layers=n_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gru=nn.GRU(n_hid, n_hid, n_layers, dropout=dropout)
        self.linear1=nn.Linear(input_num+1,n_hid)
        self.linear1_1=nn.Linear(n_hid,n_hid)
        self.in1=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1=torch.nn.InstanceNorm1d(n_hid)
        self.in2=torch.nn.InstanceNorm1d(n_hid)
        self.linear2=nn.Linear(n_hid,n_hid)
        self.linear_out=nn.Linear(n_hid,1)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.orthogonal_(self.linear1.weight)
        torch.nn.init.constant_(self.linear1.bias, 0.0)
        torch.nn.init.orthogonal_(self.linear1_1.weight)
        torch.nn.init.constant_(self.linear1_1.bias, 0.0)
        torch.nn.init.orthogonal_(self.linear2.weight)
        torch.nn.init.constant_(self.linear2.bias, 0.0)
        torch.nn.init.orthogonal_(self.linear_out.weight)
        torch.nn.init.constant_(self.linear_out.bias, 0.0)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal_(param)
        
        

    def init_hidden(self, bsz):
        #weight = next(self.parameters())
        return torch.zeros(self.n_layers, bsz, self.n_hid).to(self.device)

    def forward(self,state,hidden):
        #print("state size{}".format(state.size()))
        batch_size=state.size()[1]
        out=torch.relu(self.in1(self.linear1(state)))
        out=torch.relu(self.in1_1(self.linear1_1(out)))
        out,hidden2=self.gru(out,hidden)
        out=torch.relu(self.in2(self.linear2(out)))
        out=self.linear_out(out).view(-1,batch_size)
        return out,hidden2

class critic_network2(nn.Module):
    def __init__(self,input_num,action_dim,n_hid,n_layers,dropout=0.0):
        super(critic_network2, self).__init__()
        self.input_num=input_num
        self.action_dim=action_dim
        self.n_hid=n_hid
        self.n_layers=n_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gru=nn.GRU(n_hid, n_hid, n_layers, dropout=dropout)
        self.linear1=nn.Linear(input_num+1,n_hid)
        self.linear1_1=nn.Linear(n_hid,n_hid)
        self.in1=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1=torch.nn.InstanceNorm1d(n_hid)
        self.in2=torch.nn.InstanceNorm1d(n_hid)
        self.in2_2=torch.nn.InstanceNorm1d(n_hid)
        self.linear2=nn.Linear(n_hid+action_dim+1,n_hid)
        self.linear2_2=nn.Linear(n_hid,n_hid)
        self.linear_out=nn.Linear(n_hid,1)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.orthogonal_(self.linear1.weight)
        torch.nn.init.constant_(self.linear1.bias, 0.0)
        torch.nn.init.orthogonal_(self.linear1_1.weight)
        torch.nn.init.constant_(self.linear1_1.bias, 0.0)
        torch.nn.init.orthogonal_(self.linear2.weight)
        torch.nn.init.constant_(self.linear2.bias, 0.0)
        torch.nn.init.orthogonal_(self.linear2_2.weight)
        torch.nn.init.constant_(self.linear2_2.bias, 0.0)
        torch.nn.init.orthogonal_(self.linear_out.weight)
        torch.nn.init.constant_(self.linear_out.bias, 0.0)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal_(param)
        
        

    def init_hidden(self, bsz):
        #weight = next(self.parameters())
        return torch.zeros(self.n_layers, bsz, self.n_hid).to(self.device)

    def forward(self,state,action,doze,hidden):
        #print("state size{}".format(state.size()))
        batch_size=state.size()[1]
        out=torch.relu(self.in1(self.linear1(state)))
        out=torch.relu(self.in1_1(self.linear1_1(out)))
        out,hidden2=self.gru(out,hidden)
        out=torch.cat([out,action,doze.unsqueeze(2)],dim=2)
        out=torch.relu(self.in2(self.linear2(out)))
        out=torch.relu(self.in2_2(self.linear2_2(out)))
        out=self.linear_out(out).view(-1,batch_size)
        return out,hidden2

        
    
