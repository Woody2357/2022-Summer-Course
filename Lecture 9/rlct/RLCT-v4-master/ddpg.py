import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
from torch_networks import policy_network_ddpg as policy_network
from torch_networks import critic_network2 as critic_network
import torch.nn.functional as F
import copy
import torch.multiprocessing as mp
from env.CT_env_intensity import CT
import random
from utils import MemorySlide
import matplotlib.pyplot as plt
#import ct_animation2
from utils import RunningStat
import pickle

def run_env(ddpg,ep):
    ddpg.start_epoch(1)
    obs,_=ddpg.env.reset()
    rest_doze=ddpg.env.rest
    total_reward=0
    total_reward2=0
    total_reward3=0
    I=1
    time_long=0
    act=np.zeros((ddpg.action_dim,))
    old_act=np.zeros((ddpg.action_dim,))
    equal=1.0/ddpg.long_time
    act[0]=equal
    old_act[0]=equal
    RS=[]
    rs=0
    history=[]
    equal=1.0/ddpg.long_time
    for i in range(ddpg.long_time):
        obs_norm=np.clip((obs-ddpg.statestat.mean)/(ddpg.statestat.std+1e-8),-40,40)
        obs1=np.concatenate([obs_norm,old_act],axis=0)
        angle,doze=ddpg.action(obs1,rest_doze,old_act)
        doze1=np.tanh(doze)*0.5+0.5
        #use=rest_doze*np.clip(doze,0,1)
        #doze1=(1-equal)*np.clip(doze,0.0,None)+equal*np.clip(doze,None,0.0)+equal
        use=np.clip(doze1,0,rest_doze)
        #use=1/ddpg.long_time
        rest_doze-=use
        next_obs,reward,done,_=ddpg.env.step(angle,use,True,alg=ddpg.args.reconstruct_alg)
        if i==ddpg.long_time-1:
            #reward=ddpg.env.show_psnr()
            done=True

        if done:
            reward-=abs(rest_doze)
        time_long+=1
        if ddpg.args.reward_scale:
            reward1=np.clip(reward/(ddpg.runningstat.std+1e-8),-40,40)
        else:
            reward1=reward
        act[angle]+=use
        next_obs_norm=np.clip((next_obs-ddpg.statestat.mean)/(ddpg.statestat.std+1e-8),-40,40)
        next_obs1=np.concatenate([next_obs_norm,act],axis=0)
        
        ddpg.next_hidden(obs1,rest_doze+use)
        history.append(((obs1.copy(),rest_doze+use),(angle,doze,reward,obs.copy()),reward1,done,(next_obs1.copy(),rest_doze)))
        old_act[angle]+=use
        total_reward+=(I*reward)
        total_reward2+=reward
        total_reward3+=(I*reward1)
        I*=ddpg.gamma
        obs=next_obs
        if done:
            break

    repeat=old_act-1
    repeat=repeat.clip(0,200)
    repeat=repeat.sum()
    return history,(total_reward,total_reward2,total_reward3,time_long,ddpg.policy_net.noise,repeat,ddpg.env.show_psnr())

def expert_env(ddpg,ep):
    ddpg.start_epoch(1)
    obs,_=ddpg.env.reset()
    rest_doze=ddpg.env.rest
    total_reward=0
    total_reward2=0
    total_reward3=0
    I=1
    time_long=0
    act=np.zeros((ddpg.action_dim,))
    old_act=np.zeros((ddpg.action_dim,))
    equal=1.0/ddpg.long_time
    act[0]=equal
    old_act[0]=equal
    RS=[]
    rs=0
    history=[]
    equal=1.0/ddpg.long_time
    for i in range(ddpg.long_time):
        obs_norm=np.clip((obs-ddpg.statestat.mean)/(ddpg.statestat.std+1e-8),-40,40)
        obs1=np.concatenate([obs_norm,old_act],axis=0)
        angle,doze=ddpg.action(obs1,rest_doze,old_act)
        #use=rest_doze*np.clip(doze,0,1)
        #doze1=(1-equal)*np.clip(doze,0.0,None)+equal*np.clip(doze,None,0.0)+equal
        doze=equal
        use=doze
        #use=1/ddpg.long_time
        rest_doze-=use
        next_obs,reward,done,_=ddpg.env.step(angle,use,True,alg=ddpg.args.reconstruct_alg)
        if i==ddpg.long_time-1:
            #reward=ddpg.env.show_psnr()
            done=True

        if done:
            reward-=abs(rest_doze)
        time_long+=1
        if ddpg.args.reward_scale:
            reward1=np.clip(reward/(ddpg.runningstat.std+1e-8),-40,40)
        else:
            reward1=reward
        act[angle]+=use
        next_obs_norm=np.clip((next_obs-ddpg.statestat.mean)/(ddpg.statestat.std+1e-8),-40,40)
        next_obs1=np.concatenate([next_obs_norm,act],axis=0)
        
        ddpg.next_hidden(obs1,rest_doze+use)
        history.append(((obs1.copy(),rest_doze+use),(angle,doze,reward,obs.copy()),reward1,done,(next_obs1.copy(),rest_doze)))
        old_act[angle]+=use
        total_reward+=(I*reward)
        total_reward2+=reward
        total_reward3+=(I*reward1)
        I*=ddpg.gamma
        obs=next_obs
        if done:
            break

    repeat=old_act-1
    repeat=repeat.clip(0,200)
    repeat=repeat.sum()
    return history,(total_reward,total_reward2,total_reward3,time_long,ddpg.policy_net.noise,repeat,ddpg.env.show_psnr())


class DDPG():
    def __init__(self,args,state_dim,action_dim,hidden_cell,n_layers,actor_lr=1e-4,critic_lr=1e-4,gamma=0.99,save_path='../ctmodel/',load_path='../ctmodel/',max_process=2,max_length=50):
        self.args=args
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net=policy_network(self.state_dim+self.action_dim, self.action_dim,hidden_cell[0],n_layers[0],dropout=self.args.dropout).to(self.device)
        self.target_policy_net=policy_network(self.state_dim+self.action_dim,self.action_dim,hidden_cell[0],n_layers[0],dropout=self.args.dropout).to(self.device)

        self.critic_net1=critic_network(self.state_dim+self.action_dim,self.action_dim,hidden_cell[1],n_layers[1]).to(self.device)
        self.critic_net2=critic_network(self.state_dim+self.action_dim,self.action_dim,hidden_cell[1],n_layers[1]).to(self.device)
        self.target_critic_net1=critic_network(self.state_dim+self.action_dim,self.action_dim,hidden_cell[1],n_layers[1]).to(self.device)
        self.target_critic_net2=critic_network(self.state_dim+self.action_dim,self.action_dim,hidden_cell[1],n_layers[1]).to(self.device)
        self.gamma=args.gamma
        self.actor_lr=args.actor_lr
        self.critic_lr=args.critic_lr
        self.batch=args.mini_batch
        self.train_ep=args.train_epoch
        self.num_process=args.num_process
        self.epsilon=args.epsilon
        self.buffer_size=10000
        self.expert_buffer_size=200
        self.replay_buffer=MemorySlide(self.buffer_size)
        self.expert_buffer=MemorySlide(self.expert_buffer_size)
        self.long_time=max_length
        self.max_length=max_length
        self.save_path=save_path
        self.delay=2
        self.warmup=5
        self.tau=self.args.tau
        img_path='../ct_data'
        self.env=CT(img_path,have_noise=True)
        self.runningstat=RunningStat(())
        stat_fp=open('./obs_stat.txt','rb')
        self.statestat=pickle.load(stat_fp)
        print(f"gamma:{self.args.gamma}actor lr:{self.actor_lr},critic lr:{self.critic_lr}")

        self.avg_H=[]

        self.policy_net.share_memory()
        self.target_policy_net.share_memory()
        self.critic_net1.share_memory()
        self.target_critic_net1.share_memory()
        self.critic_net2.share_memory()
        self.target_critic_net2.share_memory()
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
        self.target_critic_net2.load_state_dict(self.critic_net2.state_dict()) 
        self.policy_net.noise=args.base_noise
        self.target_policy_net.noise=args.base_noise
        self.actor_optimizer = torch.optim.Adam(self.policy_net.parameters(), self.actor_lr,betas=(0.5,0.999))
        self.critic_optimizer1=torch.optim.Adam(self.critic_net1.parameters(),self.critic_lr,betas=(0.5,0.999))
        self.critic_optimizer2=torch.optim.Adam(self.critic_net2.parameters(),self.critic_lr,betas=(0.5,0.999))
        if args.use_linear_lr_decay:
            self.actor_scheduler=optim.lr_scheduler.StepLR(self.actor_optimizer,160,self.args.lr_decay_rate)
            self.critic_scheduler1=optim.lr_scheduler.StepLR(self.critic_optimizer1,160*self.delay,self.args.lr_decay_rate)
            self.critic_scheduler2=optim.lr_scheduler.StepLR(self.critic_optimizer2,160*self.delay,self.args.lr_decay_rate)

    

    def multi_train(self,all_ep):

        avg_rew=[]
        avg_repeat=[]
        self.ep=0
        for ep in range(all_ep):
            pools=mp.Pool(processes=self.num_process)
            res=[]
            for i in range(8):
                #self.policy_net.resample()
                if ep%10==9:
                    history=pools.apply_async(expert_env,args=(self,ep))
                else:
                    history=pools.apply_async(run_env,args=(self,ep))
                res.append(history)
            for x in res:
                history,info=x.get()
                history=self.gain_reward(history)
                avg_rew.append(info[0])
                print(f"epoch {ep}:reward {info[0]},noise level:{info[4]},time length {info[3]},non-discount:{info[1]},scale reward:{info[2]},repeat:{info[5]},psnr:{info[6]}")
                self.replay_buffer.insert(history)
            pools.close()
            pools.join()
            if self.args.pretrain:
                pools=mp.Pool(processes=self.num_process)
                res=[]
                for i in range(8):
                    history=pools.apply_async(expert_env,args=(self,ep))
                    res.append(history)
                for x in res:
                    history,info=x.get()
                    self.expert_buffer.insert(history)
                pools.close()
                pools.join()

            if ep>self.warmup:
                for i in range(self.train_ep):
                    self.train(min(len(self.replay_buffer.buffer),self.batch))
                    self.ep+=1
            if ep%10==9:
                self.test_train()
            if ep%10==9:
                self.policy_net.noise*=self.args.noise_decay
                self.target_policy_net.noise*=self.args.noise_decay
                self.policy_net.noise=np.clip(self.policy_net.noise,1e-4,None)
                self.target_policy_net.noise=np.clip(self.target_policy_net.noise,1e-4,None)
                print(f"avg reward:{np.mean(np.array(avg_rew))}")
                print(f"learning rate:actor:{self.actor_optimizer.param_groups[0]['lr']},critic:{self.critic_optimizer1.param_groups[0]['lr']}")
                self.save(self.save_path,ep)
                avg_rew=[]
                avg_repeat=[]
            if ep>10:
                self.args.pretrain=False

    def start_epoch(self,batch_size):
        start_a_hidden=self.policy_net.init_hidden(batch_size)
        start_q1_hidden=self.critic_net1.init_hidden(batch_size)
        start_q2_hidden=self.critic_net2.init_hidden(batch_size)
        start_a_hidden_t=self.target_policy_net.init_hidden(batch_size)
        start_q1_hidden_t=self.target_critic_net1.init_hidden(batch_size)
        start_q2_hidden_t=self.target_critic_net2.init_hidden(batch_size)
        self.hidden_q1_pre=start_q1_hidden
        self.hidden_q2_pre=start_q2_hidden
        self.hidden_a_pre=start_a_hidden
        self.hidden_a_pre_t=start_a_hidden_t
        self.hidden_q1_pre_t=start_q1_hidden_t
        self.hidden_q2_pre_t=start_q2_hidden_t

    def gain_reward(self,history):
        lens=len(history)
        result_history=[]
        gae=0
        rs=0
        j=0
        for i in reversed(range(lens)):
            rs=self.gamma*rs+history[i][1][2]
            j+=1
            self.runningstat.push(rs)
            observe=history[i][1][3].tolist()
            #for data in observe:
            #    self.statestat.push(data)
            gae=history[i][2]+self.gamma*gae
            result_history.append((history[i][0],(history[i][1][0],history[i][1][1],gae,history[i][1][3]),history[i][2],history[i][3],history[i][4]))
        #print(result_history)
        result_history.reverse()
        #print(result_history)
        return result_history

    def next_hidden(self,s,doze):
        s = torch.tensor(s, dtype=torch.float,device=self.device).view(-1,1,self.state_dim+self.action_dim)
        doze=torch.tensor([doze],dtype=torch.float,device=self.device).view(-1,1,1)
        state=torch.cat([s,doze],dim=2)
        with torch.no_grad():
            _,_,self.hidden_a_pre=self.policy_net(state,self.hidden_a_pre)
        return self.hidden_a_pre.clone()


    def train(self,batch_size):
        self.policy_net.train()
        self.critic_net1.train()
        self.target_policy_net.eval()
        self.target_critic_net1.eval()
        self.target_critic_net2.eval()
        #self.policy_net.resample()
        #self.actor_policy_net.resample()

        #######noraml ddpg part########
        batch_train=self.replay_buffer.sample(batch_size)
        self.start_epoch(batch_size)
        pre_obs_batch=[]
        pre_doze_batch=[]
        next_obs_batch=[]
        next_doze_batch=[]
        action_batch=[]
        if_end_batch=[]
        reward_batch=[]
        mask_batch=[]
        count_eff=0
        total_reward_batch=[]
        prop_batch=[]
        for i in range(self.max_length+3):
            mask=torch.ones(batch_size,1)
            pre_obs=[]
            pre_doze=[]
            next_obs=[]
            next_doze=[]
            reward=[]
            action=torch.zeros(batch_size,self.action_dim)
            prop=[]
            if_end=[]
            total_reward=[]
            for j in range(batch_size):
                if i<len(batch_train[j]):
                    pre_obs.append(batch_train[j][i][0][0])
                    pre_doze.append(batch_train[j][i][0][1])
                    next_obs.append(batch_train[j][i][4][0])
                    next_doze.append(batch_train[j][i][4][1])
                    action[j,batch_train[j][i][1][0]]=1
                    prop.append(batch_train[j][i][1][1])
                    reward.append(batch_train[j][i][2])
                    total_reward.append(batch_train[j][i][1][2])
                    if batch_train[j][i][3]:
                        if_end.append(0.0)
                    else:
                        if_end.append(1.0)
                    count_eff+=1
                else:
                    pre_obs.append(np.zeros(self.state_dim+self.action_dim))
                    pre_doze.append(0.0)
                    next_obs.append(np.zeros(self.state_dim+self.action_dim))
                    next_doze.append(0.0)
                    action[j,0]=1
                    prop.append(0.0)
                    reward.append(0.0)
                    if_end.append(0.0)
                    total_reward.append(0.0)
                    mask[j,0]=0
            mask_batch.append(mask.tolist())
            pre_obs_batch.append(pre_obs)
            pre_doze_batch.append(pre_doze)
            total_reward_batch.append(total_reward)
            #print(total_reward)
            if i==0:
                next_obs_batch.append(pre_obs)
                next_obs_batch.append(next_obs)
                next_doze_batch.append(pre_doze)
                next_doze_batch.append(next_doze)
            else:
                next_obs_batch.append(next_obs)
                next_doze_batch.append(next_doze)
            action_batch.append(action.tolist())
            if_end_batch.append(if_end)
            reward_batch.append(reward)
            prop_batch.append(prop)
        pre_obs=torch.tensor(pre_obs_batch,dtype=torch.float,device=self.device)
        pre_doze=torch.tensor(pre_doze_batch,dtype=torch.float,device=self.device).view(-1,batch_size,1)
        pre_state=torch.cat([pre_obs,pre_doze],dim=2)
        next_obs=torch.tensor(next_obs_batch,dtype=torch.float,device=self.device)
        next_doze=torch.tensor(next_doze_batch,dtype=torch.float,device=self.device).view(-1,batch_size,1)
        next_state=torch.cat([next_obs,next_doze],dim=2)
        action=torch.tensor(action_batch,dtype=torch.float,device=self.device).view(-1,batch_size,self.action_dim)
        doze1=torch.tensor(prop_batch,dtype=torch.float,device=self.device).view(-1,batch_size)
        if_end=torch.tensor(if_end_batch,dtype=torch.float,device=self.device)
        reward=torch.tensor(reward_batch,dtype=torch.float,device=self.device)
        #print(log_pi_old)
        total_reward=torch.tensor(total_reward_batch,dtype=torch.float,device=self.device).view(-1,batch_size)
        mask=torch.tensor(mask_batch,dtype=torch.float,device=self.device).view(-1,batch_size)

        next_action,next_doze=self.target_policy_net.batch_sample(next_state,self.hidden_a_pre_t)
        next_doze=next_doze+(0.1*torch.randn(next_doze.size()).to(self.device)).clamp(min=-0.2,max=0.2)
        ####pretrain####
        if self.args.pretrain:
            exp_size=min(len(self.expert_buffer.buffer),batch_size)
            batch_train=self.expert_buffer.sample(exp_size)
            exp_pre_obs_batch=[]
            exp_pre_doze_batch=[]
            exp_next_obs_batch=[]
            exp_next_doze_batch=[]
            exp_action_batch=[]
            exp_if_end_batch=[]
            exp_reward_batch=[]
            exp_mask_batch=[]
            exp_count_eff=0
            exp_total_reward_batch=[]
            exp_prop_batch=[]
            for i in range(self.max_length+3):
                exp_mask=torch.ones(batch_size,1)
                exp_pre_obs=[]
                exp_pre_doze=[]
                exp_next_obs=[]
                exp_next_doze=[]
                exp_reward=[]
                exp_action=torch.zeros(batch_size,self.action_dim)
                exp_prop=[]
                exp_if_end=[]
                exp_total_reward=[]
                for j in range(batch_size):
                    if i<len(batch_train[j]):
                        exp_pre_obs.append(batch_train[j][i][0][0])
                        exp_pre_doze.append(batch_train[j][i][0][1])
                        exp_next_obs.append(batch_train[j][i][4][0])
                        exp_next_doze.append(batch_train[j][i][4][1])
                        exp_action[j,batch_train[j][i][1][0]]=1
                        exp_prop.append(batch_train[j][i][1][1])
                        exp_reward.append(batch_train[j][i][2])
                        exp_total_reward.append(batch_train[j][i][1][2])
                        if batch_train[j][i][3]:
                            exp_if_end.append(0.0)
                        else:
                            exp_if_end.append(1.0)
                        exp_count_eff+=1
                    else:
                        exp_pre_obs.append(np.zeros(self.state_dim+self.action_dim))
                        exp_pre_doze.append(0.0)
                        exp_next_obs.append(np.zeros(self.state_dim+self.action_dim))
                        exp_next_doze.append(0.0)
                        exp_action[j,0]=1
                        exp_prop.append(0.0)
                        exp_reward.append(0.0)
                        exp_if_end.append(0.0)
                        exp_total_reward.append(0.0)
                        exp_mask[j,0]=0
                exp_mask_batch.append(exp_mask.tolist())
                exp_pre_obs_batch.append(exp_pre_obs)
                exp_pre_doze_batch.append(exp_pre_doze)
                exp_total_reward_batch.append(exp_total_reward)
            #print(total_reward)
                if i==0:
                    exp_next_obs_batch.append(exp_pre_obs)
                    exp_next_obs_batch.append(exp_next_obs)
                    exp_next_doze_batch.append(exp_pre_doze)
                    exp_next_doze_batch.append(exp_next_doze)
                else:
                    exp_next_obs_batch.append(exp_next_obs)
                    exp_next_doze_batch.append(exp_next_doze)
                exp_action_batch.append(exp_action.tolist())
                exp_if_end_batch.append(exp_if_end)
                exp_reward_batch.append(exp_reward)
                exp_prop_batch.append(exp_prop)
            exp_pre_obs=torch.tensor(exp_pre_obs_batch,dtype=torch.float,device=self.device)
            exp_pre_doze=torch.tensor(exp_pre_doze_batch,dtype=torch.float,device=self.device).view(-1,batch_size,1)
            exp_pre_state=torch.cat([exp_pre_obs,exp_pre_doze],dim=2)
            exp_next_obs=torch.tensor(exp_next_obs_batch,dtype=torch.float,device=self.device)
            exp_next_doze=torch.tensor(exp_next_doze_batch,dtype=torch.float,device=self.device).view(-1,batch_size,1)
            exp_next_state=torch.cat([exp_next_obs,exp_next_doze],dim=2)
            exp_action=torch.tensor(exp_action_batch,dtype=torch.float,device=self.device).view(-1,batch_size,self.action_dim)
            exp_doze1=torch.tensor(exp_prop_batch,dtype=torch.float,device=self.device).view(-1,batch_size)
            exp_if_end=torch.tensor(exp_if_end_batch,dtype=torch.float,device=self.device)
            exp_reward=torch.tensor(exp_reward_batch,dtype=torch.float,device=self.device)
            exp_total_reward=torch.tensor(exp_total_reward_batch,dtype=torch.float,device=self.device).view(-1,batch_size)
            exp_mask=torch.tensor(exp_mask_batch,dtype=torch.float,device=self.device).view(-1,batch_size)

        with torch.no_grad():
            q1,_=self.target_critic_net1(next_state,next_action,next_doze,self.hidden_q1_pre_t)
            q2,_=self.target_critic_net2(next_state,next_action,next_doze,self.hidden_q2_pre_t)
            q1_target=q1[1:,:]
            q2_target=q2[1:,:]
            q_target=reward+if_end*self.gamma*torch.min(q1_target,q2_target)
            q_target=q_target.detach()

        if self.args.pretrain:
            self.start_epoch(exp_size)
            with torch.no_grad():
                now_action,now_doze=self.policy_net.batch_sample(exp_pre_state,self.hidden_a_pre)
            q1_exp,_=self.critic_net1(exp_pre_state,exp_action,exp_doze1,self.hidden_q1_pre_t)
            q1_now,_=self.critic_net1(exp_pre_state,now_action,now_doze,self.hidden_q1_pre)
            q2_exp,_=self.critic_net2(exp_pre_state,exp_action,exp_doze1,self.hidden_q2_pre_t)
            q2_now,_=self.critic_net2(exp_pre_state,now_action,now_doze,self.hidden_q2_pre)
            adv1=q1_exp-q1_now
            adv2=q2_exp-q2_now
            adv1=(adv1*exp_mask).sum()/exp_count_eff
            adv2=(adv2*exp_mask).sum()/exp_count_eff
            adv1=adv1.clamp(min=0.0)
            adv2=adv2.clamp(min=0.0)

        
        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        q1_pred,_ = self.critic_net1(pre_state,action,doze1,self.hidden_q1_pre)
        q2_pred,_ = self.critic_net2(pre_state,action,doze1,self.hidden_q2_pre)
        closs1=mask*(q1_pred-q_target)**2
        closs2=mask*(q2_pred-q_target)**2
        closs1=closs1.sum()/count_eff
        closs2=closs2.sum()/count_eff
        if self.args.pretrain:
            closs1-=adv1
            closs2-=adv2
        closs1.backward()
        closs2.backward()
        #print(closs)
        self.critic_optimizer1.step()
        self.critic_optimizer2.step()

        if self.ep%self.delay==0:
            self.actor_optimizer.zero_grad()
            self.start_epoch(batch_size)
            pred_action,pred_doze=self.policy_net.batch_sample(pre_state,self.hidden_a_pre)
            q1_pred,_=self.critic_net1(pre_state,pred_action,pred_doze,self.hidden_q1_pre)
            ploss=mask*q1_pred

            if self.args.pretrain:
                self.start_epoch(exp_size)
                now_action,now_doze=self.policy_net.batch_sample(exp_pre_state,self.hidden_a_pre)
                q1_exp,_=self.critic_net1(exp_pre_state,exp_action,exp_doze1,self.hidden_q1_pre_t)
                q1_now,_=self.critic_net1(exp_pre_state,now_action,now_doze,self.hidden_q1_pre)
                exploss=(mask*(q1_exp-q1_now)).sum()/exp_count_eff

            #-0.02*(p1_old.log()+p2_old.log())
            ploss=-ploss.sum()/count_eff
            if self.args.pretrain:
                ploss+=exploss
            ploss.backward()
            if self.args.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),self.args.max_grad_norm)
            self.actor_optimizer.step()
            if self.args.use_linear_lr_decay:
                self.actor_scheduler.step()
            self.soft_update(self.target_policy_net,self.policy_net,self.tau)
            self.soft_update(self.target_critic_net1,self.critic_net1,self.tau)
            self.soft_update(self.target_critic_net2,self.critic_net2,self.tau)


        if self.args.use_linear_lr_decay:
            self.critic_scheduler1.step()
            self.critic_scheduler2.step()


    def soft_update(self, target, source, tau):
        '''
        soft copy source param to target.
        '''
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        

    def test_train(self,deterministic=True):
        self.env.reset()
        self.start_epoch(1)
        obs,_=self.env.reset()
        true_img=self.env.true_img
        obs,_=self.env.reset(set_pic=true_img)
        self.env.rest=1.0
        rest_doze=1.0
        total_reward=0
        total_reward2=0
        I=1
        time_long=0
        act=np.zeros((self.action_dim,))
        old_act=np.zeros((self.action_dim,))
        equal=1.0/self.long_time
        act[0]=equal
        old_act[0]=equal
        result=[]
        equal=1/self.long_time
        for i in range(self.long_time):
            obs_norm=np.clip((obs-self.statestat.mean)/(self.statestat.std+1e-8),-40,40)
            obs1=np.concatenate([obs_norm,old_act],axis=0)
            angle,doze=self.action(obs1,rest_doze,old_act,deterministic)
            doze1=np.tanh(doze)*0.5+0.5
            #doze1=(1-equal)*np.clip(doze,0.0,None)+equal*np.clip(doze,None,0.0)+equal
            #use=rest_doze*np.clip(doze,0,1)
            use=np.clip(doze1,0,rest_doze)
            #if i==self.long_time-1:
            #    use=rest_doze
            rest_doze-=use
            next_obs,reward,done,_=self.env.step(angle,use,True,alg='SART')
            act[angle]+=use
            self.next_hidden(obs1,rest_doze+use)
            if i==(self.long_time-1):
                done=True
            old_act[angle]+=use
            total_reward+=(I*reward)
            total_reward2+=reward
            I*=self.gamma
            obs=next_obs
            time_long+=1
            repeat=old_act-1
            repeat=repeat.clip(0,200)
            repeat=repeat.sum()
            if done:
                break


        result.append(self.env.show_psnr())

        obs,_=self.env.reset(set_pic=true_img)
        self.env.rest=1.0
        rest_doze=1.0
        total_reward=0
        total_reward2=0
        I=1
        time_long=0
        act=np.zeros((self.action_dim,))
        old_act=np.zeros((self.action_dim,))
        equal=1.0/self.long_time
        act[0]=equal
        old_act[0]=equal
        self.start_epoch(1)
        for i in range(self.long_time):
            obs_norm=np.clip((obs-self.statestat.mean)/(self.statestat.std+1e-8),-40,40)
            obs1=np.concatenate([obs_norm,old_act],axis=0)
            angle,doze=self.action(obs1,rest_doze,old_act,deterministic)
            use=1/self.long_time
            rest_doze-=use
            next_obs,reward,done,_=self.env.step(angle,use)
            act[angle]+=use
            next_obs1=np.concatenate([next_obs,act],axis=0)
            self.next_hidden(obs1,rest_doze+use)
            if i==(self.long_time-1):
                done=True
            old_act[angle]+=use
            total_reward+=(I*reward)
            total_reward2+=reward
            I*=self.gamma
            obs=next_obs
            time_long+=1
            repeat=old_act-1
            repeat=repeat.clip(0,200)
            repeat=repeat.sum()
            if done:
                break
        result.append(self.env.show_psnr())
        self.start_epoch(1)
        print(f"test result both doze and angle :{result[0]},only angle:{result[1]}")
    
    def action(self,s,doze,choose_act, deterministic = False):
        '''
        use the actor_policy_net to compute the action.

        s: (np.ndarray, batch_size x state_channel x num_x) the input state  
        deterministic: (bool) if False, add exploration noise to the actions. default False. 
        '''
        if deterministic:
            self.policy_net.eval()
        else:
            self.policy_net.train()
        
        s = torch.tensor(s, dtype=torch.float,device=self.device).view(-1,1,self.state_dim+self.action_dim)
        doze=torch.tensor([doze],dtype=torch.float,device=self.device).view(-1,1,1)
        state=torch.cat([s,doze],dim=2)
        choose_act=torch.tensor(choose_act,dtype=torch.float,device=self.device)
        with torch.no_grad():
            angle,doze= self.policy_net.act(state,doze.view(-1,1),self.hidden_a_pre,choose_act,deterministic)
        return angle,doze

    def save(self, save_path,ep):
        print(f"save as {save_path}{ep}")
        torch.save(self.policy_net.state_dict(), save_path + '{}_actor_AC.txt'.format(ep) )
        torch.save(self.critic_net1.state_dict(), save_path + '{}_critic1_AC.txt'.format(ep))
        torch.save(self.critic_net2.state_dict(), save_path + '{}_critic2_AC.txt'.format(ep))
        torch.save(self.actor_optimizer.state_dict(),save_path+'{}_actor_optim.txt'.format(ep))
        torch.save(self.critic_optimizer1.state_dict(),save_path+'{}_critic1_optim.txt'.format(ep))
        torch.save(self.critic_optimizer2.state_dict(),save_path+'{}_critic2_optim.txt'.format(ep))
        stat_fp=open('./obs_stat.txt','wb')
        pickle.dump(self.statestat,stat_fp)
        stat_fp.close()
        if self.args.reward_scale:
            reward_fp=open('./reward_stat.txt','wb')
            pickle.dump(self.runningstat,reward_fp)
            reward_fp.close()

    def load(self, load_path):
        self.policy_net.load_state_dict(torch.load(load_path + '_actor_AC.txt',map_location=self.device))
        self.target_critic_net1.load_state_dict(torch.load(load_path + '_critic1_AC.txt',map_location=self.device))
        self.target_critic_net2.load_state_dict(torch.load(load_path + '_critic2_AC.txt',map_location=self.device))
        self.target_policy_net.load_state_dict(torch.load(load_path + '_actor_AC.txt',map_location=self.device))
        self.critic_net1.load_state_dict(torch.load(load_path + '_critic1_AC.txt',map_location=self.device))
        self.critic_net2.load_state_dict(torch.load(load_path + '_critic2_AC.txt',map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(load_path + '_actor_optim.txt'))
        self.critic_optimizer1.load_state_dict(torch.load(load_path + '_critic1_optim.txt'))
        self.critic_optimizer2.load_state_dict(torch.load(load_path + '_critic2_optim.txt'))


    def test(self,deterministic=True):
        #print(self.actor_policy_net.linear1.weight_sigma)
        #self.actor_policy_net.eval()
        img_path='../ct_data'
        self.env=CT(img_path,have_noise=True)
        self.start_epoch(1)
        obs,_=self.env.reset()
        #true_img=self.env.img_data[200]
        true_img=self.env.true_img
        obs,_=self.env.reset(set_pic=true_img)
        self.env.rest=self.env.rest
        rest_doze=1.
        total_reward=0
        total_reward2=0
        I=1
        time_long=0
        act=np.zeros((self.action_dim,))
        old_act=np.zeros((self.action_dim,))
        equal=1.0/self.long_time
        act[0]=equal
        old_act[0]=equal
        self.long_time=50
        all_psnr_res=[]
        all_doze_res=[]
        all_pic_res=[]
        all_angle_res=[]
        ### RL SART result#####
        doze_accum=0
        tmp_psnr=[]
        tmp_doze=[]
        tmp_pic=[]
        tmp_angle=[]
        equal=1.0/self.long_time
        equal_time=self.long_time
            #print(f"obs:{obs}")
        for i in range(self.long_time):
            obs_norm=np.clip((obs-self.statestat.mean)/(self.statestat.std+1e-8),-40,40)
            obs1=np.concatenate([obs_norm,old_act],axis=0)
            angle,doze=self.action(obs1,rest_doze,old_act,deterministic=True)
            doze1=np.tanh(doze)*0.5+0.5
            #doze1=(1-equal)*np.clip(doze,0.0,None)+equal*np.clip(doze,None,0.0)+equal
            use=np.clip(doze1,0,rest_doze)
            #if i==self.long_time-1:
            #    use=rest_doze
            rest_doze-=use
            doze_accum+=use
            tmp_doze.append(doze_accum)
            next_obs,reward,done,_=self.env.step(angle,use,True,alg='SART')
            tmp_pic.append(self.env.state.copy())
            tmp_psnr.append(self.env.show_psnr())
            tmp_angle.append(angle)
            print(f"action {angle}:use:{use}:{rest_doze}:{reward}, psnr:{self.env.show_psnr()}")
            act[angle]+=use
            next_obs1=np.concatenate([next_obs,act],axis=0)
            self.next_hidden(obs1,rest_doze+use)
            if i==(self.long_time-1):
                done=True
            old_act[angle]+=use
            total_reward+=(I*reward)
            total_reward2+=reward
            I*=self.gamma
            obs=next_obs
            time_long+=1
            repeat=old_act-1
            repeat=repeat.clip(0,200)
            repeat=repeat.sum()
            if done:
                break 
        all_doze_res.append(tmp_doze)
        all_pic_res.append(tmp_pic)
        all_psnr_res.append(tmp_psnr)
        all_angle_res.append(tmp_angle)
        print(f"reward:{total_reward},non-discount reward:{total_reward2},repeat:{repeat},psnr:{self.env.show_psnr()}")
        #equal_time=time_long
        ###RL SART equal
        self.start_epoch(1)
        obs,_=self.env.reset(set_pic=true_img)
        self.env.rest=1.0
        rest_doze=self.env.rest
        total_reward=0
        total_reward2=0
        I=1
        time_long=0
        doze_accum=0
        act=np.zeros((self.action_dim,))
        old_act=np.zeros((self.action_dim,))
        act[0]=equal
        old_act[0]=equal
        tmp_psnr=[]
        tmp_doze=[]
        tmp_pic=[]
        tmp_angle=[]
        self.start_epoch(1)
        for i in range(self.long_time):
            obs_norm=np.clip((obs-self.statestat.mean)/(self.statestat.std+1e-8),-40,40)
            obs1=np.concatenate([obs_norm,old_act],axis=0) 
            angle,doze=self.action(obs1,rest_doze,old_act,True)
            use=1/equal_time
            doze=use/rest_doze
            rest_doze-=use
            doze_accum+=use
            tmp_doze.append(doze_accum)
            next_obs,reward,done,_=self.env.step(angle,use)
            tmp_pic.append(self.env.state.copy())
            tmp_psnr.append(self.env.show_psnr())
            tmp_angle.append(angle)
            print(f"action {angle}:use:{use}:{rest_doze}:{reward}, psnr:{self.env.show_psnr()}")
            act[angle]+=use
            next_obs1=np.concatenate([next_obs,act],axis=0)
            self.next_hidden(obs1,rest_doze+use)
            if i==(self.long_time-1):
                done=True
            old_act[angle]+=use
            total_reward+=(I*reward)
            total_reward2+=reward
            I*=self.gamma
            obs=next_obs
            time_long+=1
            repeat=old_act-1
            repeat=repeat.clip(0,200)
            repeat=repeat.sum()
            if done:
                break
        all_pic_res.append(tmp_pic)
        all_doze_res.append(tmp_doze)
        all_psnr_res.append(tmp_psnr)
        all_angle_res.append(tmp_angle)
        hist_img=self.env.doze_hist()
        plt.figure()
        plt.imshow(hist_img)
        plt.savefig('../result/doze_hist_EQ.png',bbox_inches='tight')
        plt.figure()
        count_num=hist_img.reshape(-1).tolist()
        count_img=[]
        max_pix=0
        for c in count_num:
            if c>10:
                count_img.append(c)
            if c>max_pix:
                max_pix=c
        plt.hist(np.array(count_img)/max_pix,1000)
        plt.savefig('../result/histomgraph_EQ.png',bbox_inches='tight')
        print(f"reward:{total_reward},non-discount reward:{total_reward2},repeat:{repeat},psnr:{self.env.show_psnr()}")
        print(f"reward:{total_reward},non-discount reward:{total_reward2},repeat:{repeat},psnr:{self.env.show_psnr()}")


        ###### RL Wavelet #####
        self.start_epoch(1)
        obs,_=self.env.reset(set_pic=true_img)
        rest_doze=self.env.rest
        total_reward=0
        total_reward2=0
        I=1
        time_long=0
        doze_accum=0
        act=np.zeros((self.action_dim,))
        old_act=np.zeros((self.action_dim,))
        act[0]=1
        old_act[0]=1
        tmp_psnr=[]
        tmp_doze=[]
        tmp_pic=[]
        tmp_angle=[]
        self.start_epoch(1)
        for i in range(self.long_time):
            obs_norm=np.clip((obs-self.statestat.mean)/(self.statestat.std+1e-8),-40,40)
            obs1=np.concatenate([obs_norm,old_act],axis=0)
            angle,doze=self.action(obs1,rest_doze,old_act,True)
            use=1/equal_time
            #use=np.clip(doze,0,rest_doze)
            rest_doze-=use
            doze_accum+=use
            tmp_doze.append(doze_accum)
            next_obs,reward,done,_=self.env.step(angle,use,True,alg='Wavelet')
            tmp_pic.append(self.env.state.copy())
            tmp_psnr.append(self.env.show_psnr())
            tmp_angle.append(angle)
            print(f"action {angle}:use:{use}:{rest_doze}:{reward}, psnr:{self.env.show_psnr()}")
            act[angle]+=1
            next_obs1=np.concatenate([next_obs,act],axis=0)
            self.next_hidden(obs1,rest_doze+use)
            if i==(self.long_time-1):
                done=True
            old_act[angle]+=1
            total_reward+=(I*reward)
            total_reward2+=reward
            I*=self.gamma
            obs=next_obs
            time_long+=1
            repeat=old_act-1
            repeat=repeat.clip(0,200)
            repeat=repeat.sum()
            if done:
                break
        all_doze_res.append(tmp_doze)
        all_pic_res.append(tmp_pic)
        all_psnr_res.append(tmp_psnr)
        all_angle_res.append(tmp_angle)
        
        print(f"reward:{total_reward},non-discount reward:{total_reward2},repeat:{repeat},psnr:{self.env.show_psnr()}")

        ###### RL TVmode#####
        self.start_epoch(1)
        obs,_=self.env.reset(set_pic=true_img)
        rest_doze=self.env.rest
        total_reward=0
        total_reward2=0
        I=1
        time_long=0
        doze_accum=0
        act=np.zeros((self.action_dim,))
        old_act=np.zeros((self.action_dim,))
        act[0]=1
        old_act[0]=1
        tmp_psnr=[]
        tmp_doze=[]
        tmp_pic=[]
        tmp_angle=[]
        self.start_epoch(1)
        for i in range(equal_time):
            obs_norm=np.clip((obs-self.statestat.mean)/(self.statestat.std+1e-8),-40,40)
            obs1=np.concatenate([obs_norm,old_act],axis=0)
            angle,doze=self.action(obs1,rest_doze,old_act,True)
            doze1=np.tanh(doze)*0.5+0.5
            use=np.clip(doze1,0,rest_doze)
            use=1/equal_time
            rest_doze-=use
            doze_accum+=use
            tmp_doze.append(doze_accum)
            next_obs,reward,done,_=self.env.step(angle,use,True,'TVmodel')
            tmp_pic.append(self.env.state.copy())
            tmp_psnr.append(self.env.show_psnr())
            tmp_angle.append(angle)
            print(f"action {angle}:use:{use}:{rest_doze}:{reward}, psnr:{self.env.show_psnr()}")
            act[angle]+=1
            next_obs1=np.concatenate([next_obs,act],axis=0)
            self.next_hidden(obs1,rest_doze+use)
            if i==(self.long_time-1):
                done=True
            old_act[angle]+=1
            total_reward+=(I*reward)
            total_reward2+=reward
            I*=self.gamma
            obs=next_obs
            time_long+=1
            repeat=old_act-1
            repeat=repeat.clip(0,200)
            repeat=repeat.sum()
            if done:
                break
        all_doze_res.append(tmp_doze)
        all_pic_res.append(tmp_pic)
        all_psnr_res.append(tmp_psnr) 
        all_angle_res.append(tmp_angle)
        print(f"reward:{total_reward},non-discount reward:{total_reward2},repeat:{repeat},psnr:{self.env.show_psnr()}")


        ######other method
        obs,_=self.env.reset(set_pic=true_img)
        rest_doze=self.env.rest
        total_reward=0
        total_reward2=0
        I=1
        time_long=0
        angle=list(range(180))
        random.shuffle(angle)
        other_doze_list=[]
        doze_accum=0
        tmp_psnr=[]
        tmp_doze=[]
        tmp_pic=[]
        tmp_angle=[]
        for i in range(equal_time):
            choose_angle=angle[i]
            choose_use=1/equal_time
            doze_accum+=choose_use
            tmp_doze.append(doze_accum)
            rest_doze-=choose_use
            print(f"action {choose_angle}:use:{use}:{rest_doze}:{reward}, psnr:{self.env.show_psnr()}")
            next_obs,reward,done,_=self.env.step(choose_angle,choose_use)
            tmp_pic.append(self.env.state.copy())
            tmp_psnr.append(self.env.show_psnr())
            tmp_angle.append(choose_angle)
            total_reward+=(I*reward)
            total_reward2+=(reward)
            I=I*self.gamma
        all_doze_res.append(tmp_doze)
        all_pic_res.append(tmp_pic)
        all_psnr_res.append(tmp_psnr)
        all_angle_res.append(tmp_angle)
        print(f"reward:{total_reward},non-discount reward:{total_reward2},repeat:{repeat},psnr:{self.env.show_psnr()}")

        obs,_=self.env.reset(set_pic=true_img)
        rest_doze=self.env.rest
        total_reward=0
        total_reward2=0
        I=1
        time_long=0
        angle=list(range(180))
        random.shuffle(angle)
        other_doze_list=[]
        doze_accum=0
        tmp_psnr=[]
        tmp_doze=[]
        tmp_pic=[]
        tmp_angle=[]
        for i in range(equal_time):
            choose_angle=angle[i]
            choose_use=1/equal_time
            doze_accum+=choose_use
            tmp_doze.append(doze_accum)
            rest_doze-=choose_use
            print(f"action {choose_angle}:use:{use}:{rest_doze}:{reward}, psnr:{self.env.show_psnr()}")
            next_obs,reward,done,_=self.env.step(choose_angle,choose_use,True,alg='Wavelet')
            tmp_pic.append(self.env.state.copy())
            tmp_psnr.append(self.env.show_psnr())
            tmp_angle.append(choose_angle)
            total_reward+=(I*reward)
            total_reward2+=(reward)
            I=I*self.gamma
        all_doze_res.append(tmp_doze)
        all_pic_res.append(tmp_pic)
        all_psnr_res.append(tmp_psnr)
        all_angle_res.append(tmp_angle)


        obs,_=self.env.reset(set_pic=true_img)
        rest_doze=self.env.rest
        total_reward=0
        total_reward2=0
        I=1
        time_long=0
        angle=list(range(180))
        random.shuffle(angle)
        other_doze_list=[]
        doze_accum=0
        tmp_psnr=[]
        tmp_doze=[]
        tmp_pic=[]
        tmp_angle=[]
        for i in range(equal_time):
            choose_angle=angle[i]
            choose_use=1/equal_time
            doze_accum+=choose_use
            tmp_doze.append(doze_accum)
            rest-=choose_use
            print(f"action {choose_angle}:use:{use}:{rest_doze}:{reward}, psnr:{self.env.show_psnr()}")
            next_obs,reward,done,_=self.env.step(choose_angle,choose_use,True,'TVmodel')
            tmp_pic.append(self.env.state.copy())
            tmp_psnr.append(self.env.show_psnr())
            tmp_angle.append(choose_angle)
            total_reward+=(I*reward)
            total_reward2+=(reward)
            I=I*self.gamma
        all_doze_res.append(tmp_doze)
        all_pic_res.append(tmp_pic)
        all_psnr_res.append(tmp_psnr)
        all_angle_res.append(tmp_angle)
        print(f"reward:{total_reward},non-discount reward:{total_reward2},repeat:{repeat},psnr:{self.env.show_psnr()}")

        self.start_epoch(1)
        obs,_=self.env.reset(set_pic=true_img)
        rest_doze=1.
        total_reward=0
        total_reward2=0
        I=1
        time_long=0
        doze_accum=0
        act=np.zeros((self.action_dim,))
        old_act=np.zeros((self.action_dim,))
        act[0]=1
        old_act[0]=1
        tmp_psnr=[]
        tmp_doze=[]
        tmp_pic=[]
        tmp_angle=[]
        self.start_epoch(1)
        print(self.long_time)
        for i in range(self.long_time):
            obs_norm=np.clip((obs-self.statestat.mean)/(self.statestat.std+1e-8),-40,40)
            obs1=np.concatenate([obs_norm,old_act],axis=0)
            angle,doze=self.action(obs1,rest_doze,old_act,True)
            doze1=np.tanh(doze)*0.5+0.5
            use=np.clip(doze1,0,rest_doze)
            rest_doze-=use
            doze_accum+=use
            tmp_doze.append(doze_accum)
            next_obs,reward,done,_=self.env.step(angle,use,True,'Wavelet')
            tmp_pic.append(self.env.state.copy())
            tmp_psnr.append(self.env.show_psnr())
            tmp_angle.append(angle)
            print(f"action {angle}:use:{use}:{rest_doze}:{reward}, psnr:{self.env.show_psnr()}")
            act[angle]+=1
            next_obs1=np.concatenate([next_obs,act],axis=0)
            self.next_hidden(obs1,rest_doze+use)
            if i==(self.long_time-1):
                done=True
            old_act[angle]+=1
            total_reward+=(I*reward)
            total_reward2+=reward
            I*=self.gamma
            obs=next_obs
            time_long+=1
            repeat=old_act-1
            repeat=repeat.clip(0,200)
            repeat=repeat.sum()
            if done:
                break
        all_pic_res.append(tmp_pic)
        all_doze_res.append(tmp_doze)
        all_psnr_res.append(tmp_psnr)
        all_angle_res.append(tmp_angle)
        print(f"reward:{total_reward},non-discount reward:{total_reward2},repeat:{repeat},psnr:{self.env.show_psnr()}")

        self.start_epoch(1)
        obs,_=self.env.reset(set_pic=true_img)
        rest_doze=1.
        total_reward=0
        total_reward2=0
        I=1
        time_long=0
        doze_accum=0
        act=np.zeros((self.action_dim,))
        old_act=np.zeros((self.action_dim,))
        act[0]=1
        old_act[0]=1
        tmp_psnr=[]
        tmp_doze=[]
        tmp_pic=[]
        tmp_angle=[]
        self.start_epoch(1)
        for i in range(self.long_time):
            obs_norm=np.clip((obs-self.statestat.mean)/(self.statestat.std+1e-8),-40,40)
            obs1=np.concatenate([obs_norm,old_act],axis=0)
            angle,doze=self.action(obs1,rest_doze,old_act,True)
            doze1=np.tanh(doze)*0.5+0.5
            use=np.clip(doze1,0,rest_doze)
            rest_doze-=use
            doze_accum+=use
            tmp_doze.append(doze_accum)
            next_obs,reward,done,_=self.env.step(angle,use,True,'TVmodel')
            tmp_pic.append(self.env.state.copy())
            tmp_psnr.append(self.env.show_psnr())
            tmp_angle.append(angle)
            print(f"action {angle}:use:{use}:{rest_doze}:{reward}, psnr:{self.env.show_psnr()}")
            act[angle]+=1
            next_obs1=np.concatenate([next_obs,act],axis=0)
            self.next_hidden(obs1,rest_doze+use)
            if i==(self.long_time-1):
                done=True
            old_act[angle]+=1
            total_reward+=(I*reward)
            total_reward2+=reward
            I*=self.gamma
            obs=next_obs
            time_long+=1
            repeat=old_act-1
            repeat=repeat.clip(0,200)
            repeat=repeat.sum()
            if done:
                break
        all_pic_res.append(tmp_pic)
        all_doze_res.append(tmp_doze)
        all_psnr_res.append(tmp_psnr)
        all_angle_res.append(tmp_angle)
        print(f"reward:{total_reward},non-discount reward:{total_reward2},repeat:{repeat},psnr:{self.env.show_psnr()}")
        
        doze_fp=open('../result/doze.txt','wb')
        pic_fp=open('../result/pic.txt','wb')
        psnr_fp=open('../result/psnr.txt','wb')
        img_fp=open('../result/use_img.txt','wb')
        angle_fp=open('../result/angle.txt','wb')
        pickle.dump(all_doze_res,doze_fp)
        pickle.dump(all_pic_res,pic_fp)
        pickle.dump(all_psnr_res,psnr_fp)
        pickle.dump(true_img,img_fp)
        pickle.dump(all_angle_res,angle_fp)
        doze_fp.close()
        pic_fp.close()
        psnr_fp.close()
        img_fp.close()
        print(f"total discount reward:{total_reward},nondiscount reward:{total_reward2},psnr:{self.env.show_psnr()}")
