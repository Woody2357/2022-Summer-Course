import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
from torch_networks import policy_network as policy_network
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

def run_env(spg,ep):
    spg.start_epoch(1)
    obs,_=spg.env.reset()
    rest_doze=spg.env.rest
    total_reward=0
    total_reward2=0
    total_reward3=0
    I=1
    time_long=0
    act=np.zeros((spg.action_dim,))
    old_act=np.zeros((spg.action_dim,))
    equal=1.0/spg.long_time
    act[0]=equal
    old_act[0]=equal
    RS=[]
    rs=0
    history=[]
    equal=1.0/spg.long_time
    for i in range(spg.long_time):
        obs_norm=np.clip((obs-spg.statestat.mean)/(spg.statestat.std+1e-8),-40,40)
        obs1=np.concatenate([obs_norm,old_act],axis=0)
        angle,doze=spg.action(obs1,rest_doze,old_act)
        doze1=np.tanh(doze)*0.5+0.5
        #use=rest_doze*np.clip(doze,0,1)
        #doze1=(1-equal)*np.clip(doze,0.0,None)+equal*np.clip(doze,None,0.0)+equal
        use=np.clip(doze1,0,rest_doze)
        #use=1/spg.long_time
        rest_doze-=use
        next_obs,reward,done,_=spg.env.step(angle,use,True,alg=spg.args.reconstruct_alg)
        if i==spg.long_time-1:
            #reward=spg.env.show_psnr()
            done=True

        if done:
            reward-=abs(rest_doze)
        time_long+=1
        if spg.args.reward_scale:
            reward1=np.clip(reward/(spg.runningstat.std+1e-8),-40,40)
        else:
            reward1=reward
        act[angle]+=use
        next_obs_norm=np.clip((next_obs-spg.statestat.mean)/(spg.statestat.std+1e-8),-40,40)
        next_obs1=np.concatenate([next_obs_norm,act],axis=0)
        
        spg.next_hidden(obs1,rest_doze+use)
        history.append(((obs1.copy(),rest_doze+use),(angle,doze,reward,obs.copy()),reward1,done,(next_obs1.copy(),rest_doze)))
        old_act[angle]+=use
        total_reward+=(I*reward)
        total_reward2+=reward
        total_reward3+=(I*reward1)
        I*=spg.gamma
        obs=next_obs
        if done:
            break

    repeat=old_act-1
    repeat=repeat.clip(0,200)
    repeat=repeat.sum()
    return history,(total_reward,total_reward2,total_reward3,time_long,spg.policy_net.noise,repeat,spg.env.show_psnr())


class SPG():
    def __init__(self,args,state_dim,action_dim,hidden_cell,n_layers,actor_lr=1e-4,critic_lr=1e-4,gamma=0.99,save_path='../ctmodel/',load_path='../ctmodel/',max_process=2,max_length=50):
        self.args=args
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net=policy_network(self.state_dim+self.action_dim, self.action_dim,hidden_cell[0],n_layers[0]).to(self.device)
        self.critic_net=critic_network(self.state_dim+self.action_dim,self.action_dim,hidden_cell[1],n_layers[1]).to(self.device)
        self.target_policy_net=policy_network(self.state_dim+self.action_dim,self.action_dim,hidden_cell[0],n_layers[0]).to(self.device)
        self.target_critic_net=critic_network(self.state_dim+self.action_dim,self.action_dim,hidden_cell[1],n_layers[1]).to(self.device)
        self.gamma=args.gamma
        self.actor_lr=args.actor_lr
        self.critic_lr=args.critic_lr
        self.batch=args.mini_batch
        self.train_ep=args.train_epoch
        self.num_process=args.num_process
        self.epsilon=args.epsilon
        self.buffer_size=1000
        self.replay_buffer=MemorySlide(self.buffer_size)
        self.long_time=max_length
        self.max_length=max_length
        self.save_path=save_path
        img_path='../ct_data'
        self.env=CT(img_path,have_noise=True)
        self.runningstat=RunningStat(())
        stat_fp=open('./obs_stat.txt','rb')
        self.statestat=pickle.load(stat_fp)
        print(f"gamma:{self.args.gamma}actor lr:{self.actor_lr},critic lr:{self.critic_lr}")

        self.avg_H=[]

        self.policy_net.share_memory()
        self.target_policy_net.share_memory()
        self.critic_net.share_memory()
        self.target_critic_net.share_memory()
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.policy_net.noise=float(args.base_noise)
        self.target_policy_net.noise=float(args.base_noise)
        self.actor_optimizer = torch.optim.Adam(self.policy_net.parameters(), self.actor_lr,betas=(0.9,0.999),amsgrad=True)
        self.critic_optimizer=torch.optim.Adam(self.critic_net.parameters(),self.critic_lr,betas=(0.9,0.999),amsgrad=True)
        if args.use_linear_lr_decay:
            self.actor_scheduler=optim.lr_scheduler.StepLR(self.actor_optimizer,80,0.995)
            self.critic_scheduler=optim.lr_scheduler.StepLR(self.critic_optimizer,80,0.995)

    

    def multi_train(self,all_ep):

        avg_rew=[]
        avg_repeat=[]
        for ep in range(all_ep):
            pools=mp.Pool(processes=self.num_process)
            res=[]
            for i in range(16):
                #self.policy_net.resample()
                history=pools.apply_async(run_env,args=(self,ep))
                res.append(history)
            for x in res:
                history,info=x.get()
                history=self.gain_reward(history)
                avg_rew.append(info[0])
                print(f"epoch {ep}:reward {info[0]},noise level:{info[4]},time length {info[3]},non-discount:{info[1]},scale reward:{info[2]},repeat:{info[5]},psnr:{info[6]}")
                self.replay_buffer.insert(history)
            for i in range(self.train_ep):
                self.train(min(len(self.replay_buffer.buffer),self.batch))
            if ep%10==9:
                self.test_train()
            if ep%10==9:
                print(self.avg_H)
                print(f"avg entropy:{np.mean(np.array(self.avg_H))}")
                self.avg_H=[]
                print(f"avg reward:{np.mean(np.array(avg_rew))}")
                print(f"learning rate:actor:{self.actor_optimizer.param_groups[0]['lr']},critic:{self.critic_optimizer.param_groups[0]['lr']}")
                self.save(self.save_path,ep)
                avg_rew=[]
                avg_repeat=[]
            pools.close()
            pools.join()

    def start_epoch(self,batch_size):
        start_a_hidden=self.policy_net.init_hidden(batch_size)
        start_v_hidden=self.critic_net.init_hidden(batch_size)
        start_a_hidden_t=self.target_policy_net.init_hidden(batch_size)
        start_v_hidden_t=self.target_critic_net.init_hidden(batch_size)
        self.hidden_v_pre=start_v_hidden
        self.hidden_a_pre=start_a_hidden
        self.hidden_a_pre_t=start_a_hidden_t
        self.hidden_v_pre_t=start_v_hidden_t

    def gain_reward(self,history):
        lens=len(history)
        result_history=[]
        gae=0
        rs=0
        j=0
        for i in reversed(range(lens)):
            rs=self.gamma*rs+history[j][1][2]
            j+=1
            self.runningstat.push(rs)
            observe=history[i][1][3].tolist()
            for data in observe:
                self.statestat.push(data)
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
        self.critic_net.train()
        self.target_critic_net.eval()
        #self.policy_net.resample()
        #self.actor_policy_net.resample()

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

        self.critic_optimizer.zero_grad()
        v_pred,_ = self.critic_net(pre_state,action,doze1,self.hidden_v_pre)

        closs = mask*(total_reward - v_pred) ** 2 
        #print(closs)
        closs=closs.sum()/count_eff
        closs.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        self.start_epoch(batch_size)
        q_pred,_=self.critic_net(pre_state,action,doze1,self.hidden_v_pre)
        target_action=action.detach()
        target_doze=doze1.detach()
        q_target=q_pred.detach()
        for i in range(16):
            self.start_epoch(batch_size)
            with torch.no_grad():
                sample_action,sample_doze=self.policy_net.batch_sample(pre_state,self.hidden_a_pre)
                q_tmp,_=self.critic_net(pre_state,sample_action,sample_doze,self.hidden_v_pre)
                target_action=sample_action*(q_tmp>q_target).unsqueeze(2)+target_action*(q_tmp<=q_target).unsqueeze(2)
                target_doze=sample_doze*(q_tmp>q_target)+target_doze*(q_tmp<=q_target)
                q_target=q_tmp*(q_tmp>q_pred)+q_target*(q_tmp<=q_target)
        target_action=target_action.detach()
        target_doze=target_doze.detach()
        q_target=q_target.detach()
        #advantage_mean=(advantage*mask).sum()/count_eff
        #advantage_std=torch.sqrt((mask*(advantage-advantage_mean)).pow(2).sum()/count_eff)
        #advantage=(advantage-advantage_mean)/(advantage_std+1e-8)
        _,_,logp1,logp2,_,H= self.policy_net.prob(pre_state,self.hidden_a_pre,target_action,target_doze,pre_doze)
        adv=(q_target-q_pred).detach()
        ploss=mask*(logp1+logp2)*adv

        #-0.02*(p1_old.log()+p2_old.log())
        ploss=-ploss.sum()/count_eff
        ploss.backward()
        with torch.no_grad():
            H=((-logp1-logp2)*mask).sum()/count_eff
            if torch.cuda.is_available():
                H=H.cpu()
            H=H.numpy()
            self.avg_H.append(H)
        if self.args.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),self.args.max_grad_norm)
        self.actor_optimizer.step()


        if self.args.use_linear_lr_decay:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
            

        self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())

        

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
        self.critic_net.eval()
        
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
        torch.save(self.critic_net.state_dict(), save_path + '{}_critic_AC.txt'.format(ep))
        torch.save(self.actor_optimizer.state_dict(),save_path+'{}_actor_optim.txt'.format(ep))
        torch.save(self.critic_optimizer.state_dict(),save_path+'{}_critic_optim.txt'.format(ep))
        stat_fp=open('./obs_stat.txt','wb')
        pickle.dump(self.statestat,stat_fp)

    def load(self, load_path):
        self.policy_net.load_state_dict(torch.load(load_path + '_actor_AC.txt',map_location=self.device))
        self.target_critic_net.load_state_dict(torch.load(load_path + '_critic_AC.txt',map_location=self.device))
        self.target_policy_net.load_state_dict(torch.load(load_path + '_actor_AC.txt',map_location=self.device))
        self.critic_net.load_state_dict(torch.load(load_path + '_critic_AC.txt',map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(load_path + '_actor_optim.txt'))
        self.critic_optimizer.load_state_dict(torch.load(load_path + '_critic_optim.txt'))


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