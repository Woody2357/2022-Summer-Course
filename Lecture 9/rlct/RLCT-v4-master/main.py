from arguments import get_args
import ppo
import ppo_rb
import spg
import ddpg
import sac
import torch.multiprocessing as mp
def main():
    args = get_args()
    print(args)
    if args.RL_alg=='PPO':
        agent=ppo.PPO(args,800,180,(256,64),(3,2),args.actor_lr,args.critic_lr,args.gamma,save_path=args.save_path,load_path='')
    elif args.RL_alg=='PPO-RB':
        agent=ppo_rb.PPO_RB(args,800,360,(256,64),(3,2),args.actor_lr,args.critic_lr,args.gamma,save_path=args.save_path,load_path='')
    elif args.RL_alg=='SPG':
        agent=spg.SPG(args,512,180,(256,64),(3,2),args.actor_lr,args.critic_lr,args.gamma,save_path=args.save_path,load_path='')
    elif args.RL_alg=='DDPG':
        agent=ddpg.DDPG(args,512,180,(256,256),(3,2),args.actor_lr,args.critic_lr,args.gamma,save_path=args.save_path,load_path='')
    elif args.RL_alg=='SAC':
        agent=sac.SAC(args,512,180,(256,256),(3,2),args.actor_lr,args.critic_lr,args.gamma,save_path=args.save_path,load_path='')
    if args.load_path is not None:
        agent.load(args.load_path)

    agent.multi_train(100000)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
