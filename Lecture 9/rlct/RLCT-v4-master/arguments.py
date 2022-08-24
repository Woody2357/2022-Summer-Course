import argparse

def get_args():
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument(
        '--actor-lr',default=4e-4,type=float,help='actor learning rate')

    parser.add_argument(
        '--critic-lr',default=4e-4,type=float,help='critic learning rate'
    )
    parser.add_argument(
        '--gamma',default=1.0,type=float,help='RL discount')

    parser.add_argument(
        '--gae-lambda',default=0.96,help='GAE lambda')

    parser.add_argument(
        '--value-clip',default=False,type=bool,help='use value clip')

    parser.add_argument(
        '--use-grad-clip',default=False,type=bool,help='use gradient clip')
    parser.add_argument(
        '--max-grad-norm',default=1.0,help='gradient clip')

    parser.add_argument(
        '--use-linear-lr-decay',default=False,type=bool,help='use linear learning rate decay')

    parser.add_argument(
        '--lr-decay-rate',default=0.995,type=float,help='lr decay rate'
    )
    parser.add_argument(
        '--epsilon',default=0.2,help='clip ppo probability ratio')
    parser.add_argument(
        '--reconstruct-alg',default='SART',help='reconstruct algorithm in environment')
    parser.add_argument(
        '--reward-scale',default=False,type=bool,help='use reward scale')
    parser.add_argument(
        '--mini-batch',default=16,type=int,help='mini batch')
    parser.add_argument(
        '--train-epoch',default=20,type=int,help='train in every step')
    parser.add_argument(
        '--num-process',default=8,type=int,help='number of process when simulate environment')
    parser.add_argument(
        '--mode',default='train',help='train or test'
    )
    parser.add_argument(
        '--base-noise',default=0.0,type=float,help='the base noise on network std'
    )
    parser.add_argument(
        '--RL-alg',default="PPO",help='the default PPO，have PPO,SPG'
    )
    parser.add_argument(
        '--pretrain',default=False,type=bool,help='use pretrain method for this method'
    )
    parser.add_argument(
        '--noise-decay',default=0.9,type=float,help='decay rate of '
    )
    parser.add_argument(
        '--tau',default=0.05,type=float,help='soft update rate'
    )
    parser.add_argument(
        '--dropout',default=0.0,type=float,help='network dropout'
    )
    parser.add_argument(
        '--save-path',default='../ctmodel',help='save path'
    )
    parser.add_argument(
        '--load-path',default=None,help='load model path'
    )
    parser.add_argument(
        '--entropy-coef',default=0.0,type=float,help='use entropy'
    )
    parser.add_argument(
        '--adv-scale',default=False,type=bool,help='use advantage value normalize'
    )
    args = parser.parse_args()
    return args