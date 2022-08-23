import os
import time
import random

# for i in range(6):
#     for dim in [10, 20, 30, 40, 50]:
#         gpu = random.randint(1,7)
#         os.system('screen python train.py --gpu-id '+str(gpu)+' --trainbs 5000 --bdbs 1000 --optim adam --lr 0.001 --iters 15000 --dim '+str(dim)+' --checkpoint ckpts/dim'+str(dim)+'_trial'+str(i)+' --weight 100 --left -1 --right 1')
#         # print(name)
#         time.sleep(100)


# for i in range(6):
#     for dim in [10, 20, 30, 40, 50]:
#         gpu = random.randint(0,9)
#         os.system('screen python train.py --gpu-id '+str(gpu)+' --trainbs 10000 --bdbs 5000 --optim adam --lr 0.001 --iters 15000 --dim '+str(dim)+' --checkpoint ckpts_largerbs_relu2/dim'+str(dim)+'_trial'+str(i)+' --weight 100 --left -1 --right 1')
#         # print(name)
#         time.sleep(100)

for i in range(6):
    for dim in [10, 20, 30, 40, 50]:
        gpu = random.randint(0,9)
        os.system('screen python train.py --gpu-id '+str(gpu)+' --trainbs 5000 --bdbs 1000 --optim adam --lr 0.001 --iters 15000 --dim '+str(dim)+' --checkpoint ckpts_bs5k_1k_relu2/dim'+str(dim)+'_trial'+str(i)+' --weight 100 --left -1 --right 1')
        # print(name)
        time.sleep(100)


# for i in range(1):
#     for dim in [30]:
#         gpu = random.randint(1,3)
#         os.system('python train.py --gpu-id '+str(gpu)+' --trainbs 10000 --bdbs 5000 --optim adam --lr 0.001 --iters 15000 --dim '+str(dim)+' --checkpoint ckpts/test --weight 100 --left -1 --right 1')
        # print(name)
        # time.sleep(100)
