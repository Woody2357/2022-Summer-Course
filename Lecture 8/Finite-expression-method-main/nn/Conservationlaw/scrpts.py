import os
import time
import random

for i in range(6):
    for dim in [6, 12, 18, 24, 30]:
        gpu = random.randint(1,9)
        os.system('screen python train.py --gpu-id '+str(gpu)+' --trainbs 5000 --bdbs 1000 --optim adam --lr 0.001 --iters 15000 --dim '+str(dim)+' --checkpoint ckpts_ReLU_pidiv4/dim'+str(dim)+'_trial'+str(i)+' --weight 100 --left -1 --right 1')
        time.sleep(200)