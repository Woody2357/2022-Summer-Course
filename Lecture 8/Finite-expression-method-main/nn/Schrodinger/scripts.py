import os
import time
import random


for i in range(0, 6):
    for dim in [6, 12, 18, 24, 30]:
        gpu = random.choice([3,4,7])
        os.system('screen python train_integral.py --gpu-id '+str(gpu)+' --trainbs 2000 --intbs 10000 --optim adam --lr 0.001 --iters 15000 --dim '+str(dim)+' --checkpoint ckpts_ReLU2_intbs/Dim'+str(dim)+'_trial'+str(i)+' --weight 1 --left -1 --right 1')
        time.sleep(120)