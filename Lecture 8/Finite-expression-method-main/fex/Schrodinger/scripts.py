import os
import time
import random


for i in range(2):
    for dim in [6, 12, 18, 24, 30]:
        gpu = random.randint(1,9)
        os.system('screen python controller_cubic_sh_firstdeflation_thenintegral.py --epoch 1000 --bs 10 --greedy 0.1 --gpu '+str(gpu)+' --ckpt Firstdeflat_thenintegral_epoch1k_Dim'+str(dim)+' --tree depth2_sub --random_step 3 --lr 0.002 --dim '+str(dim)+' --base 200000 --left -1 --right 1  --domainbs 2000 --intbs 10000')
        time.sleep(100)