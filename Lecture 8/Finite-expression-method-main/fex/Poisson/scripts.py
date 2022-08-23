import os
import time
import random

for i in range(6):
    for dim in [10, 20, 30, 40, 50]:
        gpu = random.randint(0,4)
        os.system('screen python controller_poisson.py --epoch 1000 --bs 10 --greedy 0.1 --gpu '+str(gpu)+' --ckpt Dim'+str(dim)+' --tree depth2_sub --random_step 3 --lr 0.002 --dim '+str(dim)+' --base 200000 --left -1 --right 1 --domainbs 5000 --bdbs 1000')
        time.sleep(100)