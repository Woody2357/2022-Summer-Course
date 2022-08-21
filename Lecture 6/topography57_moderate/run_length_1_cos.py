import os

# # generate the RNN model checkpoint
os.system('python RNNs_last_loss_coslr.py --epoch 200 --schedule 40 60 80 90 --lr 0.1  --seq_length 20 --train-batch 500 --checkpoint checkpoint/RNN_train_last_20_coslr')

# prediction
os.system('python RNNs_eval.py --seq_length_test 20 --checkpoint RNN_train_last_20_coslr')
