import os

# # generate the RNN model checkpoint
os.system('python RNNs_last_loss.py --epoch 100 --schedule 40 60 80 90 --lr 0.1  --seq_length 50 --train-batch 500 --checkpoint checkpoint/RNN_train_last_50')

# prediction
os.system('python RNNs_eval.py --seq_length_test 50 --checkpoint RNN_train_last_50')
