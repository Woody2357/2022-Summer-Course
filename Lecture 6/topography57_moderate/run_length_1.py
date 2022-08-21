import os

# # generate the RNN model checkpoint
os.system('python RNNs_last_loss_no_average.py --epoch 100 --schedule 40 60 80 90 --lr 0.1  --seq_length 20 --train-batch 500 --checkpoint checkpoint/RNN_train_last_20_no_average')

# prediction
os.system('python RNNs_eval.py --seq_length_test 20 --checkpoint RNN_train_last_20_no_average')
