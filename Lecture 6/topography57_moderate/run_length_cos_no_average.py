import os

# # generate the RNN model checkpoint
os.system('python RNNs_last_loss_no_average_coslr.py --epoch 200 --schedule 40 60 80 90 --lr 0.1  --seq_length 1 --train-batch 500 --checkpoint checkpoint/RNN_train_last_1_coslr_no_average')

# prediction
os.system('python RNNs_eval.py --seq_length_test 5 --checkpoint RNN_train_last_1_coslr_no_average')

# # generate the RNN model checkpoint
os.system('python RNNs_last_loss_no_average_coslr.py --epoch 200 --schedule 40 60 80 90 --lr 0.1  --seq_length 15 --train-batch 500 --checkpoint checkpoint/RNN_train_last_15_coslr_no_average')

# prediction
os.system('python RNNs_eval.py --seq_length_test 15 --checkpoint RNN_train_last_15_coslr_no_average')