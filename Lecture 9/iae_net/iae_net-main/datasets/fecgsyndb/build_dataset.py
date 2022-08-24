import os

import librosa
import wfdb
import numpy as np
from scipy import io

def fecgsyndbSetup(inputPath,
                   outputPath):

    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)

    if not os.path.exists(os.path.join(outputPath, "train", "mix")):
        os.makedirs(os.path.join(outputPath, "train", "mix"))

    if not os.path.exists(os.path.join(outputPath, "train", "s1")):
        os.makedirs(os.path.join(outputPath, "train", "s1"))

    if not os.path.exists(os.path.join(outputPath, "train", "s2")):
        os.makedirs(os.path.join(outputPath, "train", "s2"))

    if not os.path.exists(os.path.join(outputPath, "test", "mix")):
        os.makedirs(os.path.join(outputPath, "test", "mix"))

    if not os.path.exists(os.path.join(outputPath, "test", "s1")):
        os.makedirs(os.path.join(outputPath, "test", "s1"))

    if not os.path.exists(os.path.join(outputPath, "test", "s2")):
        os.makedirs(os.path.join(outputPath, "test", "s2"))

    if not os.path.exists(os.path.join(outputPath, "train_ori", "mix")):
        os.makedirs(os.path.join(outputPath, "train_ori", "mix"))

    if not os.path.exists(os.path.join(outputPath, "train_ori", "s1")):
        os.makedirs(os.path.join(outputPath, "train_ori", "s1"))

    if not os.path.exists(os.path.join(outputPath, "train_ori", "s2")):
        os.makedirs(os.path.join(outputPath, "train_ori", "s2"))

    if not os.path.exists(os.path.join(outputPath, "test_ori", "mix")):
        os.makedirs(os.path.join(outputPath, "test_ori", "mix"))

    if not os.path.exists(os.path.join(outputPath, "test_ori", "s1")):
        os.makedirs(os.path.join(outputPath, "test_ori", "s1"))

    if not os.path.exists(os.path.join(outputPath, "test_ori", "s2")):
        os.makedirs(os.path.join(outputPath, "test_ori", "s2"))

    # load signals
    index = 0
    CHUNK = 2000
    NUM_SAMPLES_PER_TRAIN = 20
    category = ['c0']

    fecg = 'fecg1'
    mecg = 'mecg'
    noise = 'noise1'

    for i in os.listdir(inputPath):
        curr_dir = os.path.join(inputPath, i)
        if os.path.isdir(curr_dir) and i[:3] == 'sub':
            print("working on folder {}".format(i))
            for j in os.listdir(curr_dir):
                work_dir = os.path.join(curr_dir, j)
                if os.path.isdir(work_dir) and j[0] != '.':
                    print(" "*4+"working on {}...".format(j))
                    list_folders = os.listdir(work_dir)
                    list_folders.sort()
                    label = []
                    for k in list_folders:
                        if k[-4:] == ".dat":
                            split_k = k.split('_')
                            temp_label = '_'.join(split_k[:3])
                            if temp_label not in label:
                                print(" "*8+"processing {}".format(temp_label))
                                label.append(temp_label)
                                for cat in category:
                                    fecg_filename = '_'.join([temp_label,fecg])
                                    mecg_filename = '_'.join([temp_label,mecg])
                                    noise_filename = '_'.join([temp_label,cat,noise])
                                    fecg_ = wfdb.rdrecord(os.path.join(work_dir,fecg_filename)).p_signal.transpose(1,0)
                                    mecg_ = wfdb.rdrecord(os.path.join(work_dir,mecg_filename)).p_signal.transpose(1,0)
                                    noise_ = wfdb.rdrecord(os.path.join(work_dir,noise_filename)).p_signal.transpose(1,0)
                                    for i in range(len(fecg)):
                                        noise_record = noise_[i]
                                        fecg_record = fecg_[i]
                                        mecg_record = mecg_[i]
                                        start_point = np.random.randint(0, high=len(fecg_record)-CHUNK, size=NUM_SAMPLES_PER_TRAIN)
                                        mix_record = fecg_record + mecg_record + noise_record

                                        for l in start_point:
                                            np.save(os.path.join(outputPath, "train", 'mix', '{}.npy'.format(index)), mix_record[l:l+CHUNK])
                                            np.save(os.path.join(outputPath, "train", 's1', '{}.npy'.format(index)), mecg_record[l:l+CHUNK])
                                            np.save(os.path.join(outputPath, "train", 's2', '{}.npy'.format(index)), fecg_record[l:l+CHUNK])

                                            index += 1
    print("number of train samples = {}".format(index))
    
    index = 0
    NUM_SAMPLES_PER_TEST = 5
    for i in os.listdir(inputPath):
        curr_dir = os.path.join(inputPath, i)
        if os.path.isdir(curr_dir) and i[:3] == 'sub':
            print("working on folder {}".format(i))
            for j in os.listdir(curr_dir):
                work_dir = os.path.join(curr_dir, j)
                if os.path.isdir(work_dir) and j[0] != '.':
                    print(" "*4+"working on {}...".format(j))
                    list_folders = os.listdir(work_dir)
                    list_folders.sort()
                    label = []
                    for k in list_folders:
                        if k[-4:] == ".dat":
                            split_k = k.split('_')
                            temp_label = '_'.join(split_k[:3])
                            if temp_label not in label:
                                print(" "*8+"processing {}".format(temp_label))
                                label.append(temp_label)
                                for cat in category:
                                    fecg_filename = '_'.join([temp_label,fecg])
                                    mecg_filename = '_'.join([temp_label,mecg])
                                    noise_filename = '_'.join([temp_label,cat,noise])
                                    fecg_ = wfdb.rdrecord(os.path.join(work_dir,fecg_filename)).p_signal.transpose(1,0)
                                    mecg_ = wfdb.rdrecord(os.path.join(work_dir,mecg_filename)).p_signal.transpose(1,0)
                                    noise_ = wfdb.rdrecord(os.path.join(work_dir,noise_filename)).p_signal.transpose(1,0)
                                    for i in range(len(fecg)):
                                        noise_record = noise_[i]
                                        fecg_record = fecg_[i]
                                        mecg_record = mecg_[i]
                                        start_point = np.random.randint(0, high=len(fecg_record)-CHUNK, size=NUM_SAMPLES_PER_TEST)
                                        mix_record = fecg_record + mecg_record + noise_record

                                        for l in start_point:
                                            np.save(os.path.join(outputPath, "test", 'mix', '{}.npy'.format(index)), mix_record[l:l+CHUNK])
                                            np.save(os.path.join(outputPath, "test", 's1', '{}.npy'.format(index)), mecg_record[l:l+CHUNK])
                                            np.save(os.path.join(outputPath, "test", 's2', '{}.npy'.format(index)), fecg_record[l:l+CHUNK])

                                            index += 1
    print("number of test samples = {}".format(index))

inputPath = "./fetal-ecg-synthetic-database-1.0.0"
outputPath = "./data"

print("Setting up dataset...")
fecgsyndbSetup(inputPath, outputPath)

# package to npz array
train_dir = "mix"
test_dir_1 = "s1"
test_dir_2 = "s2"

a = []
u = []
v = []

for filename in os.listdir("data/train/{}".format(train_dir)):
    if filename[-3:] == "npy":
        file = np.load("data/train/{}/{}".format(train_dir, filename))
        a.append(file)
        file = np.load("data/train/{}/{}".format(test_dir_1, filename))
        u.append(file)
        file = np.load("data/train/{}/{}".format(test_dir_2, filename))
        v.append(file)

for filename in os.listdir("data/test/{}".format(train_dir)):
    if filename[-3:] == "npy":
        file = np.load("data/test/{}/{}".format(train_dir, filename))
        a.append(file)
        file = np.load("data/test/{}/{}".format(test_dir_1, filename))
        u.append(file)
        file = np.load("data/test/{}/{}".format(test_dir_2, filename))
        v.append(file)

a = np.vstack(a)
u = np.vstack(u)
v = np.vstack(v)
print("a shape", a.shape)
print("u shape", u.shape)
print("v shape", v.shape)

data = {
    "mix": a,
    "s1": u,
    "s2": v,
}
io.savemat('./data/fecgsyndb.mat', data)