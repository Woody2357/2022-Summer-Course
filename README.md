# 2022-Summer-Course
This is a repository of the supplementary implementation for the 2022 summer course 'Mathematical Theory and Applications of Deep Learning', taught by Professor Haizhao Yang at Tianyuan Mathematical Center in Central China (TMCC).

The course videos and slides can be found at [the website of TMCC](https://tmcc.whu.edu.cn/info/1262/2052.htm).

Here are several tips and clarifications:

* Codes for Lecture 1-4 are based on some open sources, including [mofanpy](https://mofanpy.com) and [Hands-on RL](https://hrl.boyuai.com).
* Lecture 5 is based on [pde-net-2.0](https://github.com/ZichaoLong/PDE-Net/tree/PDE-Net-2.0). It's suggested to use python 3.7 and Pytorch 1.1.0. You also need to download [aTeam](https://github.com/ZichaoLong/aTEAM/tree/3b3b5289dcc1f9cbd54b4713819d5897579d7442) to generate data. Pre-trained parameters and coeffs are provided in the folder 'Lecture5/pre'.
* Lecture 6 is based on [LSTM_Prediction](https://github.com/HaizhaoYang/HaizhaoYang.github.io/blob/master/codes/LSTM_Prediction.zip). It's suggested to run on a GPU device.
* Lecture 7 is for Deep Galerkin Method. [A Chinese blog](https://zhuanlan.zhihu.com/p/359328643) is recommended for better understanding.
* Lecture 8 provides a copy for [FEX](https://github.com/LeungSamWai/Finite-expression-method) but a small bug is fixed in the version here.
* Lecture 9 is based on [IAE-Net](https://github.com/ongyongzheng/iae_net) and [Learning to Scan](https://github.com/simonat2011/RLCT-v4). As for iae-net, you have to carefully set parameters to suit your device because of the large amount of parameters. As for learning to scan, it is recommended to create a new environment that strictly matches the version, otherwise it will be incompatible when calling the medical package.
