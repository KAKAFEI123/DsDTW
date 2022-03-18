# DsDTW

### The DsDTW method [1] for dynamic signature verification. 

This repository implements a soft-DTW-based deep model for learning online signature representations. The introduction of soft-DTW is inspired by Cuturi et al.'s work [3]. The design of model structure is based on the recurrent adaptation network in our previous work [2]. To use this repository, you should first download the DeepSignDB database [4]. If you find this repository useful, please cite our papers [1,2]. For further application in handwritten digit verification and letter identification, please download the e-BioDigit [5] and LERID [6] databases.

### Pipeline

--dataProcess: data preprocessing. 

--main.sh: Training DsDTW models

--evaluate.sh: Extraction of signature feature vectors using trained models, and verification of the signatures according to the protocol of DeepSignDB [3].

### Environment

Tested with PyTorch 1.6 and Python 3.7.

### References

[1] Jiang J, Lai S, Jin L, et al. DsDTW: Local representation learning with Deep soft-DTW for Dynamic Signature Verification.

[2] Lai S, Jin L. Recurrent adaptation networks for online signature verification[J]. IEEE Transactions on information forensics and security, 2018, 14(6): 1624-1637.

[3] Cuturi M, Blondel M. Soft-dtw: a differentiable loss function for time-series[C]//International conference on machine learning. PMLR, 2017: 894-903.

[4] Tolosana R, Vera-Rodriguez R, Fierrez J, et al. DeepSign: Deep on-line signature verification[J]. IEEE Transactions on Biometrics, Behavior, and Identity Science, 2021.

[5] Tolosana R, Vera-Rodriguez R, Fierrez J. BioTouchPass: Handwritten passwords for touchscreen biometrics[J]. IEEE Transactions on Mobile Computing, 2019, 19(7): 1532-1543.

[6] Chen Z, Yu H X, Wu A, et al. Level online writer identification[J]. International Journal of Computer Vision, 2021, 129(5): 1394-1409.

### Contacts

jiajiajiang123@qq.com & eesxlai@qq.com & eelwjin@scut.edu.cn
