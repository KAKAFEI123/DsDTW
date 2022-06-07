## The DsDTW method [1] for dynamic signature verification.

This repository implements a soft-DTW-based deep model for learning online signature representations. The soft-DTW was cited from Cuturi et al.'s work [3]. The model structure is based on the recurrent adaptation network in our previous work [2]. To use this repository, you should first download the DeepSignDB database [4]. If you find this repository useful, please cite our papers [1,2]. 

### Pipeline 

- run.sh: Training the DsDTW models.

- evaluate_DsDTW.sh: Extracting signature feature vectors using the trained models, and verifying the signatures according to the protocol of DeepSignDB [4].

### Environment Setup

Python 3.7.5 or a later version is suggested. The package dependencies can be seen in the requirements.txt.

### References

[1] Jiang J, Lai S, Jin L, et al. DsDTW: Local Representation Learning with Deep soft-DTW for Dynamic Signature Verification[J]. IEEE Transactions on Information Forensics and Security, 2022.

[2] Lai S, Jin L. Recurrent adaptation networks for online signature verification[J]. IEEE Transactions on information forensics and security, 2018, 14(6): 1624-1637.

[3] Cuturi M, Blondel M. Soft-dtw: a differentiable loss function for time-series[C]//International conference on machine learning. PMLR, 2017: 894-903.

[4] Tolosana R, Vera-Rodriguez R, Fierrez J, et al. DeepSign: Deep on-line signature verification[J]. IEEE Transactions on Biometrics, Behavior, and Identity Science, 2021.

### Contacts

jiajiajiang123@qq.com & eesxlai@qq.com & eelwjin@scut.edu.cn
