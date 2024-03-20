# LG-Mix: Heterogeneous Personalized Federated Learning by Local-Global Updates Mixing via Convergence Rate
This is the PyTorch implemention of our paper **[Heterogeneous Personalized Federated Learning by Local-Global Updates Mixing via Convergence Rate](https://openreview.net/pdf?id=7pWRLDBAtc)** by [Meirui Jiang](https://meiruijiang.github.io/MeiruiJiang/), [Anjie Le](https://ale256.github.io/), [Xiaoxiao Li](https://xxlya.github.io/) and [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/)
##LG-Mix: > Personalized federated learning (PFL) has emerged as a promising technique for addressing the
> Personalized federated learning (PFL) has emerged as a promising technique for addressing the challenge of data heterogeneity. While recent studies have made notable progress in mitigating heterogeneity associated with label distributions, the issue of effectively handling feature heterogeneity remains an open question. In this paper, we propose a personalization approach by Local-Global updates Mixing (LG-Mix) via Neural Tangent Kernel (NTK)-based convergence. The core idea is to leverage the convergence rate induced by NTK to quantify the importance of local and global updates, and subsequently mix these updates based on their importance. Specifically, we find the trace of the NTK matrix can manifest the convergence rate, and propose an efficient and effective approximation to calculate the trace of a feature matrix instead of the NTK matrix. Such approximation significantly reduces the cost of computing NTK, and the feature matrix explicitly considers the heterogeneous features among samples. We have theoretically analyzed the convergence of our method in the over-parameterize regime, and experimentally evaluated our method on five datasets. These datasets present heterogeneous data features in natural and medical images. With comprehensive comparison to existing state-of-the-art approaches, our LG-Mix has consistently outperformed them across all datasets (largest accuracy improvement of 5.01%), demonstrating the outstanding efficacy of our method for model personalization.

## Usage
### Setup

We recommend using conda to quick setup the environment. Please use the following commands.
```bash
conda env create -f environment.yaml
conda activate torch_lgmix
```
Actually, our code has no specific requirements on any specific packages, for convenience, you can use any of your environments including the torch and other basic deep-learning packages.


### Dataset
For the datasets, we mainly follow the datasets from [FedBN](https://github.com/med-air/FedBN).

For the benchmark data of digits classification, please download the datasets [here](https://drive.google.com/file/d/1moBE_ASD5vIOaU8ZHm_Nsj0KAfX5T0Sf/view?usp=sharing), and specify the correponding path in the `dataset.py`.


### Run
`fed_train.py` is the main file to run the federated experiments
Please using following commands to train a model under FedAvg or our proposed method.
```bash
bash run.sh
```

## Citation
If you find the code useful, please cite our paper.
```latex
@inproceedings{
jiang2024heterogeneous,
title={Heterogeneous Personalized Federated Learning by Local-Global Updates Mixing via Convergence Rate},
author={Meirui Jiang and Anjie Le and Xiaoxiao Li and Qi Dou},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=7pWRLDBAtc}
}
```

For any questions, please contact mrjiang@cse.cuhk.edu.hk