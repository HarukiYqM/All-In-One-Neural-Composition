# Resource-Adaptive Federated Learning with All-In-One Neural Composition
This repository is for FLANC introduced in the following paper "Resource-Adaptive Federated Learning with All-In-One Neural Composition", NeurIPS2022, [[Link]](https://openreview.net/pdf?id=wfel7CjOYk) 


The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and test on Ubuntu 20.04 environment (Python3.6, PyTorch 1.8.2) with A6000 GPUs. 
## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction
Conventional Federated Learning (FL) systems inherently assume a uniform processing capacity among clients for deployed models. However, diverse client
hardware often leads to varying computation resources in practice. Such system heterogeneity results in an inevitable trade-off between model complexity and
data accessibility as a bottleneck. To avoid such a dilemma and achieve resource-adaptive federated learning, we introduce a simple yet effective mechanism, termed
All-In-One Neural Composition, to systematically support training complexity-adjustable models with flexible resource adaption. It is able to efficiently construct
models at various complexities using one unified neural basis shared among clients, instead of pruning the global model into local ones. The proposed mechanism endows the system with unhindered access to the full range of knowledge scattered across clients and generalizes existing pruning-based solutions by allowing soft and learnable extraction of low footprint models. Extensive experiment results on popular FL benchmarks demonstrate the effectiveness of our approach. The resulting FL system empowered by our All-In-One Neural Composition, called FLANC, manifests consistent performance gains across diverse system/data heterogeneous setups while keeping high efficiency in computation and communication.
![FLANC](/Figs/FLANC.png)
FLANC
## Run the code
 Cd to 'src', run the following script to train models.

    **Example command is in the file 'demo.sh'.**

    ```bash
    # Example CIFAR10 IID
    MODEL=ResNet18_flanc
    CUDA_VISIBLE_DEVICES=$DEVICES python main_FL.py --n_agents 100 --dir_data ../ --data_train cifar10  --n_joined 10 --split iid --local_epochs 2 --batch_size 32 --epochs 500 --decay step-250-375 --lr 0.1 --fraction_list 0.25,0.5,0.75,1 --project FLANC_CIFAR10 --template ResNet18 --model ${MODEL} --basis_fraction 0.125 --n_basis 0.25 --save FLANC  --dir_save ../experiment --save_models

    
    ```

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@inproceedings{
mei2022resourceadaptive,
title={Resource-Adaptive Federated Learning with All-In-One Neural Composition},
author={Yiqun Mei and Pengfei Guo and Mo Zhou and Vishal Patel},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=wfel7CjOYk}
}
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}

```
## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [
learning_filter_basis](https://github.com/ofsoundof/learning_filter_basis). We thank the authors for sharing their codes.
