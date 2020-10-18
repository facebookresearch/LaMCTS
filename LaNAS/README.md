# Introduction
LaNAS is an application of LA-MCTS to Neural Architecture Search (NAS), though the more general approach (LA-MCTS) was inspired from LaNAS. Here is what are included in this release:

- <a href="./LaNAS_NASBench101">**Evaluation on NASBench-101** </a>: Evaluating LaNAS on NASBench-101 without training models. 

- <a href="./LaNet">**Our Searched Models, LaNet**</a>: SoTA results: • 99.03% on CIFAR-10 • 77.7% @ 240MFLOPS on ImageNet.

- <a href="./one-shot_LaNAS">**One/Few-shot LaNAS**</a>: Using a supernet to evaluate the model, obtaining results within a few GPU days.

- <a href="./Distributed_LaNAS">**Distributed LaNAS**</a>: Distributed framework for LaNAS, usable with hundreds of GPUs.

- <a href="./LaNet">**Training heuristics used**</a>: We list all tricks used in ImageNet training to reach SoTA. 

# Publication

<a href="https://linnanwang.github.io/latent-actions.pdf">Sample-Efficient Neural Architecture Search by Learning Action Space for Monte Carlo Tree Search</a> </br>
Linnan Wang (Brown University), Saining Xie (Facebook AI Research), Teng Li(Facebook AI Research), Rodrigo Fonesca (Brown University), Yuandong Tian (Facebook AI Research)</br>

And special thanks to the enormous help from Yiyang Zhao (Worcester Polytechnic Institute).

# Performance on NASBench-101
We have strictly followed NASBench-101 guidlines in benchmarking the results, please see our paper for details.
<p align="center">
<img src='https://github.com/linnanwang/paper-image-repo/blob/master/LaNAS/Benchmark.png?raw=true' width="800">
</p>

# Performance of Searched Networks
**CIFAR-10**: 99.03% top-1 using NASNet search space, SoTA result without using ImageNet or transfer learning.

|     Model      | LaNet      | EfficientNet-B7       | GPIPE                 | PyramidNet      | XNAS           |
| -------------- | ---------- | ---------             | ----------            | --------------  | -------------- |
| top-1          | 99.03      | 98.9                  | 99.0                  | 98.64           | 98.4           |
| use ImageNet   | X          | <span>&#10003;</span> | <span>&#10003;</span> | X               | X              |


**ImageNet**: 77.7% top-1@240 MFLOPS, 80.8% top-1@600 MFLOPS using EfficientNet search space, SoTA results on the efficentNet space.


|     Model      | LaNet      | OFA       | FBNetV2-F4 | MobileNet-V3    | FBNet-B        |
| -------------- | ---------- | --------- | ---------- | --------------  | -------------- |
| top-1          | 77.7       | 76.9      | 76.0       | 75.2            | 74.1           |
| MFLOPS         | 240        | 230       | 238        | 219             | 295            |

|     Model      | LaNet      | OFA       | FBNetV3    | EfficientNet-B1|
| -------------- | ---------- | --------- |  -----------| -------------- |
| top-1          | 80.8       | 80.0      |  79.6       | 79.1           |
| MFLOPS         | 600        | 595       |  544        | 700            |


**Applying LaNet to detection**: Compared to NAS-FCOS in CVPR-2020,
|     Backbone      | Decoder      | FLOPS(G)       | AP    |
| ----------------- | ------------ | -------------- |-------|
|     LaNet         | FPN-FCOS     | 35.22          | 36.5  |
|     MobileNetV2   | FPN-FCOS     | 105.4          | 31.2  |
|     MobileNetV2   | NAS-FCOS     | 39.3           | 32.0  |

<b>We will release the ImageNet model, search framework, training pipeline, and their applications on detection, segmentation soon; stay tuned.</b>


# Trying Other CV or NLP Applications
In LaNAS, we model a network architecture as a vector encoding, i.e. [x,...,x], and there is an decoder that translate the encoding into a runnable model for PyTorch or TensorFlow. Please see the function train_net in `net_training.py`. 
That means you only need implement an evaluator / cost model for your applications to use LaNAS. 
