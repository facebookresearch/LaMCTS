CIFAR10 folder currently contains the test and training pipeline using the NASNet search space. 

The code for EfficientNet search space on ImageNet will be released later.

## Performance of LaNet
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



## Bag of Tricks
Here are the following training heuristics we have used in our project:

- ***Data Augmentation*** 
> We use CutOut, CutMix and RandAugmentation. Pytorch-Image-Models has a very nice implementation, but keep an eye of SoTA data augmentation techniques.

- ***Distillation*** 

>a) The main source of the performance improvement in recent NAS EfficientNet paper.
It seems training a student network together with a teacher from scratch can further improve the current SoTA, 
better than transferring weights from a fancy supernet. 

>b) Use a better teacher helps.

>c) A better teacher may require larger images than student, use interpolation to resize the batch to feed into student and teacher.

- ***Training Hyper-parameters***
> drop out and drop path, tune your training hyper-parameters. The learning rate cannot be too large nor too small, check your loss progress.  

- ***EMA***
> Using Exponential Moving Average (EMA) in the models, e.g. CNN, Detection, Transformers, unsupervised models, or whatever NLP or CV models, helps the performance, especially when your training finishes with fewer number of epochs.

- ***Testing different Crops***
> Try changing different crop percentages in testing, it usually improves 0.1%.

- ***Longer epochs***
> Make sure your model is sufficiently trained.

