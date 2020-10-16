## Testing LaNet

1. Download pre-trained checkpoint from <a href="https://drive.google.com/file/d/1bZsEoG-sroVyYR4F_2ozGLA5W50CT84P/view?usp=sharing">here</a>, and place and unzip it in the same folder.

2. Run the following command to test.
```
python test.py  --checkpoint  ./lanas_128_99.03 --layers 24 --init_ch 128 --arch='[2, 2, 0, 2, 1, 2, 0, 2, 2, 3, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 1, 0, 3, 4, 3, 0, 3, 1]'
```

```[2, 2, 0, 2, 1, 2, 0, 2, 2, 3, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 1, 0, 3, 4, 3, 0, 3, 1]``` is the best network found during the search. The snapshot below shows the top performing architectures (bottom to top) found during the distributed search. You can get the whole trace from <a href="../../Distributed_LaNAS">here</a>.

<p align="center">
<img src='https://github.com/linnanwang/paper-image-repo/blob/master/LaNAS/distributed_search_results.png?raw=true' width="600">
</p>



## Training LaNet
1. Install cutmix

```pip install git+https://github.com/ildoonet/cutmix```

2. run training with the following command.

```
mkdir checkpoints
python train.py --auxiliary --batch_size=32 --init_ch=128 --layer=24 --arch='[2, 2, 0, 2, 1, 2, 0, 2, 2, 3, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 1, 0, 3, 4, 3, 0, 3, 1]' --model_ema --model-ema-decay 0.9999 --auto_augment --epochs 1500
```

- **Training on the ImageNet**

Please use the training pipeline from <a href="https://github.com/rwightman/pytorch-image-models">Pytorch-Image-Models</a>. Here we describe the procedures to do so:
1. get the network from train.py, line 121
2. go to Pytorch-Image-Models
3. find pytorch-image-models/blob/master/timm/models/factory.py, replace line 57 as follows
``` 
# model = create_fn(**model_args, **kwargs) 
model = our-network
```
<b> Our ImageNet pipeline will be released soon, stay tuned. </b>


