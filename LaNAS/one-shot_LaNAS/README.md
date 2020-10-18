## one-shot/few-shot LaNAS
• <b>fast to get a working result</b>

• <b>the inaccurate prediction from supernet degrades the final network performance</b>

The one-shot LaNAS uses a pretrained supernet to predict the performance of a proposed architecture via masking. The following figure illustrates the search procedures.

<p align="center">
<img src='https://github.com/linnanwang/paper-image-repo/blob/master/LaNAS/one-shot_LaNAS_search.png?raw=true' width="600">
</p>

The training of supernet is same as the regular training except for that we apply a random mask at each iterations. 

## Evaluating search algorithms on the supernet
NASBench-101 has very limited architectures (~420K architectures), which can be easily predicted with some sort of predictor. Supernet can be a great alternative to solve this problem as it renders a search space having 10^21 architectures. Therefore, our supernet can also be used as a benchmark to evaluate different search algorithms. See Fig.6 in <a href="https://linnanwang.github.io/latent-actions.pdf">LaNAS paper</a>. Please check how LaNAS interacts with supernet, and samples the architecture and its accuracy.


## Training the supernet
You can skip this step if use our pre-trained supernet.

Our supernet is designed for NASNet search space, and changing it to a new design space requires some work to change the codes. We're working on this issue, will update later. The training of supernet is fairly easy, simply

``` python train.py ```

- **Training on the ImageNet**

Please use the training pipeline from <a href="https://github.com/rwightman/pytorch-image-models">Pytorch-Image-Models</a>. Here we describe the procedures to do so:
1. get the supernet model from supernet_train.py, line 94
2. go to Pytorch-Image-Models
3. find pytorch-image-models/blob/master/timm/models/factory.py, replace line 57 as follows
``` 
# model = create_fn(**model_args, **kwargs) 
model = our-supernet
```

## Searching with a supernet
You can download the supernet pre-trained by us from <a href="https://drive.google.com/file/d/11RqnHAcfhiSYvCSpYZDfilCI1CYmL7WK/view?usp=sharing">here<a>. Place it in the same folder, and start searching with
 
 
``` python train.py ```

The search results will be written into a results.txt, and you can read the results by 

``` python read_result.py ```

The program outputs every samples with its test accuracy, e.g.

>[[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] 81.69 3774

> <b>[1.0 .. 0.0] is the architecture encoding, which can be used to train a network later.</b>

> <b>81.69 is the test accuracy predicted from supernet via weight sharing.</b>

> <b>3774 means this is the 3774th sample.</b>

## Training a searched network
Once you pick a network after reading the results, you can train the network in the Evaluate folder.
```
cd Evaluate
#attention, you need supply the code of target architecture in the argument of masked_code
python super_individual_train.py --cutout --auxiliary --batch_size=16 --init_ch=36 --masked_code='[1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0]'
```
## Improving with few-shot NAS
Though one-shot NAS substantially reduces the computation cost by training only one supernet, to approximate the performance of every architecture in the search space via weight-sharing. However, the performance estimation can be very inaccurate due to the co-adaption among operations. 
Recently, we propose <b>few-shot NAS</b> that uses multiple supernetworks, called sub-supernet, each covering different regions of the search space to alleviate the undesired co-adaption. Since each sub-supernet only covers a small search space, compared to one-shot NAS, few-shot NAS improves the accuracy of architecture evaluation with a small increase of evaluation cost. Please see the following paper for details.

<a href="https://arxiv.org/abs/2006.06863">Few-shot Neural Architecture Search</a> </br>
in submission</br>
Yiyang Zhao (WPI), Linnan Wang (Brown), Yuandong Tian (FAIR), Rodrigo Fonseca (Brown), Tian Guo (WPI)

**To Evaluate Few-shot NAS**, please check this <a href="https://github.com/aoiang/few-shot-NAS">repository</a>. The following figures show the performance improvement of few-shot NAS.
<p align="center">
<img src='https://github.com/linnanwang/paper-image-repo/blob/master/LaNAS/few-shot-1.png?raw=true' width="1000">
<img src='https://github.com/linnanwang/paper-image-repo/blob/master/LaNAS/few-shot-2.png?raw=true' width="1000">
</p>
These figures basically tell you few-shot NAS is an effective trade-off between one-shot NAS and vanilla NAS, i.e. training from scratch that retains both good performance estimation of a network and the fast speed.


