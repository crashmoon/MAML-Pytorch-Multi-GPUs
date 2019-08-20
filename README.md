# MAML-Pytorch-Multi-GPUs
It is a reproduced version of maml, which is implemented with PyTorch 1.2.0 and support Multi-GPUs both in Meta-training phase and Meta-testing phase. 

All the hyper-parameters and tricks, e.g. gradient clip, are strictly consistent with the original paper `Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks` [HERE](https://arxiv.org/abs/1703.03400) and its `Tensorflow Implementation` [HERE](https://github.com/cbfinn/maml).

# Platform
- python: 3.7
- Pytorch: 1.2.0

# Howto
1. Downloading `MiniImagenet` dataset
2. Changing the Param.root in `train_mini_adam.py` with your own root of the MiniImagenet dataset
3. python train_mini_adam.py 


# Comparison to original MAML implementation for miniImageNet

|      | 5-way 1-shot | 5-way 5-shot |
|:----:|:------------:|:------------:|
| MAML |     48.7%    |     63.1%    |
| Ours |     48.9%    |     #TODO#   |

# reference
https://github.com/cbfinn/maml

https://github.com/dragen1860/MAML-Pytorch

https://github.com/jik0730/MAML-in-pytorch

https://arxiv.org/abs/1703.03400
