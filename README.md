# FQF and IQN in PyTorch

A PyTorch implementation of Fully parameterized Quantile Function(FQF)[[1]](https://arxiv.org/abs/1911.02140) and Implicit Quantile Networks(IQN)[[2]](https://arxiv.org/abs/1806.06923). I tried to make it easy for readers to understand algorithms. Please let me know if you have any questions.

## Installation
You can install dependencies using `pip install -r requirements.txt`.

## Examples
You can train FQF agent using hyperparameters [here](https://github.com/ku2482/fqf.pytorch/blob/master/config/fqf.yaml).

```
python train_fqf.py --cuda --env_id BreakoutNoFrameskip-v4 --seed 0
```

## References
[[1]](https://arxiv.org/abs/1911.02140) Yang, Derek, et al. "Fully Parameterized Quantile Function for Distributional Reinforcement Learning." Advances in Neural Information Processing Systems. 2019.

[[2]](https://arxiv.org/abs/1806.06923) Dabney, Will, et al. "Implicit quantile networks for distributional reinforcement learning." arXiv preprint. 2018.
