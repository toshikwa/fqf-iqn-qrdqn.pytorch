# Fully parameterized Quantile Function(FQF) in PyTorch

A PyTorch implementation of Fully parameterized Quantile Function(FQF)[[1]](#references). I tried to make it easy for readers to understand the algorithm. Please let me know if you have any questions.

## Installation
You can install dependencies using `pip install -r requirements.txt`.

## Examples
You can train FQF agent using hyperparameters [here](https://github.com/ku2482/fqf.pytorch/blob/master/config/fqf.yaml).

```
python main.py --cuda --env_id MsPacmanNoFrameskip-v4
```

## References
[[1]](https://arxiv.org/abs/1911.02140) Yang, Derek, et al. "Fully Parameterized Quantile Function for Distributional Reinforcement Learning." Advances in Neural Information Processing Systems. 2019.
