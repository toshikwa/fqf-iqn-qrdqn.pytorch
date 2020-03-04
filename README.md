# FQF, IQN and QR-DQN in PyTorch

This is a PyTorch implementation of Fully parameterized Quantile Function(FQF)[[1]](#references), Implicit Quantile Networks(IQN)[[2]](#references) and Quantile Regression DQN(QR-DQN)[[3]](#references). I tried to make it easy for readers to understand algorithms. Please let me know if you have any questions. Also, any pull requests are welcomed.

## Installation
You can install dependencies using `pip install -r requirements.txt`.

## Examples
You can train FQF agent using hyperparameters [here](https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master/config/fqf.yaml).

```
python train_fqf.py --cuda --env_id PongNoFrameskip-v4 --seed 0 --config config/fqf.yaml
```

You can also train IQN or QR-DQN agent in the same way. Note that we log results with the number of frames, which equals to the number of agent's steps multiplied by 4 (e.g. 100M frames means 25M agent's steps).

## Results
Results of examples (without n-step rewards, double q-learning, dueling network nor noisy net) are shown below, which is comparable (if no better) with the paper. Scores below are evaluated arfer every 1M frames (250k agent's steps). Result are averaged over 2 seeds and visualized with min/max.

Note that I only trained a limited number of frames due to limited resources (e.g. 100M frames instead of 200M).

### BreakoutNoFrameskip-v4
I tested FQF, IQN and QR-DQN on `BreakoutNoFrameskip-v4` for 30M frames to see algorithms worked.

<img src="https://user-images.githubusercontent.com/37267851/75846342-5a49bb00-5e1f-11ea-911c-ae287d45426f.png" width=700>


### BerzerkNoFrameskip-v4
I also tested FQF and IQN on `BerzerkNoFrameskip-v4` for 100M frames to see the difference between FQF's performance and IQN's, which is quite obvious on this task.

<img src="https://user-images.githubusercontent.com/37267851/75846243-0ccd4e00-5e1f-11ea-9c03-b93e7b505dc8.png" width=700>

## TODO

- [ ] Implement risk-averse policies for IQN.
- [ ] Test FQF-Rainbow agent.

## References
[[1]](https://arxiv.org/abs/1911.02140) Yang, Derek, et al. "Fully Parameterized Quantile Function for Distributional Reinforcement Learning." Advances in Neural Information Processing Systems. 2019.

[[2]](https://arxiv.org/abs/1806.06923) Dabney, Will, et al. "Implicit quantile networks for distributional reinforcement learning." arXiv preprint. 2018.

[[3]](https://arxiv.org/abs/1710.10044) Dabney, Will, et al. "Distributional reinforcement learning with quantile regression." Thirty-Second AAAI Conference on Artificial Intelligence. 2018.

