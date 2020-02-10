import os
import yaml
import argparse
from datetime import datetime

from fqf.env import make_pytorch_env
from fqf.agent import FQFAgent


def run(args):
    with open(os.path.join('config', 'fqf.yaml')) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    env = make_pytorch_env(args.env_id)
    test_env = make_pytorch_env(
        args.env_id, episode_life=False, clip_rewards=False)

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'FQF-{time}')

    agent = FQFAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='MsPacmanNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
