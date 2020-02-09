import os
import yaml
from datetime import datetime

from fqf.env import make_pytorch_env
from fqf.agent import FQFAgent


def run():
    with open(os.path.join('config', 'sample.yaml')) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    env = make_pytorch_env(config['env_name'])
    test_env = make_pytorch_env(
        config['env_name'], episode_life=False, clip_rewards=False)

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', config.pop('env_name'),
        f'FQF-{time}')

    agent = FQFAgent(
        env=env, test_env=test_env, log_dir=log_dir, **config)
    agent.run()


if __name__ == '__main__':
    run()
