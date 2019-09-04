class RL(object):
    def __init__(self, agent, env):
        self._agent = agent
        self._env = env

        self._train_mode = None

    def train(self):
        self._train_mode = True

    def eval(self):
        self._train_mode = False

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, _agent):
        self._agent = _agent

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, _env):
        self._env = _env

    def rollout(self):
        assert self._train_mode is not None, "Set property 'train_mode' to True or False."

        ep_rwd = []
        done = False

        s = self._env.reset()
        while not done:
            if self._train_mode:
                a = self._agent.choose_action(s, sample=True)
            else:
                a = self._agent.choose_action(s, sample=False)
            s_, r, done = self._env.step(a)

            if self._train_mode:
                self._agent.push((s, a, r))

            ep_rwd.append(r)

            if done:
                if self._train_mode:
                    self._agent.train()
                break

            # Swap states
            s = s_

        return ep_rwd

    def rollout_many(self, num_episodes):
        return [self.rollout() for _ in range(num_episodes)]


def test_rl():
    import pandas as pd
    from frl.env import Env
    from frl.agent import Agent

    # Create agent
    agent = Agent(
        in_features=26,
        num_layers=1,
        hidden_size=32,
        out_features=3,
        lr=1e-3,
        weight_decay=0,
        discount_factor=0.9
    )

    # Create env
    feat_path = './bitcoin-historical-data/coinbaseUSD_15-min_feat.csv'
    data_path = './bitcoin-historical-data/coinbaseUSD_15-min_data.csv'

    feat = pd.read_csv(feat_path, index_col=[0], parse_dates=True)[:100]
    data = pd.read_csv(data_path, index_col=[0], parse_dates=True)[:100]

    env = Env(feat, data)

    # Create an RL instance
    rl = RL(agent, env)
    rl.train()
    rl.rollout()

    rl.rollout_many(num_episodes=500)
    rl.eval()
    rl.rollout()


if __name__ == '__main__':
    test_rl()
