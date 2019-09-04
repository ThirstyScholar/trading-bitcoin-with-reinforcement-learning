import math
import numpy as np
import pandas as pd


class Env(object):

    def __init__(self, feat, data):
        self.feat = feat
        self.data = data

        self.timestamp = self.feat.index
        self.feat_val = feat.values
        self.data_val = data.values

        assert len(self.feat) == len(self.data), 'Feature and Data DataFrame should be of the same length.'
        self._len = len(self.feat)
        self._timestamp = None
        self._i_step = 0
        self._state = None
        self._action = None
        self._reward = None
        self._done = False
        self._iter = None
        self._data = None

    def _encode_position(self):
        pos = np.zeros(5)
        pos[self._action] = 1
        return pos

    def _make_state(self):
        return np.concatenate((self._state, self._encode_position()))

    def reset(self):
        self._i_step = 0
        self._action = 1
        self._reward = None
        self._done = False
        self._iter = zip(self.timestamp, self.feat_val, self.data_val)

        self._timestamp, self._state, self._data = next(self._iter)
        return self._make_state()

    def step(self, a):
        self._i_step += 1
        self._timestamp, self._state, self._data = next(self._iter)

        # Compute reward
        opn = self._data[0]
        cls = self._data[3]
        self._reward = a * math.log(cls / opn)

        # Check if episode terminates
        self._done = True if self._i_step == self._len-1 else False
        return self._make_state(), self._reward, self._done

    @property
    def state(self):
        return self._state

    @property
    def action(self):
        return self._action

    @property
    def reward(self):
        return self._reward

    @property
    def done(self):
        return self._done

    def __len__(self):
        return self._len


def test_env():
    import random

    feat_path = './bitcoin-historical-data/coinbaseUSD_15-min_feat.csv'
    data_path = './bitcoin-historical-data/coinbaseUSD_15-min_data.csv'

    feat = pd.read_csv(feat_path, index_col=[0], parse_dates=True)
    data = pd.read_csv(data_path, index_col=[0], parse_dates=True)

    env = Env(feat, data)
    s = env.reset()

    done = False
    while not done:
        a = random.randint(0, 2)
        s, r, done = env.step(a)

        if done:
            break


if __name__ == '__main__':
    test_env()
