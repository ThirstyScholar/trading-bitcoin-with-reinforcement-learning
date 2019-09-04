import matplotlib.pyplot as plt
import numpy as np
import random
import sys

from frl.data import Data
from frl.rl import RL
from frl.env import Env
from frl.agent import Agent
plt.style.use('seaborn-paper')


def main():

    # Read CSV from file and perform data preprocessing / feature extraction
    data_path = sys.argv[1] if len(sys.argv) > 1 else './bitcoin-historical-data/coinbaseUSD_1-min_data.csv'
    data_src = Data(data_path)
    data_src.preprocess()
    data_src.extract_feature()

    # Train-test split the data by 70% and 30%
    cut = int(len(data_src) * 0.7)

    train_feat = data_src.feat.iloc[:cut]
    train_data = data_src.data.iloc[:cut]
    train_len = len(train_data)

    test_feat = data_src.feat.iloc[cut:]
    test_data = data_src.data.iloc[cut:]

    # Create agent
    agent = Agent(
        in_features=28,
        num_layers=1,
        hidden_size=32,
        out_features=5,
        lr=1e-3,
        weight_decay=1e-6,
        discount_factor=0.99
    )

    # Training
    for i_episode in range(5000):
        print(f'Episode {i_episode+1}', end=' ')

        # Sample an episode of length 96
        idx = random.randint(0, train_len-96)

        _train_feat = train_feat.iloc[idx: idx+96]
        _train_data = train_data.iloc[idx: idx+96]

        train_env = Env(_train_feat, _train_data)

        train_rl = RL(agent, train_env)
        train_rl.train()
        train_rwd = sum(train_rl.rollout())
        print(f'{train_rwd:.3f}')

    # Testing
    test_env = Env(test_feat, test_data)

    test_rl = RL(agent, test_env)
    test_rl.eval()
    test_rwd = test_rl.rollout()

    # Plotting
    plot_result(test_rwd, test_data['Open'], test_data['Close'])


def plot_result(rwd_lst, open_price, close_price):
    plt.subplot(211)

    # Baseline1: BnH
    ret = np.log(close_price / close_price.shift(1))
    ret.fillna(0, inplace=True)

    bnh = np.cumsum(ret.values) * 2
    plt.plot(bnh, label='BnH')

    # Baseline2: Momentum
    log_ret = np.log(close_price / open_price)

    sma = close_price.rolling(30, min_periods=1).mean()
    signal = (close_price > sma).shift(1).astype(float) * 4  # shift by 1 since we trade on the next opening price
    signal.fillna(0, inplace=True)

    mmt = np.cumsum(log_ret.values * signal.values)  # convert to cum. simple return
    plt.plot(mmt, label='MMT')

    # RL agent performance
    rl = np.cumsum(rwd_lst)

    plt.xticks(())
    plt.ylabel('Cumulative Log-Returns')
    plt.plot(rl, label='RL')
    plt.legend()

    def mdd(x):
        max_val = None
        temp = []
        for t in x:
            if max_val is None or t > max_val:
                max_val = t
            temp.append(t - max_val)
        return temp

    plt.subplot(212)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xlabel('Timesteps')
    plt.ylabel('MDD')
    plt.plot(mdd(bnh))
    plt.plot(mdd(mmt))
    plt.plot(mdd(rl))
    plt.show()


if __name__ == '__main__':
    main()
