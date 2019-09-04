import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, in_features, num_layers, hidden_size, out_features, lr, weight_decay):
        super().__init__()

        # Define input, hidden and output layers
        self.in_layer = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU()
        )
        self.hidden_layers = nn.Sequential(*[
            layer
            for layer in [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
            for _ in range(num_layers)
        ])
        self.out_layer = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, out_features),
            nn.Softmax(dim=1)
        )

        # Create optimizer
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.hidden_layers(x)
        x = self.out_layer(x)
        return x


class Agent(object):

    def __init__(self, in_features, num_layers, hidden_size, out_features, lr, weight_decay, discount_factor):
        self.net = Net(
            in_features=in_features,
            num_layers=num_layers,
            hidden_size=hidden_size,
            out_features=out_features,
            lr=lr,
            weight_decay=weight_decay
        )
        self.net.eval()
        self.discount_factor = discount_factor

        # A buffer to store all (state, action, reward) tuple in an episode that are used for training at the end
        self.history = []

    def forward(self, x):
        return self.net(x)

    def choose_action(self, s, sample):
        """
        :param s: NumPy Ndarray
            A 1-dim numpy array.
        :param sample: bool
        :return: int
        """
        s = torch.from_numpy(s).float().unsqueeze(0)   # cast to a 2-dim Torch Tensor
        a_prob = self.net(s)
        if sample:
            a = torch.multinomial(a_prob.data, num_samples=1).item()
        else:
            a = a_prob.data.max(1)[1].item()
        return a

    def push(self, tpl):
        """
        :param tpl: tuple
            A tuple of (state, action, reward).
        :return: None
        """
        self.history.append(tpl)

    def train(self):
        s_seq, a_seq, r_seq = zip(*self.history)

        # Concatenate s_seq and cast it into a 2-dim Torch Tensor
        state = torch.from_numpy(np.concatenate(s_seq).reshape(len(s_seq), -1)).float()

        # Concatenate a_seq and cast it into a 2-dim Long Tensor
        action = torch.LongTensor(a_seq).unsqueeze(1)

        # Reverse computation of the Q-value
        q_seq = []
        q = 0
        for r in reversed(r_seq):
            q = r + self.discount_factor * q  # discount by 0.9
            q_seq.append(q)
        q_seq.reverse()

        # Standardize Q-value to improve the efficiency of gradient descent
        q_seq = np.array(q_seq)
        q_seq -= q_seq.mean()
        q_seq /= (q_seq.std() + 1e-6)
        np.clip(q_seq, -10, 10, out=q_seq)

        # Cast to 2-dim Torch Tensor
        q_val = torch.from_numpy(q_seq).float().unsqueeze(1)

        # Compute loss function: negative of Q * log(Action_Prob)
        a_prob = self.net(state).gather(1, action)
        a_prob = torch.clamp_min(a_prob, 1e-6)  # prevent too small number since we are taking the log below
        loss = -(q_val * torch.log(a_prob)).mean()

        # Perform gradient ascent
        self.net.train()
        self.net.optim.zero_grad()
        loss.backward()
        self.net.optim.step()
        self.net.eval()

        # Clear buffer after training
        del self.history[:]


def test_agent():
    agent = Agent(
        in_features=28,
        num_layers=1,
        hidden_size=32,
        out_features=5,
        lr=1e-3,
        weight_decay=0,
        discount_factor=0.9
    )

    import pandas as pd
    from frl.env import Env

    feat_path = './bitcoin-historical-data/coinbaseUSD_15-min_feat.csv'
    data_path = './bitcoin-historical-data/coinbaseUSD_15-min_data.csv'

    feat = pd.read_csv(feat_path, index_col=[0], parse_dates=True)[:100]
    data = pd.read_csv(data_path, index_col=[0], parse_dates=True)[:100]

    env = Env(feat, data)

    total_rwd_lst = []
    for i_epoch in range(1000):
        s = env.reset()

        total_rwd = 0
        done = False
        while not done:
            a = agent.choose_action(s, sample=False)
            s_, r, done = env.step(a)

            total_rwd += r

            agent.push((s, a, r))
            s = s_

            if done:
                print(f'{total_rwd:.3f}')
                total_rwd_lst.append(total_rwd)

                # Agent training when episode ends
                agent.train()

                break

    import matplotlib.pyplot as plt
    plt.plot(total_rwd_lst)
    plt.show()


if __name__ == '__main__':
    test_agent()
