import torch, random
import torch.nn.functional as F
import numpy as np
from collections import deque
from model import QNetwork

class DQNAgent:
    def __init__(self, state_dim, action_dim, device= "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)

        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.update_freq = 10 

    
    def act(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.policy_net.net[-1].out_features)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
             return self.policy_net(state).argmax(dim=1).item()


    def store(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, d = zip(*batch)
        return (torch.FloatTensor(s).to(self.device),
                torch.LongTensor(a).to(self.device),
                torch.FloatTensor(r).to(self.device),
                torch.FloatTensor(s2).to(self.device),
                torch.FloatTensor(d).to(self.device))

    def update(self, step):
        if len(self.memory) < self.batch_size:
            return
        s, a, r, s2, d = self.sample()
        q_vals = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q_next = self.target_net(s2).max(1)[0]
            q_target = r + self.gamma * q_next * (1 - d)
        loss = F.mse_loss(q_vals, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if step % self.update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
