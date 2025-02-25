import torch as th
import random
from collections import deque
from Models import Model, CNNModel
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], capacity)

    def add(self, state, action, next_state, reward, done):
        self.memory.append({'state' : th.tensor(state),
                     'action' : th.tensor(action),
                     'next_state' : th.tensor(next_state),
                     'reward' : th.tensor(reward),
                     'done' : th.tensor(done)})
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQN:
    def __init__(self,
                state_dim,
                action_dim,
                hidden_size,
                batch_size,
                target_update_freq,
                model="1",
                train_freq = 4,
                ram_cap = int(1e6),
                gamma = 0.99,
                esp_start = 1, 
                eps_end = 0.01, 
                eps_decay = 0.999, 
                lr = 1e-4):
        self.policy_net = Model(state_dim, hidden_size, action_dim) if model == "1" else CNNModel(state_dim, action_dim)
        self.target_net = Model(state_dim, hidden_size, action_dim) if model == "1" else CNNModel(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.step_count = 0
        self.target_update_freq = target_update_freq

        self.memory = ReplayMemory(ram_cap)
        self.action_dim = action_dim
        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.train_freq = train_freq
        self.batch_size = batch_size
        self.eps = esp_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.lr = lr

    def predict(self, state):
        if np.random.random() > self.eps:
            return th.argmax(self.policy_net(th.tensor(state))).item()
        else:
            return np.random.randint(0, self.action_dim)
        
    def target_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def epsilon_update(self):
        if self.eps > self.eps_end:
            self.eps = max(self.eps_end, self.eps * self.eps_decay)

        
    def push(self, state, action, next_state, reward, done):
        self.memory.add(state, action, next_state, reward, done)
        
    def backward(self):
        if len(self.memory) < self.batch_size:
            return  

        loss = None 
        
        if self.step_count % self.train_freq == 0:
            batch = self.memory.sample(self.batch_size)

            state_batch = th.stack([b['state'] for b in batch])  
            next_state_batch = th.stack([b['next_state'] for b in batch])  
            action_batch = th.tensor([b['action'] for b in batch], dtype=th.long).unsqueeze(1)  
            reward_batch = th.tensor([b['reward'] for b in batch], dtype=th.float) 
            done_batch = th.tensor([b['done'] for b in batch], dtype=th.float) 

            q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()

            with th.no_grad():
                next_q_values = self.target_net(next_state_batch).max(1)[0]
                target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch.float())

            loss = F.smooth_l1_loss(q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq  == 0:
            self.target_update()

        if loss == None: return
        return loss.item()

