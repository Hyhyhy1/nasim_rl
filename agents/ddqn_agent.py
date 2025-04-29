import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import collections
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MemoryBuffer(object):
    def __init__(self, max_size):
        self.memory_size = max_size
        self.transaction_counter=0
        self.index=0
        self.buffer = collections.deque(maxlen=self.memory_size)
        self.transition = collections.namedtuple("Transition", field_names=["state", "action", "reward", "new_state", "terminal"])

    
    def save(self, state, action, reward, new_state, terminal):
        t = self.transition(state, action, reward, new_state, terminal)
        self.buffer.append(t)
        self.transaction_counter = (self.transaction_counter + 1) % self.memory_size

    def random_sample(self, batch_size):
        assert len(self.buffer) >= batch_size # should begin sampling only when sufficiently full
        transitions = random.sample(self.buffer, k=batch_size) # number of transitions to sample
        states = torch.from_numpy(np.vstack([e.state for e in transitions if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float().to(device)
        new_states = torch.from_numpy(np.vstack([e.new_state for e in transitions if e is not None])).float().to(device)
        terminals = torch.from_numpy(np.vstack([e.terminal for e in transitions if e is not None]).astype(np.uint8)).float().to(device)
  
        return states, actions, rewards, new_states, terminals

class QNN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, action_size)
        
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)

class Agent(object):
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, is_learning=True):
        self.gamma = gamma # alpha = learn rate, gamma = discount
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec # decrement of epsilon for larger spaces
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.is_learning = is_learning
        self.memory = MemoryBuffer(mem_size)

    def save(self, state, action, reward, new_state, done):
        self.memory.save(state, action, reward, new_state, done)  

    def reduce_epsilon(self):
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                       self.epsilon_min else self.epsilon_min  
        
        
        
    
class DoubleQAgent(Agent):
    def __init__(self, observation_space_shape, action_space_n, gamma=0.99, epsilon=1.0, 
                 batch_size=128, lr=0.0003, epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, replace_q_target = 100, is_learning=True):
        
        super().__init__(lr=lr, gamma=gamma, epsilon=epsilon, batch_size=batch_size,
             epsilon_dec=epsilon_dec,  epsilon_end=epsilon_end,
             mem_size=mem_size, is_learning=is_learning)

        self.action_space_n = action_space_n
        self.replace_q_target = replace_q_target
        self.q_func = QNN(observation_space_shape, action_space_n, 42).to(device)
        self.q_func_target = QNN(observation_space_shape, action_space_n, 42).to(device)
        self.optimizer = optim.Adam(self.q_func.parameters(), lr=lr)

    
    def choose_action(self, state):
        rand = np.random.random()
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.q_func.eval()
        with torch.no_grad():
            action_values = self.q_func(state)
        self.q_func.train()
        if rand > self.epsilon or self.is_learning == False: 
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # exploring: return a random action
            return np.random.choice([i for i in range(self.action_space_n)])
        
        
    def learn(self):
        if self.memory.transaction_counter < self.batch_size: # wait before you start learning
            return
            
        # 1. Choose a sample from past transitions:
        states, actions, rewards, new_states, terminals = self.memory.random_sample(self.batch_size)
        
        # 2. Update the target values
        q_next = self.q_func_target(new_states).detach().max(1)[0].unsqueeze(1)
        q_updated = rewards + self.gamma * q_next * (1 - terminals)
        q = self.q_func(states).gather(1, actions)
        
        # 3. Update the main NN
        loss = F.mse_loss(q, q_updated)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 4. Update the target NN (every N-th step)
        if self.memory.transaction_counter % self.replace_q_target == 0: # wait before you start learning
            for target_param, local_param in zip(self.q_func_target.parameters(), self.q_func.parameters()):
                target_param.data.copy_(local_param.data)
                
        # 5. Reduce the exploration rate
        self.reduce_epsilon()