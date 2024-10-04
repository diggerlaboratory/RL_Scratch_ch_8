import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()
            

def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q.load_state_dict(torch.load("./bestPolicy.pth"))
    epsilon = 0.01 #Linear annealing from 8% to 1%
    s = env.reset()
    done = False
    score = 0
    while not done:
        a = q.sample_action(torch.from_numpy(s).float(), epsilon) 
        s_prime, r, done, info = env.step(a)
        s = s_prime
        score += r
        print(score)
        if done:
            break
    env.close()


if __name__ == '__main__':
    main()