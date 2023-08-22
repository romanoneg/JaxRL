import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical
import gym
import numpy as np

class episode:

    def __init__(self):

        # action log probability
        self.alp = []
        self.rewards = []
        self.T = 0


class mlp(nn.Module):

    # what a ̷d̷o̷g̷s̷h̷i̷t̷  "ugly" implementation of this 
    def __init__(self, obs, actions, layers, l_size):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(obs, l_size)] + 
            [nn.Linear(l_size, l_size) for _ in range(layers)] +
            [nn.Linear(l_size, actions)])

    def forward(self, x):
        # print(self.linears)
        for layer in self.linears[:-1]:
            x = F.relu(layer(x))
        last_layer = self.linears[-1]
        # unsure why the softmax is necessary in this, will have to research
        return F.softmax(last_layer(x), dim=-1)

    def act(self, state):
        # rewrite better cause wtf is log_probs.log_prob(action) like dude
        log_probs = Categorical(self(state))
        action = log_probs.sample()
        return action.item(), log_probs.log_prob(action)

def train(ep, optimizer):
    # maybe combine G and rew
    G, rew = torch.zeros(ep.T), 0
    # find better place for gamma
    gamma = 0.99

    # this hurts my eyes.... but I'll fix later
    for t in range(ep.T)[::-1]:
        rew = ep.rewards[t] + gamma * rew
        G[t] = rew

    # G = sum(G).item()
    loss = torch.sum(-torch.cat(ep.alp) * G)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

# # all the stuff to change right here
EPOCHS = 3000
reporting_interval = 100
env_name = "CartPole-v1"
render = ""


env = gym.make(env_name)
env.action_space.seed(42)
# print(env.action_space.n)

network = mlp(env.observation_space.shape[0], env.action_space.n, 1, 16)
optimizer = optim.Adam(network.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)

reported_reward = 0

for e in range(EPOCHS+1):
    state, obs = env.reset()
    ep = episode()
    terminated, truncated = False, False

    while not terminated and not truncated:

        action, action_probs = network.act(torch.from_numpy(state))
        state, reward, terminated, truncated, _ = env.step(action)

        # print(action_probs)
        ep.alp.append(action_probs.reshape(1))
        ep.rewards.append(reward)
        ep.T += 1

    loss = train(ep, optimizer)
    # scheduler.step()
    reported_reward += sum(ep.rewards) / reporting_interval

    # if render == "human":
        # env = gym.make(env_name)
        # render = ""
    if e % reporting_interval == 0 and e != 0:
        print(f"Epoch: {e}, Average reward = {reported_reward}")
        # env = gym.make(env_name, render_mode="human")
        # render = "human"
        reported_reward= 0 

    # print(f"Episode {e}: loss {loss}, total_reward {total_reward}, solved? {total_reward > 195}")
print(f"Episode {e}: loss {loss}")
env.close()