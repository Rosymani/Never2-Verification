import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from matplotlib import pyplot as plt

env = gym.make('MountainCar-v0')
env.seed(110)
np.random.seed(10)

class DQN(nn.Module):
    """ Deep Q Network """

    def __init__(self, action_space, state_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_space, 24)
        self.fc2 = nn.Linear(24, 23)
        self.fc3 = nn.Linear(23, action_space)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:

    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = .95
        self.batch_size = 64
        self.epsilon_min = .01
        self.lr = 0.001
        self.epsilon_decay = .995
        self.memory = deque(maxlen=100000)
        self.model = DQN(action_space, state_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        state = torch.from_numpy(state).float().unsqueeze(0)
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([i[0] for i in minibatch])
        actions = torch.LongTensor([i[1] for i in minibatch])
        rewards = torch.FloatTensor([i[2] for i in minibatch])
        next_states = torch.FloatTensor([i[3] for i in minibatch])
        dones = torch.FloatTensor([i[4] for i in minibatch])

        q_targets_next = self.model(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        q_expected = self.model(states).gather(1, actions.unsqueeze(1))

        loss = self.criterion(q_expected, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def get_reward(state):
    if state[0] >= 0.5:
        print("Car has reached the final state")
        return 10
    if state[0] > -0.4:
        return (1+state[0])**2
    return 0

def train_dqn(episode, save_path='dqn_model.path):
    loss = []
    agent = DQNAgent(env.action_space.n, env.observation_space.shape[0])
    for e in range(episode):
        state = env.reset()
        score = 0
        max_steps = 1000
        for i in range(max_steps):
            action = agent.act(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            reward = get_reward(next_state)
            score += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)

    # Save the trained model
    torch.save(agent.model.state_dict(), save_path)
    print("Trained model saved at", save_path)

    return loss

def random_policy(episode, step):
    for i_episode in range(episode):
        env.reset()
        for t in range(step):
            env.render()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
            print("Starting next episode")

if __name__ == '__main__':
    print(env.observation_space)
    print(env.action_space)
    episodes = 100
    loss = train_dqn(episodes)
    plt.plot([i+1 for i in range(episodes)], loss)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('MountainCar-v0 with DQN')
    plt.show()
