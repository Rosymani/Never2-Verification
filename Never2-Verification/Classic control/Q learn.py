import gym
import numpy as np

env = gym.make('MountainCar-v0')
env.reset()

# Q-learning parameters
alpha = 0.3
gamma = 0.7
epsilon = 0.2
num_episodes = 400

# Initialize Q table

num_states = (env.observation_space.high - env.observation_space.low)*\
             np.array([10, 100])
num_states = np.round(num_states, 0).astype(int) + 1
Q_table = np.random.uniform(low=-1, high=1, 
                            size=(num_states[0], num_states[1], 
                                  env.action_space.n))

# Convert observation to state index

def get_state(observation):
    state = (observation - env.observation_space.low)*\
            np.array([10, 100])
    state = np.round(state, 0).astype(int)
    return state

# Q-learning algorithm

def reward_function(state, action, reward, next_state, done, t):
    # Add penalty for taking too long
    if done and t < 200:
        reward -= 100
    return reward
   
    reward = reward_function(state, action, reward, next_state, done, t)

for episode in range(num_episodes):
    state = get_state(env.reset())
    done = False
    
    while not done:
        # Choose action with epsilon-greedy strategy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[state[0], state[1], :])

        # Take action and observe next state and reward
        
        next_state, reward, done, _ = env.step(action)
        next_state = get_state(next_state)
        
        # Modify reward with custom reward function
        

        # Update Q table
        
        Q_table[state[0], state[1], action] = (1 - alpha) * Q_table[state[0], state[1], action] \
                                              + alpha * (reward + gamma * np.max(Q_table[next_state[0], next_state[1], :]))
        
         
        # Update state
        
        state = next_state

# Evaluate agent

state = get_state(env.reset())
done = False
while not done:
    action = np.argmax(Q_table[state[0], state[1], :])
    next_state, reward, done, _ = env.step(action)
    next_state = get_state(next_state)
    state = next_state
    env.render()

env.close()


