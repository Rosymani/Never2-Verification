import gym
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
env.reset()

# Lists to store measurements for each episode
episode_rewards = []
avg_rewards_per_step_list = []
num_steps_list = []
exploration_exploitation_ratio_list = []

for episode in range(40):
    state = env.reset()
    done = False
    total_reward = 0
    actions = []

    while not done:
        env.render()
        action = env.action_space.sample()
        actions.append(action)

        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state

    episode_rewards.append(total_reward)
    num_steps = len(actions)
    num_steps_list.append(num_steps)

    avg_reward_per_step = total_reward / env.spec.max_episode_steps
    avg_rewards_per_step_list.append(avg_reward_per_step)
    
    exploration_count = sum(action == env.action_space.sample() for action in actions)
    exploitation_count = num_steps - exploration_count
    exploration_exploitation_ratio = exploration_count / exploitation_count if exploitation_count != 0 else float('inf')
    exploration_exploitation_ratio_list.append(exploration_exploitation_ratio)

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")
    print(f"Episode {episode + 1}, Average Reward per Step: {avg_reward_per_step}")
    print(f"Episode {episode + 1}, Number of Steps: {num_steps}")
    print(f"Episode {episode + 1}, Exploration vs. Exploitation Ratio: {exploration_exploitation_ratio}")

    print(f"Episode {episode + 1} completed in {num_steps} steps.")
    if info.get('is_success'):
        print("Goal achieved!")

env.close()

# Plotting the total rewards over episodes
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')

plt.subplot(1, 2, 2)
plt.plot(avg_rewards_per_step_list)
plt.xlabel('Episode')
plt.ylabel('Average Reward per Step')
plt.title('Average Reward per Step per Episode')

plt.tight_layout()
plt.show()

# Second Plot: Number of Steps and Exploration vs. Exploitation Ratio

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 3)
plt.plot(num_steps_list)
plt.xlabel('Episode')
plt.ylabel('Number of Steps')
plt.title('Number of Steps per Episode')

plt.subplot(2, 2, 4)
plt.plot(exploration_exploitation_ratio_list)
plt.xlabel('Episode')
plt.ylabel('Exploration vs. Exploitation Ratio')
plt.title('Exploration vs. Exploitation Ratio per Episode')

plt.tight_layout()
plt.show()

